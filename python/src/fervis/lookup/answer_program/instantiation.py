"""Fail-closed instantiation of canonical answer programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fervis.lineage.enums import (
    ContributionOrigin,
    ProofEdgeRole,
    ProofNodeKind,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.declared_values import exact_positive_integer
from fervis.lookup.plan_execution.relations import (
    CompletenessSourceKind,
    RelationRows,
)
from fervis.lookup.answer_program.expression_instantiation import (
    InstantiatedProgramInputs,
    instantiate_program_expressions,
)
from fervis.lookup.answer_program.fact_materialization import (
    materialize_requested_facts,
)
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.fact_plan.row_sources.model import RowSourceCatalog
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    ParameterRef,
)
from fervis.lookup.answer_program.operations import (
    ComputeSpec,
    FilterSpec,
    RankSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.contracts import (
    AnswerProgramContractError,
)
from fervis.lookup.answer_program.inputs import (
    compile_answer_program_inputs,
    program_value_expressions,
    resolve_value_expression,
    resolved_value_expression_type,
)
from fervis.lookup.answer_program.compatibility import (
    verify_program_compatibility,
)
from fervis.lookup.answer_program.values import (
    BindingSet,
)
from fervis.lookup.answer_program.expressions import (
    Expression,
    expression_constant,
    expression_input_id,
    expression_references,
)
from fervis.lookup.question_contract import QuestionContract, RequestedFact
from fervis.lookup.question_contract import MembershipTestRef
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    ResolvedOperationInput,
    ResolvedRankSpec,
)

if TYPE_CHECKING:
    from fervis.lookup.plan_execution.authorized_sources import (
        AuthorizedExecutionSources,
    )
    from fervis.lookup.plan_execution.verification.contract_types import (
        PopulationCoverage,
    )


@dataclass(frozen=True)
class ExecutionEnvironment:
    catalog: RelationCatalog
    authorized_sources: AuthorizedExecutionSources | None = None
    catalog_selection: CatalogSelectionResult | None = None
    memory_relations: tuple[RelationRows, ...] = ()
    authority_ref: str = ""


@dataclass(frozen=True)
class ExecutionProofNode:
    id: str
    kind: ProofNodeKind
    proof_refs: tuple[str, ...] = ()
    label: str = ""
    value: Any = None
    operator: str = ""
    row_population_test_refs: tuple[MembershipTestRef, ...] = ()
    condition_test_refs: tuple[MembershipTestRef, ...] = ()


@dataclass(frozen=True)
class ExecutionProofEdge:
    source: str
    target: str
    role: ProofEdgeRole


@dataclass(frozen=True)
class ExecutionProofContribution:
    origin: ContributionOrigin
    label: str
    node_refs: tuple[str, ...]
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionProofGraph:
    nodes: tuple[ExecutionProofNode, ...] = ()
    edges: tuple[ExecutionProofEdge, ...] = ()
    contributions: tuple[ExecutionProofContribution, ...] = ()

    def with_executed_relations(
        self, relations: tuple[RelationRows, ...]
    ) -> "ExecutionProofGraph":
        proof_refs_by_node_id = {
            f"relation:{relation.id}": relation.completeness.proof_refs
            for relation in relations
            if relation.completeness.proof_refs
        }
        existing_node_ids = {node.id for node in self.nodes}
        missing_relation_nodes = tuple(
            ExecutionProofNode(
                id=f"relation:{relation.id}",
                kind=ProofNodeKind.RELATION,
                proof_refs=relation.completeness.proof_refs,
            )
            for relation in relations
            if f"relation:{relation.id}" not in existing_node_ids
        )
        if not proof_refs_by_node_id and not missing_relation_nodes:
            return self
        return ExecutionProofGraph(
            nodes=(
                *tuple(
                    _node_with_proof_refs(node, proof_refs_by_node_id.get(node.id, ()))
                    for node in self.nodes
                ),
                *missing_relation_nodes,
            ),
            edges=self.edges,
            contributions=_dedupe_contributions(
                (
                    *self.contributions,
                    *_executed_relation_contributions(relations),
                )
            ),
        )


@dataclass(frozen=True)
class _MaterializedExecution:
    answer: AnswerProgram
    bindings: BindingSet
    catalog: RelationCatalog
    row_sources: RowSourceCatalog
    instantiated_inputs: InstantiatedProgramInputs
    operations: tuple[ExecutableOperation, ...]
    authority_ref: str
    proof_graph: ExecutionProofGraph
    effective_requested_facts: tuple[RequestedFact, ...]
    operation_inputs: tuple[ResolvedOperationInput, ...] = ()

    @property
    def proof_node_refs_by_result_output_id(self) -> dict[str, tuple[str, ...]]:
        return {
            fulfillment.result_output_id: (_answer_output_node_id(fulfillment),)
            for fulfillment in self.answer.fulfillment
        }

    @property
    def endpoint_arg_scope_refs(self) -> dict[str, frozenset[str]]:
        refs_by_relation: dict[str, set[str]] = {}
        for endpoint_arg in self.instantiated_inputs.endpoint_args:
            refs_by_relation.setdefault(endpoint_arg.relation_id, set()).update(
                _dedupe_refs(tuple(endpoint_arg.proof_refs))
            )
        return {
            relation_id: frozenset(refs)
            for relation_id, refs in refs_by_relation.items()
        }

    @property
    def operation_proof_refs(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for node in self.proof_graph.nodes:
            if node.kind is not ProofNodeKind.OPERATION_INPUT:
                continue
            operation_id = _operation_id_from_node(node.id)
            if not operation_id:
                continue
            grouped.setdefault(operation_id, []).extend(node.proof_refs)
        return {
            operation_id: _dedupe_refs(tuple(refs))
            for operation_id, refs in grouped.items()
        }


@dataclass(frozen=True)
class VerifiedExecution(_MaterializedExecution):
    """An execution whose compatibility and executable contracts are proven."""


def instantiate_answer_program(
    program: AnswerProgram,
    bindings: BindingSet,
    environment: ExecutionEnvironment,
) -> VerifiedExecution:
    """Validate current pins and materialize an execution without reading sources."""

    if not program.fact_template:
        raise VerificationError("answer program requires persisted fact template")
    catalog = (
        environment.authorized_sources.relation_catalog
        if environment.authorized_sources is not None
        else environment.catalog
    )
    verify_program_compatibility(
        program,
        catalog=catalog,
        memory_relations=environment.memory_relations,
    )
    question_contract = QuestionContract(requested_facts=program.fact_template)
    from fervis.lookup.plan_execution.verification.answer_program import (
        _verify_answer_program_execution,
        _verify_answer_program_structure,
    )

    structured = _verify_answer_program_structure(
        program,
        compiled_inputs=compile_answer_program_inputs(
            program,
            bindings=bindings,
        ),
        question_contract=question_contract,
        catalog=catalog,
        memory_relations=environment.memory_relations,
        catalog_selection=environment.catalog_selection,
        authorized_sources=environment.authorized_sources,
    )
    materialized = _materialize_execution(
        answer=structured.program,
        bindings=structured.bindings,
        catalog=catalog,
        row_sources=structured.row_sources,
        authority_ref=environment.authority_ref,
    )

    _verify_answer_program_execution(
        structured,
        materialized=materialized,
        question_contract=QuestionContract(
            requested_facts=materialized.effective_requested_facts
        ),
        catalog=catalog,
        catalog_selection=environment.catalog_selection,
    )
    return VerifiedExecution(
        answer=materialized.answer,
        bindings=materialized.bindings,
        catalog=materialized.catalog,
        row_sources=materialized.row_sources,
        instantiated_inputs=materialized.instantiated_inputs,
        operations=materialized.operations,
        authority_ref=materialized.authority_ref,
        proof_graph=materialized.proof_graph,
        effective_requested_facts=materialized.effective_requested_facts,
        operation_inputs=materialized.operation_inputs,
    )


def _materialize_execution(
    *,
    answer: AnswerProgram,
    bindings: BindingSet,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    authority_ref: str = "",
) -> _MaterializedExecution:
    instantiated_inputs = instantiate_program_expressions(
        bindings=bindings,
        catalog=catalog or RelationCatalog(),
        relations=answer.relations,
        parameters=answer.parameters,
        row_sources=row_sources,
    )
    operations, operation_inputs = _instantiate_operations(
        answer,
        bindings=bindings,
    )
    proof_graph = _execution_proof_graph(
        answer,
        instantiated_inputs=instantiated_inputs,
        values=_program_input_values(answer, bindings=bindings),
        operation_inputs=operation_inputs,
    )
    proof_graph = _with_population_coverage(
        proof_graph,
        answer=answer,
        catalog=catalog,
        row_sources=row_sources,
        instantiated_inputs=instantiated_inputs,
        operation_inputs=operation_inputs,
    )
    effective_requested_facts = materialize_requested_facts(
        answer.fact_template,
        population_choices=instantiated_inputs.population_choices,
    )
    return _MaterializedExecution(
        answer=answer,
        bindings=bindings,
        catalog=catalog or RelationCatalog(),
        row_sources=row_sources,
        instantiated_inputs=instantiated_inputs,
        operations=operations,
        authority_ref=authority_ref,
        proof_graph=_require_valid_proof_graph(proof_graph),
        effective_requested_facts=effective_requested_facts,
        operation_inputs=operation_inputs,
    )


def _with_population_coverage(
    graph: ExecutionProofGraph,
    *,
    answer: AnswerProgram,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    instantiated_inputs: InstantiatedProgramInputs,
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> ExecutionProofGraph:
    from fervis.lookup.plan_execution.verification.contracts import (
        _relation_contracts,
        _scalar_contracts,
    )
    from fervis.lookup.plan_execution.verification.execution_proof import (
        ExecutionProofContext,
    )
    from fervis.lookup.plan_execution.verification.result_projection import (
        _result_output_proofs,
    )

    proof_context = ExecutionProofContext(
        endpoint_arg_scope_refs=_instantiated_endpoint_arg_refs(instantiated_inputs),
        operation_refs=_operation_input_refs(graph),
    )
    relation_contracts = _relation_contracts(
        answer,
        catalog=catalog,
        row_sources=row_sources,
        proof_context=proof_context,
    )
    scalar_contracts = _scalar_contracts(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    result_proofs = _result_output_proofs(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    coverage_by_node_id = {
        **{
            f"relation:{relation_id}": contract.population_proof.population_coverage
            for relation_id, contract in relation_contracts.items()
        },
        **{
            f"scalar:{scalar_id}": contract.proof.population_coverage
            for scalar_id, contract in scalar_contracts.items()
        },
        **{
            _answer_output_node_id(fulfillment): result_proofs[
                fulfillment.result_output_id
            ].population_coverage
            for fulfillment in answer.fulfillment
            if fulfillment.result_output_id in result_proofs
        },
    }
    return ExecutionProofGraph(
        nodes=tuple(
            _node_with_population_coverage(
                node,
                coverage_by_node_id.get(node.id),
            )
            for node in graph.nodes
        ),
        edges=graph.edges,
        contributions=graph.contributions,
    )


def _instantiated_endpoint_arg_refs(
    inputs: InstantiatedProgramInputs,
) -> dict[str, frozenset[str]]:
    refs: dict[str, set[str]] = {}
    for item in inputs.endpoint_args:
        refs.setdefault(item.relation_id, set()).update(item.proof_refs)
    return {relation_id: frozenset(items) for relation_id, items in refs.items()}


def _operation_input_refs(
    graph: ExecutionProofGraph,
) -> dict[str, frozenset[str]]:
    refs: dict[str, set[str]] = {}
    for node in graph.nodes:
        if node.kind is not ProofNodeKind.OPERATION_INPUT:
            continue
        operation_id = _operation_id_from_node(node.id)
        if operation_id:
            refs.setdefault(operation_id, set()).update(node.proof_refs)
    return {operation_id: frozenset(items) for operation_id, items in refs.items()}


def _instantiate_operations(
    answer: AnswerProgram,
    *,
    bindings: BindingSet,
) -> tuple[tuple[ExecutableOperation, ...], tuple[ResolvedOperationInput, ...]]:
    operations: list[ExecutableOperation] = []
    inputs: list[ResolvedOperationInput] = []
    for operation in answer.operations:
        spec = operation.spec
        if isinstance(spec, ComputeSpec):
            resolved_inputs = _resolve_expression_inputs(
                spec.expression,
                bindings=bindings,
                operation_id=operation.id,
            )
            operations.append(
                ExecutableOperation(
                    id=operation.id,
                    spec=spec,
                    output_relation=operation.output_relation,
                )
            )
            inputs.extend(resolved_inputs)
            continue
        if isinstance(spec, (FilterSpec, UniversalConditionSpec)):
            for expression in (
                spec.predicate.left,
                *((spec.predicate.right,) if spec.predicate.right is not None else ()),
            ):
                inputs.extend(
                    _resolve_expression_inputs(
                        expression,
                        bindings=bindings,
                        operation_id=operation.id,
                    )
                )
        if not isinstance(spec, RankSpec):
            operations.append(
                ExecutableOperation(
                    id=operation.id,
                    spec=spec,
                    output_relation=operation.output_relation,
                )
            )
            continue
        try:
            resolved_limit = resolve_value_expression(
                spec.limit,
                bindings=bindings,
            )
        except AnswerProgramContractError as exc:
            raise VerificationError(f"{exc.code}: {exc}") from exc
        try:
            limit = exact_positive_integer(resolved_limit.value)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if "positive integer" in message:
                raise VerificationError(
                    "rank limit must be a positive integer"
                ) from exc
            raise VerificationError("rank limit must be numeric") from exc
        operations.append(
            ExecutableOperation(
                id=operation.id,
                spec=ResolvedRankSpec(
                    input_relation=spec.input_relation,
                    order_by=spec.order_by,
                    tie_policy=spec.tie_policy,
                    limit=limit,
                    tie_breakers=spec.tie_breakers,
                ),
                output_relation=operation.output_relation,
            )
        )
        inputs.append(
            ResolvedOperationInput(
                operation_id=operation.id,
                input_id="rank_limit",
                value=limit,
                proof_refs=resolved_limit.proof_refs,
            )
        )
    return tuple(operations), _dedupe_operation_inputs(inputs)


def _dedupe_operation_inputs(
    inputs: list[ResolvedOperationInput],
) -> tuple[ResolvedOperationInput, ...]:
    by_identity: dict[tuple[str, str], ResolvedOperationInput] = {}
    for item in inputs:
        identity = (item.operation_id, item.input_id)
        existing = by_identity.get(identity)
        if existing is not None and existing != item:
            raise VerificationError("conflicting operation input identity")
        by_identity[identity] = item
    return tuple(by_identity.values())


def _resolve_expression_inputs(
    expression: Expression,
    *,
    bindings: BindingSet,
    operation_id: str,
) -> tuple[ResolvedOperationInput, ...]:
    references = expression_references(expression)
    return (
        *(
            _resolve_expression_value(
                item,
                bindings=bindings,
                operation_id=operation_id,
            )
            for item in references.parameters
        ),
        *(
            _resolve_expression_value(
                item,
                bindings=bindings,
                operation_id=operation_id,
            )
            for item in references.constants
        ),
    )


def _resolve_expression_value(
    expression: ParameterRef | ConstantRef,
    *,
    bindings: BindingSet,
    operation_id: str,
) -> ResolvedOperationInput:
    resolved = resolve_value_expression(expression, bindings=bindings)
    proof_refs = _dedupe_refs((*resolved.source_refs, *resolved.proof_refs))
    return ResolvedOperationInput(
        operation_id=operation_id,
        input_id=expression_input_id(expression),
        value=resolved.value,
        value_type=resolved_value_expression_type(expression, resolved),
        proof_refs=proof_refs,
    )


def _program_input_values(
    answer: AnswerProgram,
    *,
    bindings: BindingSet,
) -> tuple[FactValue, ...]:
    values = [binding.value for binding in bindings.bindings]
    values.extend(
        constant.value
        for named in program_value_expressions(answer)
        if (constant := expression_constant(named.expression)) is not None
    )
    return tuple({value.id: value for value in values}.values())


def _require_valid_proof_graph(
    proof_graph: ExecutionProofGraph,
) -> ExecutionProofGraph:
    node_ids = [node.id for node in proof_graph.nodes]
    duplicate_node_ids = tuple(
        node_id for node_id in dict.fromkeys(node_ids) if node_ids.count(node_id) > 1
    )
    if duplicate_node_ids:
        raise VerificationError(
            "duplicate proof graph node ids: " + ", ".join(duplicate_node_ids)
        )
    known_node_ids = set(node_ids)
    missing_edge_endpoints = tuple(
        endpoint
        for edge in proof_graph.edges
        for endpoint in (edge.source, edge.target)
        if endpoint not in known_node_ids
    )
    if missing_edge_endpoints:
        raise VerificationError(
            "proof graph edges reference missing nodes: "
            + ", ".join(dict.fromkeys(missing_edge_endpoints))
        )
    return proof_graph


def _execution_proof_graph(
    answer: AnswerProgram,
    *,
    instantiated_inputs: InstantiatedProgramInputs,
    values: tuple[FactValue, ...],
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> ExecutionProofGraph:
    nodes: list[ExecutionProofNode] = []
    edges: list[ExecutionProofEdge] = []
    contributions: list[ExecutionProofContribution] = []
    explicit_labels_by_ref = _explicit_labels_by_proof_ref(answer.fact_template)
    relations = answer.relations
    for arg in instantiated_inputs.endpoint_args:
        node_id = f"endpoint_arg:{arg.relation_id}:{arg.param_ref}"
        nodes.append(
            ExecutionProofNode(
                id=node_id,
                kind=ProofNodeKind.ENDPOINT_ARG,
                proof_refs=tuple(arg.proof_refs),
                label=_assignment_label(arg.param_ref, arg.value),
                value=arg.value,
            )
        )
        contributions.extend(
            _node_contributions(
                node_id=node_id,
                label=_assignment_label(arg.param_ref, arg.value),
                proof_refs=arg.proof_refs,
                explicit_labels_by_ref=explicit_labels_by_ref,
            )
        )
        edges.append(
            ExecutionProofEdge(
                source=node_id,
                target=f"relation:{arg.relation_id}",
                role=ProofEdgeRole.SCOPES,
            )
        )
    for choice in instantiated_inputs.population_choices:
        node_id = (
            "population_choice:"
            f"{choice.relation_id}:"
            f"{choice.controller_kind.value}:"
            f"{choice.controller_id}"
        )
        label = _population_choice_label(
            choice.field_id,
            included_values=choice.included_values,
            excluded_values=choice.excluded_values,
        )
        nodes.append(
            ExecutionProofNode(
                id=node_id,
                kind=ProofNodeKind.POPULATION_CHOICE,
                proof_refs=tuple(choice.proof_refs),
                label=label,
                value={
                    "requested_fact_ids": list(choice.requested_fact_ids),
                    "semantic_control_ref": choice.semantic_control_ref,
                    "included_values": list(choice.included_values),
                    "excluded_values": list(choice.excluded_values),
                    "review_scope_decisions": [
                        {
                            "membership_test_id": decision.membership_test_id,
                            "decision": decision.decision.value,
                            "axis_kind": decision.axis_kind,
                            "axis_id": decision.axis_id,
                            "owner_surface_ids": list(decision.owner_surface_ids),
                            "proof_refs": list(decision.proof_refs),
                        }
                        for decision in choice.review_scope_decisions
                    ],
                },
            )
        )
        contributions.append(
            ExecutionProofContribution(
                origin=ContributionOrigin.CONTEXTUAL,
                label=label,
                node_refs=(node_id,),
                proof_refs=tuple(choice.proof_refs),
            )
        )
        edges.append(
            ExecutionProofEdge(
                source=node_id,
                target=f"relation:{choice.relation_id}",
                role=_population_choice_edge_role(choice.excluded_values),
            )
        )
    for operation_input in operation_inputs:
        node_id = (
            f"operation_input:{operation_input.operation_id}:{operation_input.input_id}"
        )
        nodes.append(
            ExecutionProofNode(
                id=node_id,
                kind=ProofNodeKind.OPERATION_INPUT,
                proof_refs=tuple(operation_input.proof_refs),
                label=_assignment_label(
                    operation_input.input_id,
                    operation_input.value,
                ),
                value=operation_input.value,
            )
        )
        contributions.extend(
            _node_contributions(
                node_id=node_id,
                label=_assignment_label(
                    operation_input.input_id,
                    operation_input.value,
                ),
                proof_refs=operation_input.proof_refs,
                explicit_labels_by_ref=explicit_labels_by_ref,
            )
        )
        edges.append(
            ExecutionProofEdge(
                source=node_id,
                target=f"operation:{operation_input.operation_id}",
                role=(
                    ProofEdgeRole.RANK_LIMIT
                    if operation_input.input_id == "rank_limit"
                    else ProofEdgeRole.INPUT
                ),
            )
        )
    for operation in answer.operations:
        operation_node_id = f"operation:{operation.id}"
        nodes.append(
            ExecutionProofNode(
                id=operation_node_id,
                kind=ProofNodeKind.OPERATION,
            )
        )
        for input_relation in operation.input_relation_ids:
            edges.append(
                ExecutionProofEdge(
                    source=f"relation:{input_relation}",
                    target=operation_node_id,
                    role=ProofEdgeRole.INPUT,
                )
            )
        output_relation = operation.output_relation
        if output_relation:
            edges.append(
                ExecutionProofEdge(
                    source=operation_node_id,
                    target=f"relation:{output_relation}",
                    role=ProofEdgeRole.PRODUCES,
                )
            )
        output_scalar = operation.output_scalar
        if output_scalar:
            edges.append(
                ExecutionProofEdge(
                    source=operation_node_id,
                    target=f"scalar:{output_scalar}",
                    role=ProofEdgeRole.PRODUCES,
                )
            )
    relation_node_ids = tuple(
        dict.fromkeys(
            (
                *(relation.id for relation in relations),
                *(
                    output_relation
                    for operation in answer.operations
                    for output_relation in (operation.output_relation,)
                    if output_relation
                ),
            )
        )
    )
    for relation_id in relation_node_ids:
        nodes.append(
            ExecutionProofNode(
                id=f"relation:{relation_id}",
                kind=ProofNodeKind.RELATION,
            )
        )
    scalar_node_ids = tuple(
        dict.fromkeys(
            (
                *(
                    output_scalar
                    for operation in answer.operations
                    for output_scalar in (operation.output_scalar,)
                    if output_scalar
                ),
                *(
                    scalar_output.scalar_id
                    for scalar_output in (
                        answer.result_projection.scalar_outputs
                        if answer.result_projection is not None
                        else ()
                    )
                ),
            )
        )
    )
    for scalar_id in scalar_node_ids:
        nodes.append(
            ExecutionProofNode(
                id=f"scalar:{scalar_id}",
                kind=ProofNodeKind.SCALAR,
            )
        )
    nodes.extend(_answer_output_nodes(answer))
    edges.extend(_answer_output_edges(answer))
    return ExecutionProofGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        contributions=_dedupe_contributions(tuple(contributions)),
    )


def _answer_output_nodes(answer: AnswerProgram) -> tuple[ExecutionProofNode, ...]:
    return tuple(
        ExecutionProofNode(
            id=node_id,
            kind=ProofNodeKind.ANSWER_OUTPUT,
        )
        for node_id in dict.fromkeys(
            _answer_output_node_id(fulfillment) for fulfillment in answer.fulfillment
        )
    )


def _answer_output_edges(answer: AnswerProgram) -> tuple[ExecutionProofEdge, ...]:
    source_node_by_output_id = {
        output.id: output.source_node_id
        for output in answer.result_projection.relation_outputs
    }
    source_node_by_output_id.update(
        {
            output.id: output.source_node_id
            for output in answer.result_projection.scalar_outputs
        }
    )
    edges: list[ExecutionProofEdge] = []
    for fulfillment in answer.fulfillment:
        source_node_id = source_node_by_output_id.get(fulfillment.result_output_id)
        if source_node_id is None:
            continue
        edges.append(
            ExecutionProofEdge(
                source=source_node_id,
                target=_answer_output_node_id(fulfillment),
                role=ProofEdgeRole.PRODUCES,
            )
        )
    return tuple(edges)


def _answer_output_node_id(fulfillment: FactFulfillment) -> str:
    return (
        f"answer_output:{fulfillment.requested_fact_id}:{fulfillment.answer_output_id}"
    )


def _operation_id_from_node(node_id: str) -> str:
    parts = node_id.split(":")
    if len(parts) < 2:
        return ""
    return parts[1]


def _explicit_labels_by_proof_ref(
    requested_facts: tuple[RequestedFact, ...],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for fact in requested_facts:
        for known_input in fact.known_inputs:
            output.setdefault(f"known_input:{known_input.id}", known_input.text)
    return output


def _node_contributions(
    *,
    node_id: str,
    label: str,
    proof_refs: tuple[str, ...],
    explicit_labels_by_ref: dict[str, str],
) -> tuple[ExecutionProofContribution, ...]:
    contributions: list[ExecutionProofContribution] = []
    for proof_ref in proof_refs:
        explicit_label = explicit_labels_by_ref.get(proof_ref)
        if explicit_label:
            contributions.append(
                ExecutionProofContribution(
                    origin=ContributionOrigin.EXPLICIT,
                    label=explicit_label,
                    node_refs=(node_id,),
                    proof_refs=(proof_ref,),
                )
            )
    contributions.append(
        ExecutionProofContribution(
            origin=_applied_origin(proof_refs),
            label=label,
            node_refs=(node_id,),
            proof_refs=tuple(proof_refs),
        )
    )
    return tuple(contributions)


def _executed_relation_contributions(
    relations: tuple[RelationRows, ...],
) -> tuple[ExecutionProofContribution, ...]:
    return tuple(
        ExecutionProofContribution(
            origin=ContributionOrigin.CONTEXTUAL,
            label=relation.id,
            node_refs=(f"relation:{relation.id}",),
            proof_refs=tuple(relation.completeness.proof_refs),
        )
        for relation in relations
        if relation.completeness.source_kind is CompletenessSourceKind.MEMORY_READ
        and relation.completeness.proof_refs
    )


def _applied_origin(proof_refs: tuple[str, ...]) -> ContributionOrigin:
    if any(ref.startswith("known_input:") for ref in proof_refs):
        return ContributionOrigin.DERIVED
    return ContributionOrigin.CONTEXTUAL


def _assignment_label(name: str, value: object) -> str:
    return f"{_short_ref(name)}={_render_value(value)}"


def _filter_label(field_id: str, *, operator: str, value: object) -> str:
    if operator == "equals":
        return _assignment_label(field_id, value)
    return f"{_short_ref(field_id)} {operator} {_render_value(value)}"


def _population_choice_label(
    field_id: str,
    *,
    included_values: tuple[str, ...],
    excluded_values: tuple[str, ...],
) -> str:
    label = f"Included {_short_ref(field_id)} values [{_render_value(included_values)}]"
    if excluded_values:
        return f"{label}. Excluded: {_render_value(excluded_values)}"
    return label


def _population_choice_edge_role(
    excluded_values: tuple[str, ...],
) -> ProofEdgeRole:
    if excluded_values:
        return ProofEdgeRole.NARROWS
    return ProofEdgeRole.SCOPES


def _short_ref(value: str) -> str:
    return value.rsplit(".", 1)[-1].rsplit(":", 1)[-1]


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ", ".join(_render_value(item) for item in value)
    return str(value)


def _dedupe_contributions(
    contributions: tuple[ExecutionProofContribution, ...],
) -> tuple[ExecutionProofContribution, ...]:
    output: list[ExecutionProofContribution] = []
    seen: set[tuple[ContributionOrigin, str, tuple[str, ...], tuple[str, ...]]] = set()
    for item in contributions:
        key = (item.origin, item.label, item.node_refs, item.proof_refs)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return tuple(output)


def _node_with_proof_refs(
    node: ExecutionProofNode,
    proof_refs: tuple[str, ...],
) -> ExecutionProofNode:
    if not proof_refs:
        return node
    return ExecutionProofNode(
        id=node.id,
        kind=node.kind,
        proof_refs=_dedupe_refs((*proof_refs, *node.proof_refs)),
        label=node.label,
        value=node.value,
        operator=node.operator,
        row_population_test_refs=node.row_population_test_refs,
        condition_test_refs=node.condition_test_refs,
    )


def _node_with_population_coverage(
    node: ExecutionProofNode,
    coverage: PopulationCoverage | None,
) -> ExecutionProofNode:
    if coverage is None:
        return node
    return ExecutionProofNode(
        id=node.id,
        kind=node.kind,
        proof_refs=node.proof_refs,
        label=node.label,
        value=node.value,
        operator=node.operator,
        row_population_test_refs=tuple(
            sorted(
                coverage.row_tests,
                key=lambda ref: (
                    ref.requested_fact_id,
                    ref.membership_test_id,
                ),
            )
        ),
        condition_test_refs=tuple(
            sorted(
                coverage.condition_tests,
                key=lambda ref: (
                    ref.requested_fact_id,
                    ref.membership_test_id,
                ),
            )
        ),
    )


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))
