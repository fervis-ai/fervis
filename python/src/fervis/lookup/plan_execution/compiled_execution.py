"""Compiled execution artifact for deterministic fact-plan execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lineage.enums import (
    ContributionOrigin,
    ProofEdgeRole,
    ProofNodeKind,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.plan_execution.relations import (
    CompletenessSourceKind,
    RelationRows,
)
from fervis.lookup.plan_execution.value_compiler import (
    CompiledValueUses,
    compile_value_uses,
)
from fervis.lookup.fact_plan.fact_plan import AnswerPlan
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
    RenderScalarOutput,
)
from fervis.lookup.fact_plan.row_sources import RowSourceCatalog
from fervis.lookup.fact_plan.values import FactValue


@dataclass(frozen=True)
class ExecutionProofNode:
    id: str
    kind: ProofNodeKind
    proof_refs: tuple[str, ...] = ()
    label: str = ""
    value: Any = None
    operator: str = ""


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

    def refs_for_kind(self, kind: ProofNodeKind) -> tuple[str, ...]:
        return _dedupe_refs(
            tuple(
                ref
                for node in self.nodes
                if node.kind == kind
                for ref in node.proof_refs
            )
        )

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
class CompiledFactExecution:
    answer: AnswerPlan
    row_sources: RowSourceCatalog
    value_uses: CompiledValueUses
    proof_graph: ExecutionProofGraph

    @property
    def proof_node_refs_by_render_output_id(self) -> dict[str, tuple[str, ...]]:
        return {
            fulfillment.render_output_id: (_answer_output_node_id(fulfillment),)
            for fulfillment in self.answer.fulfillment
        }

    @property
    def endpoint_arg_scope_refs(self) -> dict[str, frozenset[str]]:
        refs_by_relation: dict[str, set[str]] = {}
        for endpoint_arg in self.value_uses.endpoint_args:
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

    @property
    def row_filter_scope_refs(self) -> dict[str, frozenset[str]]:
        refs_by_relation: dict[str, set[str]] = {}
        for row_filter in self.value_uses.row_filters:
            refs_by_relation.setdefault(row_filter.relation_id, set()).update(
                _dedupe_refs(tuple(row_filter.proof_refs))
            )
        return {
            relation_id: frozenset(refs)
            for relation_id, refs in refs_by_relation.items()
        }


def compile_fact_execution(
    *,
    answer: AnswerPlan,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    available_values: tuple[FactValue, ...] = (),
    available_value_uses: tuple[Any, ...] = (),
) -> CompiledFactExecution:
    value_uses = compile_value_uses(
        values=(*answer.values, *available_values),
        value_uses=answer.value_uses,
        catalog=catalog or RelationCatalog(),
        relations=answer.relations,
        row_sources=row_sources,
        grounded_input_uses=available_value_uses,
    )
    return CompiledFactExecution(
        answer=answer,
        row_sources=row_sources,
        value_uses=value_uses,
        proof_graph=_execution_proof_graph(
            answer,
            render_spec=answer.render_spec,
            value_uses=value_uses,
            values=(*answer.values, *available_values),
        ),
    )


def _execution_proof_graph(
    answer: AnswerPlan,
    *,
    render_spec: object,
    value_uses: CompiledValueUses,
    values: tuple[FactValue, ...],
) -> ExecutionProofGraph:
    nodes: list[ExecutionProofNode] = []
    edges: list[ExecutionProofEdge] = []
    contributions: list[ExecutionProofContribution] = []
    explicit_labels_by_ref = _explicit_labels_by_proof_ref(values)
    relations = answer.relations
    for arg in value_uses.endpoint_args:
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
    for row_filter in value_uses.row_filters:
        node_id = f"row_filter:{row_filter.relation_id}:{row_filter.field_id}"
        nodes.append(
            ExecutionProofNode(
                id=node_id,
                kind=ProofNodeKind.ROW_FILTER,
                proof_refs=tuple(row_filter.proof_refs),
                label=_filter_label(
                    row_filter.field_id,
                    operator=row_filter.operator.value,
                    value=row_filter.value,
                ),
                value=row_filter.value,
                operator=row_filter.operator.value,
            )
        )
        contributions.extend(
            _node_contributions(
                node_id=node_id,
                label=_filter_label(
                    row_filter.field_id,
                    operator=row_filter.operator.value,
                    value=row_filter.value,
                ),
                proof_refs=row_filter.proof_refs,
                explicit_labels_by_ref=explicit_labels_by_ref,
            )
        )
        edges.append(
            ExecutionProofEdge(
                source=node_id,
                target=f"relation:{row_filter.relation_id}",
                role=ProofEdgeRole.NARROWS,
            )
        )
    for choice in value_uses.population_choices:
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
                    "included_values": list(choice.included_values),
                    "excluded_values": list(choice.excluded_values),
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
    for scalar in value_uses.scalar_inputs:
        node_id = f"scalar_input:{scalar.operation_id}:{scalar.input_id}"
        nodes.append(
            ExecutionProofNode(
                id=node_id,
                kind=ProofNodeKind.OPERATION_INPUT,
                proof_refs=tuple(scalar.proof_refs),
                label=_assignment_label(scalar.input_id, scalar.value),
                value=scalar.value,
            )
        )
        contributions.extend(
            _node_contributions(
                node_id=node_id,
                label=_assignment_label(scalar.input_id, scalar.value),
                proof_refs=_dedupe_refs((*scalar.source_refs, *scalar.proof_refs)),
                explicit_labels_by_ref=explicit_labels_by_ref,
            )
        )
        edges.append(
            ExecutionProofEdge(
                source=node_id,
                target=f"operation:{scalar.operation_id}",
                role=ProofEdgeRole.INPUT,
            )
        )
    for rank_limit in value_uses.rank_limits:
        node_id = f"rank_limit:{rank_limit.operation_id}"
        nodes.append(
            ExecutionProofNode(
                id=node_id,
                kind=ProofNodeKind.OPERATION_INPUT,
                proof_refs=tuple(rank_limit.proof_refs),
                label=_assignment_label("rank_limit", rank_limit.value),
                value=rank_limit.value,
            )
        )
        contributions.extend(
            _node_contributions(
                node_id=node_id,
                label=_assignment_label("rank_limit", rank_limit.value),
                proof_refs=rank_limit.proof_refs,
                explicit_labels_by_ref=explicit_labels_by_ref,
            )
        )
        edges.append(
            ExecutionProofEdge(
                source=node_id,
                target=f"operation:{rank_limit.operation_id}",
                role=ProofEdgeRole.RANK_LIMIT,
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
        for input_relation in _operation_input_relation_ids(operation.spec):
            edges.append(
                ExecutionProofEdge(
                    source=f"relation:{input_relation}",
                    target=operation_node_id,
                    role=ProofEdgeRole.INPUT,
                )
            )
        output_relation = str(getattr(operation, "output_relation", "") or "")
        if output_relation:
            edges.append(
                ExecutionProofEdge(
                    source=operation_node_id,
                    target=f"relation:{output_relation}",
                    role=ProofEdgeRole.PRODUCES,
                )
            )
        output_scalar = str(getattr(operation.spec, "output_scalar", "") or "")
        if output_scalar:
            edges.append(
                ExecutionProofEdge(
                    source=operation_node_id,
                    target=f"scalar:{output_scalar}",
                    role=ProofEdgeRole.PRODUCES,
                )
            )
    for relation in relations:
        nodes.append(
            ExecutionProofNode(
                id=f"relation:{relation.id}",
                kind=ProofNodeKind.RELATION,
            )
        )
    for scalar_output in tuple(getattr(render_spec, "scalar_outputs", ()) or ()):
        nodes.append(
            ExecutionProofNode(
                id=f"scalar:{scalar_output.scalar_id}",
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


def _answer_output_nodes(answer: AnswerPlan) -> tuple[ExecutionProofNode, ...]:
    return tuple(
        ExecutionProofNode(
            id=_answer_output_node_id(fulfillment),
            kind=ProofNodeKind.ANSWER_OUTPUT,
        )
        for fulfillment in answer.fulfillment
    )


def _answer_output_edges(answer: AnswerPlan) -> tuple[ExecutionProofEdge, ...]:
    if answer.render_spec is None:
        return ()
    outputs_by_id = {
        output.id: output
        for output in (
            *answer.render_spec.relation_outputs,
            *answer.render_spec.scalar_outputs,
        )
    }
    edges: list[ExecutionProofEdge] = []
    for fulfillment in answer.fulfillment:
        output = outputs_by_id.get(fulfillment.render_output_id)
        if output is None:
            continue
        source_node_id = _render_output_source_node_id(output)
        if not source_node_id:
            continue
        edges.append(
            ExecutionProofEdge(
                source=source_node_id,
                target=_answer_output_node_id(fulfillment),
                role=ProofEdgeRole.PRODUCES,
            )
        )
    return tuple(edges)


def _answer_output_node_id(fulfillment: object) -> str:
    return (
        "answer_output:"
        f"{getattr(fulfillment, 'requested_fact_id')}:"
        f"{getattr(fulfillment, 'answer_output_id')}"
    )


def _operation_input_relation_ids(spec: object) -> tuple[str, ...]:
    relation_ids: list[str] = []
    for attr in ("input_relation", "left", "right"):
        value = str(getattr(spec, attr, "") or "")
        if value:
            relation_ids.append(value)
    for value in tuple(getattr(spec, "inputs", ()) or ()):
        relation_id = str(value or "")
        if relation_id:
            relation_ids.append(relation_id)
    for attr in (
        "candidate",
        "observed",
        "candidate_subject",
        "required_dimension",
        "observation",
    ):
        value = getattr(spec, attr, None)
        relation_id = str(getattr(value, "relation_id", "") or "")
        if relation_id:
            relation_ids.append(relation_id)
    return tuple(dict.fromkeys(relation_ids))


def _render_output_source_node_id(
    output: RenderRelationOutput | RenderScalarOutput,
) -> str:
    if isinstance(output, RenderRelationOutput):
        return f"relation:{output.relation_id}"
    if isinstance(output, RenderScalarOutput):
        return f"scalar:{output.scalar_id}"
    return ""


def _operation_id_from_node(node_id: str) -> str:
    parts = node_id.split(":")
    if len(parts) < 2:
        return ""
    return parts[1]


def _explicit_labels_by_proof_ref(values: tuple[FactValue, ...]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        label = value.label or value.id
        for proof_ref in value.proof_refs:
            if proof_ref.startswith("known_input:") and label:
                output[proof_ref] = label
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
    )


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))
