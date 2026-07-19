"""Relation contract construction for fact-plan verification."""

from ._shared import (
    AnswerProgram,
    Relation,
    RelationCatalog,
    RowSourceCatalog,
    SourceKind,
    read_evidence_ref,
    read_field_evidence_ref,
    row_source_evidence_ref,
    row_source_field_evidence_ref,
)
from .contract_types import (
    PopulationCoverage,
    ProofLineage,
    RelationContract,
    RelationEntityKey,
    RelationEntityKeyComponent,
    ScalarContract,
)
from fervis.lookup.answer_program.relations import PopulationCoverageRole
from .execution_proof import ExecutionProofContext
from .operation_contracts import _operation_relation_contract
from fervis.lookup.answer_program.operations import (
    ComputeSpec,
    Operation,
)
from fervis.lookup.answer_program.expressions import fold_expression
from fervis.lookup.answer_program.expressions import expression_input_id
from fervis.lookup.answer_program.values import NodeOutputRef
from fervis.lookup.plan_execution.operation_runtime import ResolvedOperationInput
from .sources import _row_source_for_relation, _source_mechanic_proof_refs


def _relation_contracts(
    answer: AnswerProgram,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    proof_context: ExecutionProofContext,
) -> dict[str, RelationContract]:
    contracts = {
        relation.id: _base_relation_contract(
            relation,
            catalog=catalog,
            row_sources=row_sources,
            proof_context=proof_context,
        )
        for relation in answer.relations
    }
    for operation in answer.operations:
        if not operation.output_relation:
            continue
        contracts[operation.output_relation] = _operation_relation_contract(
            operation,
            contracts,
            proof_context=proof_context,
        )
    return contracts


def _scalar_contracts(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> dict[str, ScalarContract]:
    """Fold scalar proof through the existing compute-expression graph."""

    inputs = _declared_scalar_input_contracts(answer)
    for item in operation_inputs:
        key = (item.operation_id, item.input_id)
        declared = inputs.get(key, ScalarContract())
        inputs[key] = ScalarContract(
            proof=ProofLineage(
                value_refs=frozenset({*declared.proof.value_refs, *item.proof_refs}),
                population_coverage=declared.proof.population_coverage,
            ),
            population_derived=declared.population_derived,
        )
    operations = {operation.id: operation for operation in answer.operations}
    scalars: dict[str, ScalarContract] = {}

    for operation in answer.operations:
        spec = operation.spec
        if not isinstance(spec, ComputeSpec):
            continue
        proof = fold_expression(
            spec.expression,
            field=lambda _item: ScalarContract(),
            parameter=lambda item: _scalar_input_contract(
                inputs,
                operation_id=operation.id,
                input_id=expression_input_id(item),
            ),
            constant=lambda item: _scalar_input_contract(
                inputs,
                operation_id=operation.id,
                input_id=expression_input_id(item),
            ),
            environment=lambda _item: ScalarContract(),
            output=lambda item: _node_output_contract(
                item,
                operations=operations,
                relation_contracts=relation_contracts,
                scalar_contracts=scalars,
            ),
            unary=lambda _item, operand: operand,
            binary=lambda _item, left, right: left.combine(right),
            function=lambda _item, arguments: _combine_scalar_contracts(arguments),
        )
        scalars[spec.output_scalar] = proof
    return scalars


def _combine_scalar_contracts(
    contracts: tuple[ScalarContract, ...],
) -> ScalarContract:
    output = ScalarContract()
    for contract in contracts:
        output = output.combine(contract)
    return output


def _declared_scalar_input_contracts(
    answer: AnswerProgram,
) -> dict[tuple[str, str], ScalarContract]:
    contracts: dict[tuple[str, str], ScalarContract] = {}
    for operation in answer.operations:
        spec = operation.spec
        if not isinstance(spec, ComputeSpec):
            continue
        for input_coverage in spec.input_population_coverage:
            claims = input_coverage.claims
            contracts[(operation.id, input_coverage.input_id)] = ScalarContract(
                proof=ProofLineage(
                    value_refs=frozenset(
                        proof_ref for claim in claims for proof_ref in claim.proof_refs
                    ),
                    population_coverage=PopulationCoverage(
                        row_tests=frozenset(
                            claim.test_ref
                            for claim in claims
                            if claim.role is PopulationCoverageRole.ROW_POPULATION
                        ),
                        condition_tests=frozenset(
                            claim.test_ref
                            for claim in claims
                            if claim.role is PopulationCoverageRole.OPERATION_CONDITION
                        ),
                    ),
                ),
                population_derived=True,
            )
    return contracts


def _scalar_input_contract(
    inputs: dict[tuple[str, str], ScalarContract],
    *,
    operation_id: str,
    input_id: str,
) -> ScalarContract:
    return inputs.get((operation_id, input_id), ScalarContract())


def _node_output_contract(
    ref: NodeOutputRef,
    *,
    operations: dict[str, Operation],
    relation_contracts: dict[str, RelationContract],
    scalar_contracts: dict[str, ScalarContract],
) -> ScalarContract:
    operation = operations.get(ref.node_id)
    if operation is None:
        return ScalarContract()
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        if spec.output_scalar != ref.output_id:
            return ScalarContract()
        return scalar_contracts.get(ref.output_id, ScalarContract())
    output_relation = operation.output_relation
    relation = relation_contracts.get(output_relation)
    if relation is None:
        return ScalarContract()
    return ScalarContract(
        proof=relation.field_proofs.get(ref.output_id, ProofLineage()),
        population_derived=True,
    )


def _base_relation_contract(
    relation: Relation,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    proof_context: ExecutionProofContext,
) -> RelationContract:
    fields = {field.field_id: frozenset(field.roles) for field in relation.fields}
    population_proof = _relation_source_population_proof(
        relation,
        catalog=catalog,
        row_sources=row_sources,
        endpoint_arg_scope_refs=proof_context.endpoint_arg_scope_refs,
    )
    field_proofs = {
        field.field_id: _binding_proof(
            relation,
            field.field_id,
            catalog=catalog,
            row_sources=row_sources,
        ).merge(population_proof)
        for field in relation.fields
    }
    field_types: dict[str, str] = {}
    if relation.source.kind in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        try:
            row_source = _row_source_for_relation(relation, row_sources=row_sources)
            field_types = {
                field.field_id: row_source.field(field.field_id).type.value
                for field in relation.fields
            }
        except KeyError:
            field_types = {}
    return RelationContract(
        fields=fields,
        grain_keys=relation.grain_keys,
        field_proofs=field_proofs,
        field_types=field_types,
        entity_keys=_relation_entity_keys(relation, row_sources=row_sources),
        population_proof=population_proof,
    )


def _relation_entity_keys(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> tuple[RelationEntityKey, ...]:
    if relation.source.kind not in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        return ()
    row_source = _row_source_for_relation(relation, row_sources=row_sources)
    relation_field_ids = {field.field_id for field in relation.fields}
    keys = [
        RelationEntityKey(
            entity_kind=key.entity_kind,
            key_id=key.id,
            components=tuple(
                RelationEntityKeyComponent(
                    component_id=component.id,
                    field_id=component.field_id,
                )
                for component in key.components
            ),
        )
        for key in row_source.candidate_keys
        if all(component.field_id in relation_field_ids for component in key.components)
    ]
    keys.extend(
        RelationEntityKey(
            entity_kind=reference.target_entity_kind,
            key_id=reference.target_key_id,
            components=tuple(
                RelationEntityKeyComponent(
                    component_id=component.target_component_id,
                    field_id=component.local_field_id,
                )
                for component in reference.components
            ),
        )
        for reference in row_source.entity_references
        if all(
            component.local_field_id in relation_field_ids
            for component in reference.components
        )
    )
    return tuple(dict.fromkeys(keys))


def _relation_source_population_proof(
    relation: Relation,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    endpoint_arg_scope_refs: dict[str, frozenset[str]],
) -> ProofLineage:
    if catalog is None or relation.source.kind not in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        return ProofLineage()
    try:
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
    except KeyError:
        return ProofLineage()
    value_refs: set[str] = set()
    proof_refs = _source_mechanic_proof_refs(relation)
    if row_source.read_id:
        value_refs.add(read_evidence_ref(row_source.read_id))
    else:
        value_refs.add(row_source_evidence_ref(row_source.id))
    proof_refs.update(endpoint_arg_scope_refs.get(relation.id, frozenset()))
    row_tests = frozenset(
        claim.test_ref
        for claim in relation.source.population_coverage_claims
        if claim.role is PopulationCoverageRole.ROW_POPULATION
    )
    condition_tests = frozenset(
        claim.test_ref
        for claim in relation.source.population_coverage_claims
        if claim.role is PopulationCoverageRole.OPERATION_CONDITION
    )
    return ProofLineage(
        value_refs=frozenset({*value_refs, *proof_refs}),
        population_coverage=PopulationCoverage(
            row_tests=row_tests,
            condition_tests=condition_tests,
        ),
    )


def _binding_proof(
    relation: Relation,
    field_id: str,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
) -> ProofLineage:
    refs = {field_id}
    if catalog is None or relation.source.kind not in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        return ProofLineage.value(frozenset(refs))
    try:
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        row_source_field = row_source.field(field_id)
    except KeyError:
        return ProofLineage.value(frozenset(refs))
    refs.add(
        read_field_evidence_ref(
            read_id=row_source.read_id, field_id=row_source_field.id
        )
        if row_source.read_id
        else row_source_field_evidence_ref(
            row_source_id=row_source.id,
            field_id=row_source_field.id,
        )
    )
    refs.update(row_source_field.fact_refs)
    return ProofLineage.value(frozenset(refs))
