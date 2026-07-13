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
from .contract_types import ProofLineage, RelationContract
from .execution_proof import ExecutionProofContext
from .operation_contracts import _operation_relation_contract
from .sources import _row_source_for_relation


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
        row_filter_scope_refs=proof_context.row_filter_scope_refs,
    )
    field_proofs = {
        field.field_id: _binding_proof(
            relation,
            field.field_id,
            catalog=catalog,
            row_sources=row_sources,
        ).with_population_scope(population_proof.population_scope_refs)
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
        population_proof=population_proof,
    )


def _relation_source_population_proof(
    relation: Relation,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    endpoint_arg_scope_refs: dict[str, frozenset[str]],
    row_filter_scope_refs: dict[str, frozenset[str]],
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
    scope_refs: set[str] = set()
    if row_source.read_id:
        value_refs.add(read_evidence_ref(row_source.read_id))
    else:
        value_refs.add(row_source_evidence_ref(row_source.id))
    for source_filter in relation.source.row_filters:
        scope_refs.update(str(ref) for ref in source_filter.proof_refs if str(ref))
    for binding in relation.source.param_bindings:
        scope_refs.update(str(ref) for ref in binding.proof_refs if str(ref))
    scope_refs.update(str(ref) for ref in relation.source.proof_refs if str(ref))
    scope_refs.update(row_filter_scope_refs.get(relation.id, frozenset()))
    scope_refs.update(endpoint_arg_scope_refs.get(relation.id, frozenset()))
    return ProofLineage(
        value_refs=frozenset(value_refs),
        population_scope_refs=frozenset(scope_refs),
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
