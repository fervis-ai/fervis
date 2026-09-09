"""Physical graph checks shared by identity discovery and factual execution."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from fervis.lookup.answer_program.model import (
    RelationProgram,
    RelationGuaranteeDeclaration,
)
from fervis.lookup.answer_program.inputs import CompiledProgramInputs
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceCatalog,
    build_row_source_catalog,
)
from fervis.lookup.plan_execution.relations import RelationRows
from ._shared import verify_operation
from .contract_types import RelationContract
from .sources import (
    _allowed_read_ids,
    _verify_sources,
    _verify_relations,
    _verify_program_expression_targets,
    _verify_required_source_params,
    _verify_api_relation_catalog_refs,
    verify_dependent_argument_contracts,
)
from .operations import (
    _verify_operation_references,
    _verify_compute_scalar_availability,
    _verify_operation_field_references,
    _verify_coverage_operation_relation_contracts,
)
from .contracts import _relation_contracts
from .execution_proof import ExecutionProofContext

_Program = TypeVar("_Program", bound=RelationProgram)


@dataclass(frozen=True)
class PreparedRelationProgram(Generic[_Program]):
    program: _Program
    bindings: BindingSet
    row_sources: RowSourceCatalog
    structural_contracts: dict[str, RelationContract] = field(default_factory=dict)


def prepare_relation_program(
    answer: _Program,
    *,
    compiled_inputs: CompiledProgramInputs,
    catalog: RelationCatalog | None,
    memory_relations: tuple[RelationRows, ...] = (),
    catalog_selection=None,
    authorized_sources=None,
) -> PreparedRelationProgram[_Program]:
    bindings = compiled_inputs.bindings
    row_sources = (
        build_row_source_catalog(catalog, memory_relations=memory_relations)
        if catalog is not None
        else RowSourceCatalog()
    )
    _verify_sources(
        answer,
        row_sources=row_sources,
        allowed_read_ids=_allowed_read_ids(
            catalog_selection=catalog_selection,
            authorized_sources=authorized_sources,
        ),
    )
    _verify_relations(answer.relations)
    for operation in answer.operations:
        verify_operation(operation)
    _verify_operation_references(answer)
    from fervis.lookup.relational_sql.reference_matching import verify_reference_candidate_completeness
    verify_reference_candidate_completeness(answer)
    _verify_program_expression_targets(
        answer,
        bindings=bindings,
        catalog=catalog,
        row_sources=row_sources,
    )
    if catalog is not None:
        _verify_required_source_params(
            answer,
            row_sources=row_sources,
        )
    _verify_compute_scalar_availability(answer)
    contracts = {}
    if catalog is not None:
        # Structural field/key contracts are already known. Validate them before
        # any read; current-run evidence is checked again after materialization.
        contracts=_relation_contracts(answer,catalog=catalog,row_sources=row_sources,
            proof_context=ExecutionProofContext.empty())
        _verify_operation_field_references(answer,relation_contracts=contracts)
        verify_dependent_argument_contracts(answer,relation_contracts=contracts,row_sources=row_sources)
        from fervis.lookup.relational_sql.parameter_usage import validate_bound_program_parameters
        validate_bound_program_parameters(answer,bindings,row_sources,contracts)
    return PreparedRelationProgram(answer, bindings, row_sources, contracts)


def verify_prepared_relation_program(
    structured: PreparedRelationProgram,
    *,
    materialized,
    catalog,
    guarantee_declarations: tuple[RelationGuaranteeDeclaration, ...] = (),
):
    answer = structured.program
    row_sources = structured.row_sources
    if catalog is not None:
        _verify_api_relation_catalog_refs(
            answer.relations,
            catalog,
            row_sources=row_sources,
            instantiated_inputs=materialized.instantiated_inputs,
        )
    proof_context = ExecutionProofContext.from_materialized_execution(
        materialized,
    )
    relation_contracts = _relation_contracts(
        answer,
        catalog=catalog,
        row_sources=row_sources,
        proof_context=proof_context,
        guarantee_declarations=guarantee_declarations,
    )
    verify_dependent_argument_contracts(
        answer, relation_contracts=relation_contracts, row_sources=row_sources
    )
    _verify_operation_field_references(answer, relation_contracts=relation_contracts)
    _verify_coverage_operation_relation_contracts(
        answer,
        relation_contracts=relation_contracts,
    )
    return relation_contracts
