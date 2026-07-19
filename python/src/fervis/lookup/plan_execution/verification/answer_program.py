"""Answer-program checks for fact-plan verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._shared import (
    AnswerProgram,
    AuthorizedExecutionSources,
    CatalogSelectionResult,
    QuestionContract,
    RequestedFact,
    RelationCatalog,
    RelationRows,
    RowSourceCatalog,
    VerificationError,
    build_row_source_catalog,
    verify_operation,
)
from .contract_types import RelationContract
from .contracts import _relation_contracts
from .execution_proof import ExecutionProofContext
from .operations import (
    _verify_answer_uses_evidence_input,
    _verify_compute_scalar_availability,
    _verify_coverage_operation_relation_contracts,
    _verify_operation_field_references,
    _verify_operation_references,
)
from .question_contract import _verify_question_contract
from .result_projection import (
    _result_output_fact_refs,
    _result_output_proofs,
    _verify_result_output_targets,
    _verify_result_references,
)
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    MembershipTestRef,
)
from .sources import (
    _allowed_read_ids,
    _verify_api_relation_catalog_refs,
    _verify_compute_input_population_coverage_claims,
    _verify_relations,
    _verify_required_source_params,
    _verify_source_population_coverage_claims,
    _verify_sources,
    _verify_program_expression_targets,
)
from fervis.lookup.answer_program.inputs import CompiledProgramInputs
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.answer_program.contracts import AnswerProgramContractError
from fervis.lookup.answer_program.revisions import verify_capability_declarations

if TYPE_CHECKING:
    from fervis.lookup.answer_program.instantiation import _MaterializedExecution


@dataclass(frozen=True)
class _StructuredAnswerProgram:
    program: AnswerProgram
    bindings: BindingSet
    row_sources: RowSourceCatalog


def _verify_answer_program_structure(
    answer: AnswerProgram,
    *,
    compiled_inputs: CompiledProgramInputs,
    question_contract: QuestionContract,
    catalog: RelationCatalog | None,
    memory_relations: tuple[RelationRows, ...],
    catalog_selection: CatalogSelectionResult | None,
    authorized_sources: AuthorizedExecutionSources | None,
) -> _StructuredAnswerProgram:
    bindings = compiled_inputs.bindings
    _verify_question_contract(question_contract)
    if not answer.operations:
        raise VerificationError("answer plan requires at least one operation")
    try:
        verify_capability_declarations(answer)
    except AnswerProgramContractError as exc:
        raise VerificationError(f"{exc.code}: {exc}") from exc
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
    _verify_source_population_coverage_claims(
        answer,
        question_contract=question_contract,
        row_sources=row_sources,
        bindings=bindings,
    )
    _verify_compute_input_population_coverage_claims(
        answer,
        question_contract=question_contract,
        bindings=bindings,
    )
    _verify_relations(answer.relations)
    for operation in answer.operations:
        verify_operation(operation)
    _verify_operation_references(answer)
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
    _verify_answer_uses_evidence_input(answer)
    _verify_result_output_targets(answer, require_output=False)
    relation_contracts = _relation_contracts(
        answer,
        catalog=catalog,
        row_sources=row_sources,
        proof_context=ExecutionProofContext.empty(),
    )
    _verify_population_fulfillment(
        answer,
        question_contract=question_contract,
        relation_contracts=relation_contracts,
        operation_inputs=(),
    )
    return _StructuredAnswerProgram(
        program=answer,
        bindings=bindings,
        row_sources=row_sources,
    )


def _verify_answer_program_execution(
    structured: _StructuredAnswerProgram,
    *,
    materialized: _MaterializedExecution,
    question_contract: QuestionContract,
    catalog: RelationCatalog | None,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
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
    )
    _verify_operation_field_references(answer, relation_contracts=relation_contracts)
    _verify_coverage_operation_relation_contracts(
        answer,
        relation_contracts=relation_contracts,
    )
    _verify_result_references(answer, relation_contracts=relation_contracts)
    _verify_fact_fulfillment(
        answer,
        question_contract=question_contract,
        relation_contracts=relation_contracts,
        operation_inputs=materialized.operation_inputs,
        catalog_selection=catalog_selection,
    )


def _verify_fact_fulfillment(
    answer: AnswerProgram,
    *,
    question_contract: QuestionContract,
    relation_contracts: dict[str, RelationContract],
    operation_inputs,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    requested = {fact.id: fact for fact in question_contract.requested_facts}
    requested_outputs = {
        fact.id: {output.id for output in fact.support_answer_outputs}
        for fact in requested.values()
    }
    fulfilled_outputs: set[tuple[str, str]] = set()
    fulfillments: set[tuple[str, str, str]] = set()
    result_output_ids = {
        output.id for output in answer.result_projection.relation_outputs
    } | {output.id for output in answer.result_projection.scalar_outputs}
    result_output_fact_refs = _result_output_fact_refs(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    _verify_population_fulfillment(
        answer,
        question_contract=question_contract,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    for item in answer.fulfillment:
        fact = requested.get(item.requested_fact_id)
        if fact is None:
            raise VerificationError("fulfillment references unknown requested fact")
        if item.answer_output_id not in requested_outputs[fact.id]:
            raise VerificationError("fulfillment references unknown answer output")
        fulfillment_key = (
            fact.id,
            item.answer_output_id,
            item.result_output_id,
        )
        if fulfillment_key in fulfillments:
            raise VerificationError("duplicate fulfillment for answer output")
        if item.result_output_id not in result_output_ids:
            raise VerificationError("fulfillment result output is not projected")
        if not result_output_fact_refs.get(item.result_output_id):
            raise VerificationError("fulfillment result output requires evidence proof")
        _verify_fulfillment_input_refs(
            fact,
            proof_refs=result_output_fact_refs[item.result_output_id],
        )
        fulfillments.add(fulfillment_key)
        fulfilled_outputs.add((fact.id, item.answer_output_id))
    missing = {
        (fact.id, output.id)
        for fact in requested.values()
        for output in fact.support_answer_outputs
    } - fulfilled_outputs
    if missing:
        raise VerificationError("requested fact answer output is not fulfilled")


def _verify_fulfillment_input_refs(
    fact: RequestedFact,
    *,
    proof_refs: frozenset[str],
) -> None:
    for input_ref in fact.input_refs:
        if f"known_input:{input_ref}" not in proof_refs:
            raise VerificationError(
                f"fulfillment result output missing input proof: {input_ref}"
            )


def _verify_population_fulfillment(
    answer: AnswerProgram,
    *,
    question_contract: QuestionContract,
    relation_contracts: dict[str, RelationContract],
    operation_inputs,
) -> None:
    facts = {fact.id: fact for fact in question_contract.requested_facts}
    result_proofs = _result_output_proofs(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    for fulfillment in answer.fulfillment:
        fact = facts.get(fulfillment.requested_fact_id)
        proof = result_proofs.get(fulfillment.result_output_id)
        if fact is None or proof is None:
            continue
        population = fact.answer_population
        if population is None:
            continue
        required = frozenset(
            MembershipTestRef(
                requested_fact_id=fact.id,
                membership_test_id=test.id,
            )
            for test in population.membership_tests
            if test.kind is not AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY
        )
        missing = required - proof.population_coverage.row_tests
        if missing:
            raise VerificationError(
                "fulfillment result does not enforce answer population tests: "
                + ", ".join(sorted(test.membership_test_id for test in missing))
            )
