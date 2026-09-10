"""Answer-program structural and execution checks."""

from __future__ import annotations

from typing import Protocol

from ._shared import (
    AnswerProgram,
    AuthorizedExecutionSources,
    CatalogSelectionResult,
    RelationCatalog,
    RelationRows,
    VerificationError,
)
from .relation_program import PreparedRelationProgram, prepare_relation_program, verify_prepared_relation_program
from .contract_types import RelationContract
from .execution_proof import ExecutionProofSource
from .operations import (
    _verify_answer_uses_evidence_input,
)
from .result_projection import (
    _result_output_fact_refs,
    _result_output_semantic_guarantees,
    _verify_result_output_targets,
    _verify_result_references,
)
from fervis.lookup.question_contract import analyze_requested_fact, QueryRequestedFact
from fervis.lookup.qualification import qualification_entails
from fervis.lookup.answer_program.inputs import CompiledProgramInputs
from fervis.lookup.answer_program.expression_instantiation import (
    InstantiatedProgramInputs,
)
from fervis.lookup.answer_program.contracts import AnswerProgramContractError
from fervis.lookup.answer_program.revisions import verify_capability_declarations
from fervis.lookup.plan_execution.operation_runtime import ResolvedOperationInput


class MaterializedAnswerProgram(ExecutionProofSource, Protocol):
    @property
    def instantiated_inputs(self) -> InstantiatedProgramInputs: ...

    @property
    def operation_inputs(self) -> tuple[ResolvedOperationInput, ...]: ...



def prepare_answer_program(
    answer: AnswerProgram,
    *,
    compiled_inputs: CompiledProgramInputs,
    catalog: RelationCatalog | None,
    memory_relations: tuple[RelationRows, ...],
    catalog_selection: CatalogSelectionResult | None,
    authorized_sources: AuthorizedExecutionSources | None,
) -> PreparedRelationProgram[AnswerProgram]:
    _verify_semantic_templates(answer)
    if not answer.operations and not (
        answer.result_projection.relation_outputs
        or answer.result_projection.scalar_outputs
    ):
        raise VerificationError("answer plan requires at least one operation")
    try:
        verify_capability_declarations(answer)
    except AnswerProgramContractError as exc:
        raise VerificationError(f"{exc.code}: {exc}") from exc
    prepared = prepare_relation_program(answer,compiled_inputs=compiled_inputs,catalog=catalog,
        memory_relations=memory_relations,catalog_selection=catalog_selection,authorized_sources=authorized_sources)
    _verify_answer_uses_evidence_input(answer)
    _verify_result_output_targets(answer, require_output=False)
    if prepared.structural_contracts:
        _verify_result_references(answer, relation_contracts=prepared.structural_contracts)
    return prepared


def verify_prepared_answer_program(
    structured: PreparedRelationProgram[AnswerProgram],
    *,
    materialized: MaterializedAnswerProgram,
    catalog: RelationCatalog | None,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    answer = structured.program
    relation_contracts = verify_prepared_relation_program(structured,materialized=materialized,catalog=catalog,
        guarantee_declarations=answer.relation_guarantees)
    _verify_result_references(answer, relation_contracts=relation_contracts)
    _verify_fact_fulfillment(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=materialized.operation_inputs,
        catalog_selection=catalog_selection,
    )


def _verify_fact_fulfillment(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
    operation_inputs,
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    requested = {fact.id: fact for fact in answer.fact_template}
    requested_outputs = {
        fact.id: {output.id for output in fact.outputs}
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
    result_output_guarantees = _result_output_semantic_guarantees(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    semantic_inputs = {item.id: item for item in answer.inputs}
    semantic_denotations = {
        item.input_ref: item for item in answer.input_denotations
    }
    semantic_indexes = {
        fact.id: analyze_requested_fact(
            fact,
            inputs=semantic_inputs,
            input_denotations=semantic_denotations,
        )
        for fact in answer.fact_template
        if not isinstance(fact, QueryRequestedFact)
    }
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
        if isinstance(fact, QueryRequestedFact):
            from fervis.lookup.relational_sql.request_contract import verify_query_output
            verify_query_output(fact, item, answer, relation_contracts)
            fulfillments.add(fulfillment_key)
            fulfilled_outputs.add((fact.id, item.answer_output_id))
            continue
        semantic_guarantee = result_output_guarantees.get(
            item.result_output_id, {}
        ).get(fact.id)
        if semantic_guarantee is None:
            raise VerificationError("fulfillment result lacks semantic guarantee")
        semantic_index = semantic_indexes[fact.id]
        if not qualification_entails(
            semantic_guarantee.qualification.formula,
            semantic_index.output_qualification(item.answer_output_id),
        ):
            raise VerificationError("fulfillment does not enforce qualification")
        subject = semantic_guarantee.subject
        obligation = semantic_index.subject_obligation
        if (
            subject.subject_set_ref != obligation.subject_set_ref.token
            or subject.interpretation != type(obligation).__name__
        ):
            raise VerificationError("fulfillment does not enforce subject interpretation")
        fulfillments.add(fulfillment_key)
        fulfilled_outputs.add((fact.id, item.answer_output_id))
    missing = {
        (fact.id, output.id)
        for fact in requested.values()
        for output in fact.outputs
    } - fulfilled_outputs
    if missing:
        raise VerificationError("requested fact answer output is not fulfilled")


def _verify_semantic_templates(answer: AnswerProgram) -> None:
    if not answer.fact_template:
        raise VerificationError("answer program requires a semantic fact template")
    from fervis.lookup.question_contract.model import validate_input_denotations
    try:
        validate_input_denotations(answer.inputs, answer.input_denotations)
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    inputs = {item.id: item for item in answer.inputs}
    denotations = {item.input_ref: item for item in answer.input_denotations}
    if len(denotations) != len(answer.input_denotations):
        raise VerificationError("answer program repeats an input denotation")
    if set(inputs) != set(denotations):
        raise VerificationError("answer program input denotations are incomplete")
    if len(inputs) != len(answer.inputs):
        raise VerificationError("answer program repeats a semantic input")
    declared_signatures = {
        parameter.input_ref: frozenset(parameter.input_use_refs)
        for parameter in answer.parameters
        if parameter.input_ref
    }
    if len(declared_signatures) != sum(
        bool(parameter.input_ref) for parameter in answer.parameters
    ):
        raise VerificationError("answer program repeats an input parameter signature")
    required_signatures: dict[str, set[str]] = {}
    for fact in answer.fact_template:
        if isinstance(fact, QueryRequestedFact):
            from fervis.lookup.relational_sql.request_contract import verify_query_request
            verify_query_request(fact, answer)
            for ref in fact.input_refs:
                if ref not in inputs:
                    raise VerificationError('SQL request references an undeclared input')
                required_signatures.setdefault(ref, set()).add(fact.input_use_ref(ref))
            continue
        try:
            index = analyze_requested_fact(
                fact,
                inputs=inputs,
                input_denotations=denotations,
            )
        except (TypeError, ValueError) as exc:
            raise VerificationError("answer program contains an invalid semantic fact") from exc
        for use in index.input_use_sites:
            required_signatures.setdefault(use.input_ref, set()).add(use.use_ref)
    if declared_signatures != {
        ref: frozenset(use_refs) for ref, use_refs in required_signatures.items()
    }:
        raise VerificationError("answer program input signatures do not match semantic uses")
