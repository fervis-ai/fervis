"""Answer-plan checks for fact-plan verification."""

from __future__ import annotations

from ._shared import (
    AnswerPlan,
    AuthorizedExecutionSources,
    CatalogSelectionResult,
    FactValue,
    QuestionContract,
    RequestedFact,
    RelationCatalog,
    RelationRows,
    RowSourceCatalog,
    VerificationError,
    build_row_source_catalog,
    verify_operation,
    verify_value_contract,
)
from .contract_types import RelationContract
from .contracts import _relation_contracts
from .execution_proof import ExecutionProofContext
from .operations import (
    _verify_answer_uses_evidence_relation,
    _verify_compute_scalar_availability,
    _verify_coverage_operation_relation_contracts,
    _verify_operation_field_references,
    _verify_operation_references,
)
from .question_contract import _verify_question_contract
from .render import _render_output_fact_refs, _verify_render_references
from .sources import (
    _allowed_read_ids,
    _verify_api_relation_catalog_refs,
    _verify_relations,
    _verify_required_source_params,
    _verify_sources,
    _verify_value_use_targets,
)
from fervis.lookup.plan_execution.compiled_execution import (
    compile_fact_execution,
)


def _verify_answer_plan(
    answer: AnswerPlan,
    *,
    question_contract: QuestionContract,
    catalog: RelationCatalog | None,
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[object, ...],
    memory_relations: tuple[RelationRows, ...],
    catalog_selection: CatalogSelectionResult | None,
    authorized_sources: AuthorizedExecutionSources | None,
) -> None:
    _verify_question_contract(question_contract)
    if not answer.operations:
        raise VerificationError("answer plan requires at least one operation")
    if answer.render_spec is None:
        raise VerificationError("answer plan requires render spec")
    executable_values = available_values
    _verify_plan_authored_values(answer)
    verify_value_contract(
        values=answer.values,
        value_uses=answer.value_uses,
        available_values=executable_values,
    )
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
    if catalog is not None:
        _verify_api_relation_catalog_refs(
            answer.relations,
            catalog,
            value_uses=answer.value_uses,
            values=(*answer.values, *executable_values),
            row_sources=row_sources,
            available_value_uses=available_value_uses,
        )
    for operation in answer.operations:
        verify_operation(operation)
    _verify_operation_references(answer)
    _verify_value_use_targets(
        answer,
        catalog=catalog,
        row_sources=row_sources,
        available_values=executable_values,
        available_value_uses=available_value_uses,
    )
    if catalog is not None:
        _verify_required_source_params(
            answer,
            row_sources=row_sources,
            available_values=executable_values,
            available_value_uses=available_value_uses,
        )
    _verify_compute_scalar_availability(answer)
    _verify_answer_uses_evidence_relation(answer)
    compiled = compile_fact_execution(
        answer=answer,
        catalog=catalog,
        row_sources=row_sources,
        available_values=executable_values,
        available_value_uses=available_value_uses,
    )
    proof_context = ExecutionProofContext.from_compiled_execution(
        compiled,
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
    _verify_render_references(answer, relation_contracts=relation_contracts)
    _verify_fact_fulfillment(
        answer,
        question_contract=question_contract,
        relation_contracts=relation_contracts,
        available_values=executable_values,
        catalog_selection=catalog_selection,
    )


def _verify_plan_authored_values(answer: AnswerPlan) -> None:
    if answer.values:
        raise VerificationError("fact plan values are not model-authored")


def _verify_fact_fulfillment(
    answer: AnswerPlan,
    *,
    question_contract: QuestionContract,
    relation_contracts: dict[str, RelationContract],
    available_values: tuple[FactValue, ...],
    catalog_selection: CatalogSelectionResult | None,
) -> None:
    requested = {fact.id: fact for fact in question_contract.requested_facts}
    requested_outputs = {
        fact.id: {output.id for output in fact.answer_outputs}
        for fact in requested.values()
    }
    fulfilled_outputs: set[tuple[str, str]] = set()
    fulfillments: set[tuple[str, str, str]] = set()
    render_output_ids = {
        render_output.id
        for render_output in (
            *answer.render_spec.relation_outputs,
            *answer.render_spec.scalar_outputs,
        )
    }
    render_output_fact_refs = _render_output_fact_refs(
        answer,
        relation_contracts=relation_contracts,
        available_values=available_values,
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
            item.render_output_id,
        )
        if fulfillment_key in fulfillments:
            raise VerificationError("duplicate fulfillment for answer output")
        if item.render_output_id not in render_output_ids:
            raise VerificationError("fulfillment render output is not rendered")
        if not render_output_fact_refs.get(item.render_output_id):
            raise VerificationError("fulfillment render output requires evidence proof")
        _verify_fulfillment_input_refs(
            fact,
            proof_refs=render_output_fact_refs[item.render_output_id],
        )
        fulfillments.add(fulfillment_key)
        fulfilled_outputs.add((fact.id, item.answer_output_id))
    missing = {
        (fact.id, output.id)
        for fact in requested.values()
        for output in fact.answer_outputs
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
            raise VerificationError("fulfillment render output missing input proof")
