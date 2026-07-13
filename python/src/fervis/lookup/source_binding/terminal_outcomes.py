"""Backend-owned terminal outcomes for source binding."""

from __future__ import annotations

from fervis.lookup.relation_catalog.selection import (
    catalog_selection_evidence_ref,
)
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    PlanImpossible,
)
from fervis.lookup.answer_program.values import known_input_id_for_value
from fervis.lookup.fact_planning.fact_requirements import (
    fact_endpoint_requirements,
)
from fervis.lookup.fact_plan.row_sources import read_evidence_ref
from fervis.lookup.source_binding.candidates import (
    source_candidate_registry,
    source_binding_prompt_candidate_fulfillment_answer_output_ids,
    source_binding_prompt_candidate_requested_fact_ids,
)
from fervis.lookup.source_binding.candidates.model import SourceCandidate
from fervis.lookup.source_binding.candidates.contracts import ValueEvidence
from fervis.lookup.source_binding.model import SourceBindingRequest


def source_binding_clarification_input_ids(
    request: SourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    requirements = fact_endpoint_requirements(
        catalog=request.relation_catalog,
        catalog_selection=request.catalog_selection,
        available_values=request.available_values,
        available_value_uses=request.available_value_uses,
    )
    required_input_ids: list[str] = []
    choice_input_ids: list[str] = []
    for item in requirements.clarifiable_missing_inputs:
        if item.choices:
            choice_input_ids.append(item.id)
            continue
        required_input_ids.append(item.id)
    return {
        "required_catalog_input_ids": tuple(required_input_ids),
        "required_catalog_choice_input_ids": tuple(choice_input_ids),
    }


def backend_impossible_without_answer_candidates(
    request: SourceBindingRequest,
) -> PlanImpossible | None:
    clarification_ids = source_binding_clarification_input_ids(request)
    if any(clarification_ids.values()):
        return None
    if _has_value_source_candidates(request):
        return None
    blocked_facts = _blocked_facts_without_answer_output_candidates(request)
    if not blocked_facts:
        return None
    return PlanImpossible(blocked_facts=blocked_facts)


def _has_value_source_candidates(request: SourceBindingRequest) -> bool:
    candidates = source_candidate_registry(request).candidates_by_id.values()
    known_input_value_ids = {
        value.id
        for value in request.available_values
        if known_input_id_for_value(value)
    }
    return any(
        _is_answer_capable_value_candidate(
            candidate,
            known_input_value_ids=known_input_value_ids,
        )
        for candidate in candidates
        if candidate.kind == "value"
    )


def _is_answer_capable_value_candidate(
    candidate: SourceCandidate,
    *,
    known_input_value_ids: set[str],
) -> bool:
    value_id = candidate.value_id
    if value_id not in known_input_value_ids:
        return True
    return _candidate_has_number_value_evidence(candidate)


def _candidate_has_number_value_evidence(candidate: SourceCandidate) -> bool:
    return any(
        isinstance(item, ValueEvidence) and item.type == "number"
        for item in candidate.evidence_items
    )


def _blocked_facts_without_answer_output_candidates(
    request: SourceBindingRequest,
) -> tuple[BlockedFact, ...]:
    answer_output_ids_by_candidate = (
        source_binding_prompt_candidate_fulfillment_answer_output_ids(request)
    )
    requested_fact_ids_by_candidate = (
        source_binding_prompt_candidate_requested_fact_ids(request)
    )
    supported_output_ids_by_fact: dict[str, set[str]] = {}
    for candidate_id, output_ids in answer_output_ids_by_candidate.items():
        for requested_fact_id in requested_fact_ids_by_candidate.get(
            candidate_id,
            (),
        ):
            supported_output_ids_by_fact.setdefault(requested_fact_id, set()).update(
                output_ids
            )
    return tuple(
        _blocked_fact_for_requested_fact(fact.id, request=request)
        for fact in request.requested_facts
        if any(
            output.id not in supported_output_ids_by_fact.get(fact.id, set())
            for output in fact.support_answer_outputs
        )
    )


def _blocked_fact_for_requested_fact(
    requested_fact_id: str,
    *,
    request: SourceBindingRequest,
) -> BlockedFact:
    selected_read_ids = _selected_read_ids_for_requested_fact(
        requested_fact_id,
        request=request,
    )
    if selected_read_ids:
        return BlockedFact(
            requested_fact_id=requested_fact_id,
            basis=BlockedFactBasis.CATALOG_ACCESS,
            evidence_refs=tuple(
                read_evidence_ref(read_id) for read_id in selected_read_ids
            ),
            reviewed_read_ids=selected_read_ids,
            explanation="No selected API read can provide an answer-capable source binding.",
        )
    return BlockedFact(
        requested_fact_id=requested_fact_id,
        basis=BlockedFactBasis.CATALOG_ACCESS,
        evidence_refs=(
            catalog_selection_evidence_ref(requested_fact_id=requested_fact_id),
        ),
        explanation="No API read candidates are available for the requested fact.",
    )


def _selected_read_ids_for_requested_fact(
    requested_fact_id: str,
    *,
    request: SourceBindingRequest,
) -> tuple[str, ...]:
    for selection in request.catalog_selection.requested_fact_selections:
        if selection.requested_fact_id == requested_fact_id:
            return tuple(selection.selected_read_ids)
    return ()
