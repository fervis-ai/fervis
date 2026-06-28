"""Stable model-facing handles for source-binding candidates."""

from ._shared import Any
from .candidate_tree import CandidateTreeContext, map_source_candidate_tree
from .evidence import _candidate_with_evidence_items
from .fulfillment_slots import _candidate_with_fulfillment_slots
from .params import (
    _candidate_with_param_decision_options,
    _candidate_with_param_population_contracts,
)


def _with_stable_source_candidate_handles(
    payload: dict[str, Any],
    *,
    requested_facts: tuple[Any, ...] = (),
    project_fulfillment_slots: bool = True,
) -> dict[str, Any]:
    output = dict(payload)
    next_index = 1
    used_ids: set[str] = set()
    reserved_new_api_ids = _reserved_new_api_source_candidate_ids(payload)

    def assign(
        candidate: dict[str, Any],
        *,
        context: CandidateTreeContext,
    ) -> dict[str, Any]:
        nonlocal next_index
        assigned = dict(candidate)
        candidate_id = str(assigned.get("source_candidate_id") or "")
        should_preserve_id = (
            assigned.get("kind") == "new_api_read"
            and candidate_id.startswith("source_")
            and candidate_id not in used_ids
        )
        if not should_preserve_id:
            blocked_ids = used_ids | reserved_new_api_ids
            while f"source_{next_index}" in blocked_ids:
                next_index += 1
            candidate_id = f"source_{next_index}"
            next_index += 1
        assigned["source_candidate_id"] = candidate_id
        used_ids.add(candidate_id)
        assigned.pop("evidence_items", None)
        slot_requested_facts = _requested_facts_for_tree_context(
            candidate,
            context=context,
            requested_facts=requested_facts,
        )
        assigned = _candidate_with_param_decision_options(assigned)
        assigned = _candidate_with_evidence_items(assigned)
        assigned = _candidate_with_param_population_contracts(
            assigned,
            requested_facts=slot_requested_facts,
        )
        if not (
            project_fulfillment_slots
            and _should_project_fulfillment_slots_for_context(
                candidate,
                context=context,
            )
        ):
            return assigned
        return _candidate_with_fulfillment_slots(
            assigned,
            requested_facts=slot_requested_facts,
        )

    return map_source_candidate_tree(
        output,
        lambda candidate, context: assign(candidate, context=context),
        top_level_keys=("utility_source_candidates", "value_source_candidates"),
    )


def _reserved_new_api_source_candidate_ids(payload: dict[str, Any]) -> set[str]:
    return {
        candidate_id
        for fact_sources in payload.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict) and candidate.get("kind") == "new_api_read"
        for candidate_id in (str(candidate.get("source_candidate_id") or ""),)
        if candidate_id.startswith("source_")
    }


def _with_fulfillment_slots(
    payload: dict[str, Any],
    *,
    requested_facts: tuple[Any, ...] = (),
) -> dict[str, Any]:
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _candidate_with_fulfillment_slots_for_tree(
            candidate,
            context=context,
            requested_facts=requested_facts,
        ),
        top_level_keys=("utility_source_candidates", "value_source_candidates"),
    )


def _candidate_with_fulfillment_slots_for_tree(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
    requested_facts: tuple[Any, ...],
) -> dict[str, Any]:
    if context.top_level_key and not _should_project_fulfillment_slots(
        key=context.top_level_key,
        candidate=candidate,
    ):
        return candidate
    slot_requested_facts = (
        _requested_facts_for_fact_id(
            context.requested_fact_id,
            requested_facts=requested_facts,
        )
        if context.requested_fact_id
        else _requested_facts_for_candidate(
            candidate,
            requested_facts=requested_facts,
        )
    )
    return _candidate_with_fulfillment_slots(
        candidate,
        requested_facts=slot_requested_facts,
    )


def _requested_facts_for_tree_context(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
    requested_facts: tuple[Any, ...],
) -> tuple[Any, ...]:
    if context.requested_fact_id:
        return _requested_facts_for_fact_id(
            context.requested_fact_id,
            requested_facts=requested_facts,
        )
    return _requested_facts_for_candidate(candidate, requested_facts=requested_facts)


def _should_project_fulfillment_slots_for_context(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
) -> bool:
    if not context.top_level_key:
        return True
    return _should_project_fulfillment_slots(
        key=context.top_level_key,
        candidate=candidate,
    )


def _requested_facts_for_fact_id(
    requested_fact_id: str,
    *,
    requested_facts: tuple[Any, ...],
) -> tuple[Any, ...]:
    if not requested_fact_id:
        return requested_facts
    matched = tuple(
        fact
        for fact in requested_facts
        if str(getattr(fact, "id", "") or "") == requested_fact_id
    )
    return matched or requested_facts


def _should_project_fulfillment_slots(
    *,
    key: str,
    candidate: dict[str, Any],
) -> bool:
    if key == "utility_source_candidates":
        return False
    if key == "value_source_candidates":
        if str(candidate.get("type") or "") == "time_scope":
            return False
        return bool(
            candidate.get("answer_output_ids")
            or candidate.get("prior_answer_output_ids")
            or "value" in candidate
        )
    return True


def _requested_facts_for_candidate(
    candidate: dict[str, Any],
    *,
    requested_facts: tuple[Any, ...],
) -> tuple[Any, ...]:
    applies_to = {
        str(item)
        for item in candidate.get("applies_to_requested_facts") or ()
        if str(item)
    }
    if not applies_to:
        return requested_facts
    return tuple(
        fact
        for fact in requested_facts
        if str(getattr(fact, "id", "") or "") in applies_to
    )
