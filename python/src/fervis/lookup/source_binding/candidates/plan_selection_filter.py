"""Plan selection filtering for source-binding candidate payloads."""

from __future__ import annotations

from .candidate_tree import CandidateTreeContext, map_source_candidate_tree
from ._shared import SourceBindingRequest


def filter_prompt_payload_by_plan_selection(
    payload: dict[str, object],
    request: SourceBindingRequest,
) -> dict[str, object]:
    selected_by_fact = _plan_selection_source_candidate_ids_by_fact(request)
    selected_support_sets_by_fact_candidate = (
        _plan_selection_fulfillment_support_set_ids_by_fact_candidate(request)
    )
    selected_support_sets_by_candidate = (
        _plan_selection_fulfillment_support_set_ids_by_candidate(request)
    )
    selected_all = {
        candidate_id
        for candidate_ids in selected_by_fact.values()
        for candidate_id in candidate_ids
    }
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _filter_candidate_for_plan_selection(
            candidate,
            context=context,
            selected_by_fact=selected_by_fact,
            selected_all=selected_all,
            selected_support_sets_by_fact_candidate=(
                selected_support_sets_by_fact_candidate
            ),
            selected_support_sets_by_candidate=selected_support_sets_by_candidate,
        ),
        top_level_keys=("utility_source_candidates", "value_source_candidates"),
    )


def _filter_candidate_for_plan_selection(
    candidate: dict[str, object],
    *,
    context: CandidateTreeContext,
    selected_by_fact: dict[str, set[str]],
    selected_all: set[str],
    selected_support_sets_by_fact_candidate: dict[tuple[str, str], set[str]],
    selected_support_sets_by_candidate: dict[str, set[str]],
) -> dict[str, object] | None:
    candidate_id = str(candidate.get("source_candidate_id") or "")
    if context.requested_fact_id:
        selected = selected_by_fact.get(context.requested_fact_id)
        if selected is None:
            return candidate
        if candidate_id not in selected:
            return None
        selected_support_set_ids = selected_support_sets_by_fact_candidate.get(
            (context.requested_fact_id, candidate_id),
            set(),
        )
    else:
        if candidate_id not in selected_all:
            return None
        selected_support_set_ids = selected_support_sets_by_candidate.get(
            candidate_id,
            set(),
        )
    return _filter_candidate_fulfillment_support_sets(
        candidate,
        selected_support_set_ids=selected_support_set_ids,
    )


def _filter_candidate_fulfillment_support_sets(
    candidate: dict[str, object],
    *,
    selected_support_set_ids: set[str],
) -> dict[str, object]:
    if not selected_support_set_ids:
        return candidate
    output = dict(candidate)
    output["fulfillment_support_sets"] = [
        support_set
        for support_set in candidate.get("fulfillment_support_sets") or ()
        if isinstance(support_set, dict)
        and _support_set_binding_id(support_set) in selected_support_set_ids
    ]
    binding_surface = output.get("binding_surface")
    if isinstance(binding_surface, dict):
        output["binding_surface"] = {
            **binding_surface,
            "fulfillment_support_sets": [
                support_set
                for support_set in binding_surface.get("fulfillment_support_sets") or ()
                if isinstance(support_set, dict)
                and _support_set_binding_id(support_set) in selected_support_set_ids
            ],
        }
    return output


def _plan_selection_source_candidate_ids_by_fact(
    request: SourceBindingRequest,
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for plan in request.plan_selection.plan_selections:
        selected = output.setdefault(plan.requested_fact_id, set())
        selected.update(member.source_candidate_id for member in plan.source_members)
    return output


def _plan_selection_fulfillment_support_set_ids_by_fact_candidate(
    request: SourceBindingRequest,
) -> dict[tuple[str, str], set[str]]:
    output: dict[tuple[str, str], set[str]] = {}
    for plan in request.plan_selection.plan_selections:
        for member in plan.source_members:
            key = (plan.requested_fact_id, member.source_candidate_id)
            output.setdefault(key, set()).update(member.fulfillment_support_set_ids)
    return output


def _plan_selection_fulfillment_support_set_ids_by_candidate(
    request: SourceBindingRequest,
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for plan in request.plan_selection.plan_selections:
        for member in plan.source_members:
            output.setdefault(member.source_candidate_id, set()).update(
                member.fulfillment_support_set_ids
            )
    return output


def _support_set_binding_id(support_set: dict[str, object]) -> str:
    return str(
        support_set.get("fulfillment_support_set_id")
        or support_set.get("fulfillment_choice_id")
        or ""
    )
