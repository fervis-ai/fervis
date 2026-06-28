"""Indexes derived from typed source-candidate registries."""

from ._shared import Any
from .model import SourceCandidate
from ..param_surface import param_has_default_value, param_requires_finite_choice_review


def _candidate_requested_fact_ids(
    candidates: dict[str, SourceCandidate],
    *,
    candidate_ids: tuple[str, ...] = (),
) -> dict[str, str]:
    return {
        candidate.id: candidate.requested_fact_id
        for candidate in _indexed_candidates(candidates, candidate_ids)
    }


def _candidate_required_param_decision_ids(
    candidate: SourceCandidate,
) -> tuple[str, ...]:
    return tuple(
        param_id
        for param in candidate.params
        if isinstance(param, dict)
        and param.get("decision_options")
        and not param_requires_finite_choice_review(param)
        and (
            param.get("required") is True
            or (bool(param.get("choices")) and not param_has_default_value(param))
        )
        for param_id in (str(param.get("param_id") or ""),)
        if param_id
    )


def _candidate_fulfillment_support_set_ids_by_answer_output(
    candidates: dict[str, SourceCandidate],
    *,
    candidate_ids: tuple[str, ...] = (),
) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        candidate.id: _fulfillment_support_set_ids_by_answer_output(candidate)
        for candidate in _indexed_candidates(candidates, candidate_ids)
    }


def _fulfillment_support_set_ids_by_answer_output(
    candidate: SourceCandidate,
) -> dict[str, tuple[str, ...]]:
    output: dict[str, list[str]] = {}
    for item in _candidate_fulfillment_support_sets(candidate):
        answer_output_id = str(item.get("answer_output_id") or "")
        choice_id = str(item.get("fulfillment_choice_id") or "")
        if not answer_output_id or not choice_id:
            continue
        output.setdefault(answer_output_id, []).append(choice_id)
    return {
        answer_output_id: tuple(dict.fromkeys(choice_ids))
        for answer_output_id, choice_ids in output.items()
    }


def _candidate_fulfillment_answer_output_ids(
    candidates: dict[str, SourceCandidate],
    *,
    candidate_ids: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    return {
        candidate.id: tuple(
            dict.fromkeys(
                answer_output_id
                for item in _candidate_fulfillment_support_sets(candidate)
                for answer_output_id in (str(item.get("answer_output_id") or ""),)
                if answer_output_id
            )
        )
        for candidate in _indexed_candidates(candidates, candidate_ids)
    }


def _candidate_population_binding_ids(
    candidates: dict[str, SourceCandidate],
    *,
    candidate_ids: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    return {
        candidate.id: tuple(
            binding_id
            for item in candidate.population_bindings
            if isinstance(item, dict)
            for binding_id in (str(item.get("population_binding_id") or ""),)
            if binding_id
        )
        for candidate in _indexed_candidates(candidates, candidate_ids)
    }


def _candidate_fulfillment_support_sets(
    candidate: SourceCandidate,
) -> tuple[dict[str, Any], ...]:
    payload = candidate.payload or {}
    return tuple(
        item
        for item in payload.get("fulfillment_support_sets") or ()
        if isinstance(item, dict)
    )


def _indexed_candidates(
    candidates: dict[str, SourceCandidate],
    candidate_ids: tuple[str, ...],
) -> tuple[SourceCandidate, ...]:
    if not candidate_ids:
        return tuple(candidates.values())
    return tuple(
        candidates[candidate_id]
        for candidate_id in candidate_ids
        if candidate_id in candidates
    )
