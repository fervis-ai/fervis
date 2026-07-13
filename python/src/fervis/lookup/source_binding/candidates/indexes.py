"""Indexes derived from typed source-candidate registries."""

from .model import SourceCandidate


def _candidate_requested_fact_ids(
    candidates: dict[str, SourceCandidate],
    *,
    candidate_ids: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    return {
        candidate.id: candidate.applies_to_requested_fact_ids
        for candidate in _indexed_candidates(candidates, candidate_ids)
    }


def _candidate_required_param_decision_ids(
    candidate: SourceCandidate,
) -> tuple[str, ...]:
    return tuple(
        param_id
        for param in candidate.params
        if param.decision_options
        and not param.finite_choice_review
        and (param.required or (bool(param.choices) and not param.has_default))
        for param_id in (param.id,)
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
    for item in candidate.fulfillment_support_sets:
        answer_output_id = item.answer_output_id
        choice_id = item.fulfillment_choice_id
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
                for item in candidate.fulfillment_support_sets
                for answer_output_id in (item.answer_output_id,)
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
            item.id for item in candidate.population_bindings if item.id
        )
        for candidate in _indexed_candidates(candidates, candidate_ids)
    }


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
