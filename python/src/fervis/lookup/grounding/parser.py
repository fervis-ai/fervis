"""Parse and validate grounding compatibility reviews."""

from __future__ import annotations

from typing import Any

from fervis.lookup.grounding import provider_contract as provider_output
from fervis.lookup.grounding.model import (
    InputBindingCompatibility,
    InputBindingOption,
    GroundingRequest,
    GroundingCompatibilityResult,
    KnownInputBindingTask,
    LookupTextResolutionDecision,
    TimeResolutionIntent,
    option_has_targetable_identity,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.time_intents import normalize_grounding_date_intent


def parse_grounding_compatibility(
    payload: dict[str, Any],
    *,
    request: GroundingRequest,
) -> GroundingCompatibilityResult:
    output = provider_output.GroundingOutput.parse(payload)
    time_resolutions = _time_resolutions_from_payload(
        output.known_time_resolutions,
        request=request,
    )
    raw_items = output.known_input_binding_reviews
    if not isinstance(raw_items, dict):
        raise ValueError("known_input_binding_reviews must be an object")
    tasks_by_id = {task.known_input_id: task for task in request.tasks}
    options_by_task = {
        task.known_input_id: {option.id: option for option in task.options}
        for task in request.tasks
    }
    compatibilities: list[InputBindingCompatibility] = []
    seen: set[str] = set()
    for raw_known_input_id, raw_review in raw_items.items():
        known_input_id = _text(raw_known_input_id)
        if known_input_id in seen:
            raise ValueError("duplicate grounding review")
        if known_input_id not in tasks_by_id:
            raise ValueError("review references unknown known input")
        review = provider_output.KnownInputBindingReviewOutput.parse(raw_review)
        task = tasks_by_id[known_input_id]
        compatible_option_ids = _compatible_option_ids_from_reviews(
            review.option_reviews,
            task=task,
            options_by_id=options_by_task[known_input_id],
        )
        seen.add(known_input_id)
        compatibilities.append(
            InputBindingCompatibility(
                known_input_id=known_input_id,
                binding_option_ids=compatible_option_ids,
            )
        )
    missing = set(tasks_by_id) - seen
    if missing:
        raise ValueError("grounding compatibility review missing known input")
    return GroundingCompatibilityResult(
        compatibilities=tuple(compatibilities),
        time_resolutions=time_resolutions,
    )


def _time_resolutions_from_payload(
    raw_items: object,
    *,
    request: GroundingRequest,
) -> tuple[TimeResolutionIntent, ...]:
    if not isinstance(raw_items, dict):
        raise ValueError("known_time_resolutions must be an object")
    tasks_by_id = {task.known_input_id: task for task in request.time_tasks}
    if set(raw_items) != set(tasks_by_id):
        raise ValueError("grounding time resolutions must cover every time input")
    output: list[TimeResolutionIntent] = []
    for known_input_id, raw_item in raw_items.items():
        resolution = provider_output.KnownTimeResolutionOutput.parse(raw_item)
        task = tasks_by_id[_text(known_input_id)]
        date_intent = provider_output.DateIntentOutput.parse(resolution.date_intent)
        normalized = normalize_grounding_date_intent(
            {"expression": date_intent.expression, "intent": date_intent.intent},
            path=f"known_time_resolutions.{task.known_input_id}.date_intent",
        )
        expression = str(date_intent.expression or "").strip()
        if expression != task.time_expression:
            raise ValueError("grounding date_intent expression mismatch")
        output.append(
            TimeResolutionIntent(
                known_input_id=task.known_input_id,
                date_intent=normalized,
            )
        )
    return tuple(output)


def _compatible_option_ids_from_reviews(
    raw_reviews: object,
    *,
    task: KnownInputBindingTask,
    options_by_id: dict[str, InputBindingOption],
) -> tuple[str, ...]:
    if not isinstance(raw_reviews, dict):
        raise ValueError("option_reviews must be an object")
    if set(raw_reviews) != set(options_by_id):
        raise ValueError("grounding option reviews must cover every binding option")
    compatible_option_ids: list[str] = []
    for option_id, option in options_by_id.items():
        raw_review = provider_output.OptionReviewOutput.parse(raw_reviews.get(option_id))
        expected_question = resolver_fit_question_for_option(
            task=task,
            option=option,
        )
        if _text(raw_review.resolver_fit_question) != expected_question:
            raise ValueError("grounding resolver_fit_question mismatch")
        decision = _text(raw_review.decision)
        if decision not in {item.value for item in LookupTextResolutionDecision}:
            raise ValueError("unsupported grounding identity decision")
        _text(raw_review.because)
        if decision == LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT.value:
            if not option_has_targetable_identity(option):
                raise ValueError("grounding option cannot return targetable identity")
            compatible_option_ids.append(option_id)
    return tuple(compatible_option_ids)


def _text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("grounding compatibility review requires non-empty text")
    return text
