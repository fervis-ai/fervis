"""Parse and validate grounding compatibility reviews."""

from __future__ import annotations

from fervis.lookup.grounding import provider_contract as provider_output
from fervis.lookup.grounding.model import (
    InputBindingSelection,
    InputBindingOption,
    InputBindingPurpose,
    InputBindingResultKind,
    GroundingRequest,
    GroundingSelectionResult,
    KnownInputBindingTask,
    ResolvedInputBinding,
    TimeResolutionIntent,
)
from fervis.lookup.grounding.time_intents import normalize_grounding_date_intent


def parse_grounding_compatibility(
    payload: dict[str, object],
    *,
    request: GroundingRequest,
) -> GroundingSelectionResult:
    output = provider_output.GroundingOutput.parse(payload)
    time_resolutions = _time_resolutions_from_payload(
        output.known_time_resolutions,
        request=request,
    )
    reviews = output.known_input_bindings
    tasks_by_id = {task.known_input_id: task for task in request.tasks}
    options_by_task = {
        task.known_input_id: {option.id: option for option in task.options}
        for task in request.tasks
    }
    selections: list[InputBindingSelection] = []
    seen: set[str] = set()
    for known_input_id, review in reviews.items():
        if known_input_id in seen:
            raise ValueError("duplicate grounding review")
        if known_input_id not in tasks_by_id:
            raise ValueError("review references unknown known input")
        task = tasks_by_id[known_input_id]
        binding = _selected_binding(
            review,
            task=task,
            options_by_id=options_by_task[known_input_id],
        )
        seen.add(known_input_id)
        selections.append(
            InputBindingSelection(
                known_input_id=known_input_id,
                binding=binding,
            )
        )
    missing = set(tasks_by_id) - seen
    if missing:
        raise ValueError("grounding compatibility review missing known input")
    return GroundingSelectionResult(
        selections=tuple(selections),
        time_resolutions=time_resolutions,
    )


def _time_resolutions_from_payload(
    resolutions: dict[str, provider_output.KnownTimeResolutionOutput],
    *,
    request: GroundingRequest,
) -> tuple[TimeResolutionIntent, ...]:
    tasks_by_id = {task.known_input_id: task for task in request.time_tasks}
    if set(resolutions) != set(tasks_by_id):
        raise ValueError("grounding time resolutions must cover every time input")
    output: list[TimeResolutionIntent] = []
    for known_input_id, resolution in resolutions.items():
        task = tasks_by_id[known_input_id]
        date_intent = resolution.date_intent
        expression = _required_text(date_intent.expression)
        normalized = normalize_grounding_date_intent(
            expression,
            date_intent.intent,
            path=f"known_time_resolutions.{task.known_input_id}.date_intent",
        )
        if expression != task.time_expression:
            raise ValueError("grounding date_intent expression mismatch")
        output.append(
            TimeResolutionIntent(
                known_input_id=task.known_input_id,
                date_intent=normalized,
            )
        )
    return tuple(output)


def _selected_binding(
    review: provider_output.KnownInputBindingOutput,
    *,
    task: KnownInputBindingTask,
    options_by_id: dict[str, InputBindingOption],
) -> ResolvedInputBinding | None:
    option_id = _required_text(review.selected_option_id)
    _required_text(review.selection_basis)
    if option_id == "none":
        if review.input_value != "" or review.result_kind != "none":
            raise ValueError("unselected grounding input must not supply a value")
        return None
    option = options_by_id.get(option_id)
    if option is None or option.route is None:
        raise ValueError("grounding selected an unknown binding option")
    result_kind = InputBindingResultKind(review.result_kind)
    input_value = _option_input_value(
        review.input_value,
        task=task,
        option=option,
        result_kind=result_kind,
    )
    if (
        option.purpose is InputBindingPurpose.IDENTITY_VALIDATION
        and result_kind is not InputBindingResultKind.CANONICAL_IDENTITY
    ):
        raise ValueError("identity validation must return canonical identity")
    matched_field_ref = (review.matched_field_ref or "").strip()
    if option.purpose is InputBindingPurpose.REFERENCE_GROUNDING:
        allowed_field_refs = (
            option.route.canonical_lookup_field_refs
            if result_kind is InputBindingResultKind.CANONICAL_IDENTITY
            else option.route.lookup_field_refs
        )
        if allowed_field_refs and matched_field_ref not in allowed_field_refs:
            raise ValueError("reference grounding requires a selected lookup field")
        if not allowed_field_refs and matched_field_ref:
            raise ValueError("grounding selected an unexpected lookup field")
    elif matched_field_ref:
        raise ValueError("grounding selected an unexpected lookup field")
    return ResolvedInputBinding(
        option_id=option_id,
        input_value=input_value,
        result_kind=result_kind,
        matched_field_ref=matched_field_ref,
    )


def _option_input_value(
    value: str | int | float | bool,
    *,
    task: KnownInputBindingTask,
    option: InputBindingOption,
    result_kind: InputBindingResultKind,
) -> str | int | float | bool:
    if option.purpose is InputBindingPurpose.REFERENCE_GROUNDING:
        input_text = str(value).strip()
        if not input_text:
            raise ValueError("reference grounding requires a non-empty input value")
        if (
            result_kind is InputBindingResultKind.CANONICAL_IDENTITY
            and value != task.lookup_text
        ):
            raise ValueError(
                "canonical identity grounding must use the declared lookup text"
            )
        return value
    input_text = str(value).strip()
    if not input_text:
        raise ValueError("identity validation requires a path-key input value")
    if input_text.casefold() not in task.lookup_text.casefold():
        raise ValueError("identity-validation input must come from the known input")
    return value


def _required_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("grounding compatibility review requires non-empty text")
    return text
