"""Provider schema for grounding compatibility reviews."""

from __future__ import annotations

from fervis.lookup.grounding import provider_contract as provider_output
from fervis.lookup.grounding.model import (
    GroundingRequest,
    InputBindingOption,
    InputBindingPurpose,
    InputBindingResultKind,
    KnownInputBindingTask,
)
from fervis.lookup.grounding.time_intents import TIME_INTENT_FIELDS


def build_grounding_schema(request: GroundingRequest) -> dict[str, object]:
    review_properties: dict[str, object] = {}
    for binding_task in request.tasks:
        review_properties[binding_task.known_input_id] = {
            "oneOf": [
                *(
                    schema
                    for option in binding_task.options
                    for schema in _selected_option_schemas(binding_task, option)
                ),
                _no_selected_option_schema(),
            ]
        }
    time_resolution_properties: dict[str, object] = {}
    for time_task in request.time_tasks:
        time_resolution_properties[time_task.known_input_id] = (
            provider_output.KnownTimeResolutionOutput.schema(
                {
                    "date_intent": provider_output.DateIntentOutput.schema(
                        {
                            "expression": {
                                "type": "string",
                                "enum": [time_task.time_expression],
                            },
                            "intent": _time_intent_schema(),
                        }
                    )
                }
            )
        )
    return provider_output.GroundingOutput.schema(
        {
            "known_time_resolutions": {
                "type": "object",
                "additionalProperties": False,
                "properties": time_resolution_properties,
                "required": [task.known_input_id for task in request.time_tasks],
            },
            "known_input_bindings": {
                "type": "object",
                "additionalProperties": False,
                "properties": review_properties,
                "required": [task.known_input_id for task in request.tasks],
            },
        },
    )


def _selected_option_schemas(
    task: KnownInputBindingTask,
    option: InputBindingOption,
) -> tuple[dict[str, object], ...]:
    if option.purpose is InputBindingPurpose.IDENTITY_VALIDATION:
        return (
            _selected_option_schema(
                option,
                result_kind=InputBindingResultKind.CANONICAL_IDENTITY,
                input_value_schema=_identity_validation_value_schema(option),
            ),
        )
    route = option.route
    if route is None:
        raise ValueError("grounding option requires a route")
    schemas: list[dict[str, object]] = []
    if (
        route.lookup_param_ref
        or not route.lookup_field_ids
        or route.identity_lookup_field_ids
    ):
        schemas.append(
            _selected_option_schema(
                option,
                result_kind=InputBindingResultKind.CANONICAL_IDENTITY,
                input_value_schema={"type": "string", "enum": [task.lookup_text]},
            )
        )
    schemas.extend(
        _matched_field_option_schema(option, field_ref=field_ref)
        for field_ref in dict.fromkeys(route.lookup_field_refs)
    )
    return tuple(schemas)


def _matched_field_option_schema(
    option: InputBindingOption,
    *,
    field_ref: str,
) -> dict[str, object]:
    route = option.route
    if route is None:
        raise ValueError("grounding option requires a route")
    field = next(
        (item for item in route.selected_output_fields if item.field_ref == field_ref),
        None,
    )
    input_value_schema: dict[str, object] = {"type": "string", "minLength": 1}
    if field is not None and field.choices:
        input_value_schema = {"type": "string", "enum": list(field.choices)}
    return _selected_option_schema(
        option,
        result_kind=InputBindingResultKind.MATCHED_VALUE,
        input_value_schema=input_value_schema,
        matched_field_ref=field_ref,
    )


def _selected_option_schema(
    option: InputBindingOption,
    *,
    result_kind: InputBindingResultKind,
    input_value_schema: dict[str, object],
    matched_field_ref: str = "",
) -> dict[str, object]:
    properties = {
        "selected_option_id": {"enum": [option.id]},
        "input_value": input_value_schema,
        "result_kind": {"enum": [result_kind.value]},
        "selection_basis": {"type": "string", "minLength": 1},
    }
    if matched_field_ref:
        properties["matched_field_ref"] = {"enum": [matched_field_ref]}
    schema = provider_output.KnownInputBindingOutput.schema(properties)
    if matched_field_ref:
        required = schema["required"]
        if not isinstance(required, list):
            raise ValueError("grounding schema required fields must be an array")
        schema["required"] = [*required, "matched_field_ref"]
    return schema


def _no_selected_option_schema() -> dict[str, object]:
    return provider_output.KnownInputBindingOutput.schema(
        {
            "selected_option_id": {"enum": ["none"]},
            "input_value": {"type": "string", "enum": [""]},
            "result_kind": {"enum": ["none"]},
            "selection_basis": {"type": "string", "minLength": 1},
        }
    )


def _identity_validation_value_schema(
    option: InputBindingOption,
) -> dict[str, object]:
    route = option.route
    type_name = route.lookup_param_type.casefold() if route is not None else ""
    if type_name == "integer":
        return {"type": "integer"}
    if type_name in {"number", "double", "float"}:
        return {"type": "number"}
    if type_name == "boolean":
        return {"type": "boolean"}
    return {"type": "string", "minLength": 1}


def _time_intent_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _time_intent_variant(
                time_shape="point_date",
                unit={"enum": ["none", "day"]},
                mode={"enum": ["none"]},
                year={"type": "integer", "minimum": 1},
                month={"type": "integer", "minimum": 1, "maximum": 12},
                day={"type": "integer", "minimum": 1, "maximum": 31},
                year_policy={"enum": ["none"]},
            ),
            _time_intent_variant(
                time_shape="point_date",
                unit={"enum": ["none", "day"]},
                mode={"enum": ["none"]},
                year={"enum": [0]},
                month={"type": "integer", "minimum": 1, "maximum": 12},
                day={"type": "integer", "minimum": 1, "maximum": 31},
                year_policy={"enum": ["most_recent"]},
            ),
            _time_intent_variant(
                time_shape="point_relative",
                unit={"enum": ["day"]},
                mode={"enum": ["none"]},
                relative_offset={"type": "integer"},
            ),
            _time_intent_variant(
                time_shape="period_relative",
                unit={"enum": ["day", "week", "month", "quarter", "year"]},
                mode={"enum": ["full", "to_date"]},
                relative_offset={"type": "integer"},
            ),
            _time_intent_variant(
                time_shape="period_named",
                unit={"enum": ["month", "quarter", "year"]},
                mode={"enum": ["full", "to_date"]},
                year={"type": "integer", "minimum": 0},
                year_policy={"enum": ["none", "most_recent"]},
                named_value={"type": "integer", "minimum": 1},
            ),
            _time_intent_variant(
                time_shape="range",
                unit={"enum": ["none"]},
                mode={"enum": ["none"]},
                year={"type": "integer", "minimum": 0},
                month={"type": "integer", "minimum": 1, "maximum": 12},
                day={"type": "integer", "minimum": 1, "maximum": 31},
                year_policy={"enum": ["none", "most_recent"]},
                end_year={"type": "integer", "minimum": 0},
                end_month={"type": "integer", "minimum": 1, "maximum": 12},
                end_day={"type": "integer", "minimum": 1, "maximum": 31},
                end_year_policy={"enum": ["none", "most_recent"]},
            ),
            _time_intent_variant(
                time_shape="open_range",
                unit={"enum": ["none"]},
                mode={"enum": ["none"]},
                year={"type": "integer", "minimum": 0},
                month={"type": "integer", "minimum": 1, "maximum": 12},
                day={"type": "integer", "minimum": 1, "maximum": 31},
                year_policy={"enum": ["none", "most_recent"]},
            ),
            _time_intent_variant(
                time_shape="window",
                unit={"enum": ["day", "week", "month"]},
                mode={"enum": ["none"]},
                count={"type": "integer", "minimum": 1},
                direction={"enum": ["past", "future"]},
            ),
        ]
    }


def _time_intent_variant(
    *,
    time_shape: str,
    unit: dict[str, object],
    mode: dict[str, object],
    **overrides: dict[str, object],
) -> dict[str, object]:
    properties: dict[str, object] = {
        "time_shape": {"enum": [time_shape]},
        "unit": unit,
        "mode": mode,
        "year": {"enum": [0]},
        "month": {"enum": [0]},
        "day": {"enum": [0]},
        "year_policy": {"enum": ["none"]},
        "relative_offset": {"enum": [0]},
        "named_value": {"enum": [0]},
        "end_year": {"enum": [0]},
        "end_month": {"enum": [0]},
        "end_day": {"enum": [0]},
        "end_year_policy": {"enum": ["none"]},
        "count": {"enum": [0]},
        "direction": {"enum": ["none"]},
    }
    properties.update(overrides)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(TIME_INTENT_FIELDS),
    }
