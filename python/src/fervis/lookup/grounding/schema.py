"""Provider schema for grounding compatibility reviews."""

from __future__ import annotations

from fervis.lookup.grounding import provider_contract as provider_output
from fervis.lookup.grounding.model import (
    GroundingRequest,
    LookupTextResolutionDecision,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.surface import (
    ResolverOptionSurface,
    resolver_option_surface,
)
from fervis.lookup.grounding.time_intents import TIME_INTENT_FIELDS


def build_grounding_schema(request: GroundingRequest) -> dict[str, object]:
    review_properties: dict[str, object] = {}
    for task in request.tasks:
        option_review_properties = {
            option.id: _option_review_schema(
                resolver_fit_question=resolver_fit_question_for_option(
                    task=task,
                    option=option,
                ),
                surface=resolver_option_surface(request, option),
                lookup_text=task.lookup_text,
            )
            for option in task.options
        }
        review_properties[task.known_input_id] = (
            provider_output.KnownInputBindingReviewOutput.schema(
                {
                    "option_reviews": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": option_review_properties,
                        "required": list(option_review_properties),
                    }
                }
            )
        )
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
            "known_input_binding_reviews": {
                "type": "object",
                "additionalProperties": False,
                "properties": review_properties,
                "required": [task.known_input_id for task in request.tasks],
            },
        },
    )


def _option_review_schema(
    *,
    resolver_fit_question: str,
    surface: ResolverOptionSurface,
    lookup_text: str,
) -> dict[str, object]:
    common = {
        "resolver_fit_question": {
            "type": "string",
            "enum": [resolver_fit_question],
        },
        "because": {"type": "string", "minLength": 1},
    }
    variants = []
    if surface.response_match_fields and surface.required_request_parameters_accept(
        lookup_text=lookup_text
    ):
        variants.append(
            provider_output.OptionReviewOutput.schema(
                {
                    **common,
                    "request_values": _request_values_schema(
                        surface,
                        lookup_text=lookup_text,
                    ),
                    "response_match_alternatives": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                field.path for field in surface.response_match_fields
                            ],
                        },
                        "minItems": 1,
                    },
                    "decision": {
                        "type": "string",
                        "enum": [
                            LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT.value
                        ],
                    },
                }
            )
        )
    variants.append(
        provider_output.OptionReviewOutput.schema(
            {
                **common,
                "request_values": {
                    **_empty_object_schema(),
                },
                "response_match_alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 0,
                },
                "decision": {
                    "type": "string",
                    "enum": [
                        LookupTextResolutionDecision.CANNOT_RESOLVE_LOOKUP_TEXT.value
                    ],
                },
            }
        )
    )
    return {"oneOf": variants}


def _request_values_schema(
    surface: ResolverOptionSurface,
    *,
    lookup_text: str,
) -> dict[str, object]:
    parameters = surface.compatible_request_parameters(lookup_text=lookup_text)
    properties = {
        parameter.param_ref: {
            "enum": [
                surface.compiled_request_value(
                    parameter.param_ref,
                    lookup_text=lookup_text,
                )[1]
            ]
        }
        for parameter in parameters
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": [
            parameter.param_ref
            for parameter in parameters
            if parameter.required and parameter.default is None
        ],
    }


def _empty_object_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }


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
