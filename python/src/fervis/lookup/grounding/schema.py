"""Provider schema for grounding compatibility reviews."""

from __future__ import annotations

from fervis.lookup.grounding import provider_contract as provider_output
from fervis.lookup.grounding.model import (
    GroundingRequest,
    IdentifierKind,
    LookupTextResolutionDecision,
    ResourceTypeMatch,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.surface import (
    ResolverOptionSurface,
    resolver_option_surface,
)
from fervis.lookup.grounding.time_intents import TIME_INTENT_FIELDS
from fervis.lookup.fact_plan.row_sources import RowSourceField, RowSourceParam


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
                    "resource_type_basis": {"type": "string", "minLength": 1},
                    "resource_type_compatibility": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            resource_type: {
                                "type": "string",
                                "enum": [match.value for match in ResourceTypeMatch],
                            }
                            for resource_type in task.shown_resource_types
                        },
                        "required": list(task.shown_resource_types),
                    },
                    "identifier_kind_basis": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "identifier_kind": {
                        "type": "string",
                        "enum": [kind.value for kind in IdentifierKind],
                    },
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
    properties: dict[str, object] = {
        "resource_type": {
            "type": "string",
            "enum": [surface.option.candidate.entity_kind],
        },
        "resolver_fit_question": {
            "type": "string",
            "enum": [resolver_fit_question],
        },
        "because": {"type": "string", "minLength": 1},
    }
    compatible_parameters = surface.compatible_request_parameters(
        lookup_text=lookup_text
    )
    compatible_response_fields = surface.compatible_response_match_fields(
        lookup_text=lookup_text
    )
    positive_allowed = (
        bool(compatible_response_fields)
        and bool(compatible_parameters)
        and surface.required_request_parameters_accept(
            lookup_text=lookup_text
        )
    )
    resolution_schema: dict[str, object]
    if positive_allowed:
        resolution_schema = {
            "oneOf": [
                _negative_resolution_schema(),
                _positive_resolution_schema(
                    surface,
                    lookup_text=lookup_text,
                    parameters=compatible_parameters,
                    response_fields=compatible_response_fields,
                ),
            ]
        }
    else:
        resolution_schema = _negative_resolution_schema()
    properties["resolution"] = resolution_schema
    return provider_output.OptionReviewOutput.schema(properties)


def _positive_resolution_schema(
    surface: ResolverOptionSurface,
    *,
    lookup_text: str,
    parameters: tuple[RowSourceParam, ...],
    response_fields: tuple[RowSourceField, ...],
) -> dict[str, object]:
    parameter_schemas = [
        provider_output.LookupRequestParamOutput.schema(
            {
                "param_ref": {
                    "type": "string",
                    "enum": [parameter.param_ref],
                },
                "value": {
                    "enum": [
                        surface.compiled_request_value(
                            parameter.param_ref,
                            lookup_text=lookup_text,
                        )[1]
                    ]
                },
            }
        )
        for parameter in parameters
    ]
    parameter_item_schema = (
        parameter_schemas[0]
        if len(parameter_schemas) == 1
        else {"oneOf": parameter_schemas}
    )
    return provider_output.ResolverResolutionOutput.schema(
        {
            "decision": {
                "type": "string",
                "enum": [
                    LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT.value
                ],
            },
            "lookup_request_params": {
                "type": "array",
                "items": parameter_item_schema,
                "minItems": 1,
                "maxItems": len(parameters),
            },
            "returned_identity_verification_fields": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [field.path for field in response_fields],
                },
                "minItems": 1,
                "maxItems": len(response_fields),
                "uniqueItems": True,
            },
        }
    )


def _negative_resolution_schema() -> dict[str, object]:
    return provider_output.ResolverResolutionOutput.schema(
        {
            "decision": {
                "type": "string",
                "enum": [
                    LookupTextResolutionDecision.CANNOT_RESOLVE_LOOKUP_TEXT.value
                ],
            },
            "lookup_request_params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
                "maxItems": 0,
            },
            "returned_identity_verification_fields": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            },
        }
    )


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
