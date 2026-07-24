"""Strict provider schema for semantic identity grounding."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.grounding import semantic_provider_contract as output
from fervis.lookup.grounding.identity import (
    IdentifierKind,
    InputBindingPurpose,
    InputBindingOption,
    LookupTextResolutionDecision,
    ResourceTypeMatch,
)
from fervis.lookup.grounding.semantic import SemanticGroundingRequest
from fervis.lookup.grounding.surface import (
    ResolverOptionSurface,
    resolver_option_surface_from_catalog,
)
from fervis.lookup.grounding.time_resolution.provider_contract import (
    DateIntentOutput,
    KnownTimeResolutionOutput,
)
from fervis.lookup.grounding.time_resolution.schema import time_intent_schema


def build_semantic_grounding_schema(
    request: SemanticGroundingRequest,
) -> dict[str, object]:
    reviews = {
        task.task_ref: _task_review_schema(request, task_ref=task.task_ref)
        for task in request.tasks
    }
    return output.SemanticGroundingOutput.schema(
        {
            "time_resolutions": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    task.task_ref: KnownTimeResolutionOutput.schema(
                        {
                            "date_intent": DateIntentOutput.schema(
                                {
                                    "expression": {
                                        "type": "string",
                                        "enum": [task.expression],
                                    },
                                    "intent": time_intent_schema(),
                                }
                            )
                        }
                    )
                    for task in request.time_tasks
                },
                "required": [task.task_ref for task in request.time_tasks],
            },
            "reference_reviews": {
                "type": "object",
                "additionalProperties": False,
                "properties": reviews,
                "required": list(reviews),
            },
        }
    )


def _task_review_schema(
    request: SemanticGroundingRequest,
    *,
    task_ref: str,
) -> dict[str, object]:
    task = next(item for item in request.tasks if item.task_ref == task_ref)
    purposes = tuple(
        purpose
        for purpose in InputBindingPurpose
        if any(option.purpose is purpose for option in task.options)
    )
    variants = [
        _purpose_review_schema(request, task_ref=task_ref, purpose=purpose)
        for purpose in purposes
    ]
    return variants[0] if len(variants) == 1 else {"oneOf": variants}


def _purpose_review_schema(
    request: SemanticGroundingRequest,
    *,
    task_ref: str,
    purpose: InputBindingPurpose,
) -> dict[str, object]:
    task = next(item for item in request.tasks if item.task_ref == task_ref)
    purpose_options = tuple(
        option for option in task.options if option.purpose is purpose
    )
    resource_types = tuple(
        sorted({option.candidate.entity_kind for option in purpose_options})
    )
    operand = request.input(task.input_ref).operand
    operands = (operand,) if isinstance(operand, str) else operand
    resource_type_reviews = {
        resource_type: _resource_type_review_schema(
            request,
            options=tuple(
                option
                for option in purpose_options
                if option.candidate.entity_kind == resource_type
            ),
            operands=operands,
        )
        for resource_type in resource_types
    }
    return output.IdentityGroundingReviewOutput.schema(
        {
            "identifier_kind_basis": {"type": "string", "minLength": 1},
            "identifier_kind": {
                "enum": _identifier_kinds_for_purpose(task.options, purpose=purpose)
            },
            "purpose": {"enum": [purpose.value]},
            "resource_type_reviews": {
                "type": "object",
                "additionalProperties": False,
                "properties": resource_type_reviews,
                "required": list(resource_types),
            },
        }
    )


def _identifier_kinds_for_purpose(
    options: tuple[InputBindingOption, ...],
    *,
    purpose: InputBindingPurpose,
) -> list[str]:
    if purpose is InputBindingPurpose.IDENTITY_VALIDATION:
        return [IdentifierKind.PRIMARY_KEY.value]
    if any(
        option.purpose is InputBindingPurpose.IDENTITY_VALIDATION
        for option in options
    ):
        return [IdentifierKind.DESCRIPTIVE.value]
    return [item.value for item in IdentifierKind]


def _resolver_mechanics_schema(
    surface: ResolverOptionSurface,
    *,
    operands: tuple[str, ...],
    allow_positive: bool = True,
) -> dict[str, object]:
    parameters = tuple(
        parameter
        for parameter in surface.selectable_request_parameters
        if all(
            parameter in surface.compatible_request_parameters(lookup_text=operand)
            for operand in operands
        )
    )
    fields = tuple(
        field
        for field in surface.response_match_fields
        if all(
            field in surface.compatible_response_match_fields(lookup_text=operand)
            for operand in operands
        )
    )
    positive_allowed = (
        bool(parameters)
        and bool(fields)
        and all(
            surface.required_request_parameters_accept(lookup_text=operand)
            for operand in operands
        )
    )
    negative = output.ResolverMechanicsOutput.schema(
        {
            "decision": {
                "enum": [LookupTextResolutionDecision.CANNOT_RESOLVE_LOOKUP_TEXT.value]
            },
            "lookup_request_params": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            },
            "returned_identity_verification_fields": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 0,
            },
        }
    )
    if not allow_positive or not positive_allowed:
        return negative
    positive = output.ResolverMechanicsOutput.schema(
        {
            "decision": {
                "enum": [LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT.value]
            },
            "lookup_request_params": {
                "type": "array",
                "items": {"enum": [item.param_ref for item in parameters]},
                "minItems": 1,
                "maxItems": len(parameters),
                "uniqueItems": True,
            },
            "returned_identity_verification_fields": {
                "type": "array",
                "items": {"enum": [item.path for item in fields]},
                "minItems": 1,
                "maxItems": len(fields),
                "uniqueItems": True,
            },
        }
    )
    return {"oneOf": [negative, positive]}


def _resource_type_review_schema(
    request: SemanticGroundingRequest,
    *,
    options: tuple[InputBindingOption, ...],
    operands: tuple[str, ...],
) -> dict[str, object]:
    def route_reviews(*, allow_positive: bool) -> dict[str, object]:
        properties = {
            option.id: output.IdentityRouteReviewOutput.schema(
                {
                    "assessment_basis": {"type": "string", "minLength": 1},
                    "resolution": _resolver_mechanics_schema(
                        resolver_option_surface_from_catalog(
                            request.resolver_catalog,
                            option,
                        ),
                        operands=operands,
                        allow_positive=allow_positive,
                    ),
                }
            )
            for option in options
        }
        return _closed_object(properties)

    return {
        "oneOf": [
            output.ResourceTypeReviewOutput.schema(
                {
                    "compatibility_basis": {"type": "string", "minLength": 1},
                    "compatibility": {
                        "enum": [ResourceTypeMatch.POSSIBLE_DENOTED_KIND.value]
                    },
                    "route_reviews": route_reviews(allow_positive=True),
                }
            ),
            output.ResourceTypeReviewOutput.schema(
                {
                    "compatibility_basis": {"type": "string", "minLength": 1},
                    "compatibility": {
                        "enum": [ResourceTypeMatch.UNRELATED_RESOURCE_KIND.value]
                    },
                    "route_reviews": route_reviews(allow_positive=False),
                }
            ),
        ]
    }


def _closed_object(properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


__all__ = ["build_semantic_grounding_schema"]
