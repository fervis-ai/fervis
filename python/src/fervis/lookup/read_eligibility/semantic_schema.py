"""Strict provider schema for semantic Read Eligibility."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.read_eligibility import semantic_provider_contract as output
from fervis.lookup.read_eligibility.semantic import SemanticReadEligibilityRequest


def build_semantic_read_eligibility_schema(
    request: SemanticReadEligibilityRequest,
) -> dict[str, object]:
    reads = {
        candidate.candidate_ref: _read_assessment_schema(
            tuple(field.field_ref for field in candidate.fields)
        )
        for candidate in request.read_candidates
    }
    outcomes = {
        task.task_ref: _identity_outcome_schema(request, task_ref=task.task_ref)
        for task in request.identity_tasks
    }
    return output.SemanticReadEligibilityOutput.schema(
        {
            "read_assessments_by_requested_fact": _closed_object(
                {
                    index.requested_fact_id: _closed_object(reads)
                    for index in request.indexes
                }
            ),
            "identity_outcomes": _closed_object(outcomes),
        }
    )


def _read_assessment_schema(field_refs: tuple[str, ...]) -> dict[str, object]:
    common = {"assessment_basis": {"type": "string", "minLength": 1}}
    return {
        "oneOf": [
            output.ReadRequirementAssessmentOutput.schema(
                {
                    **common,
                    "relevant_field_refs": _bounded_array(field_refs),
                    "decision": {"enum": ["RETAIN"]},
                }
            ),
            output.ReadRequirementAssessmentOutput.schema(
                {
                    **common,
                    "relevant_field_refs": _bounded_array(()),
                    "decision": {"enum": ["DROP"]},
                }
            ),
        ]
    }


def _identity_outcome_schema(
    request: SemanticReadEligibilityRequest,
    *,
    task_ref: str,
) -> dict[str, object]:
    task = next(item for item in request.identity_tasks if item.task_ref == task_ref)
    option_ids = tuple(item.canonical_option_id for item in task.canonical_options)
    route_ids = tuple(item.route_ref for item in task.resolver_routes)
    return output.IdentityRouteOutcomeOutput.schema(
        {
            "canonical_option_assessments": {
                "type": "array",
                "minItems": len(option_ids),
                "maxItems": len(option_ids),
                "items": output.CanonicalOptionAssessmentOutput.schema(
                    {
                        "canonical_option_id": {"enum": list(option_ids)},
                        "assessment": {"type": "string", "minLength": 1},
                        "decision": {"enum": ["FITS", "DOES_NOT_FIT"]},
                    }
                ),
            },
            "canonical_option_basis": {"type": "string", "minLength": 1},
            "canonical_option_id": _nullable_enum(option_ids),
            "resolver_route_assessments": {
                "type": "array",
                "minItems": len(route_ids),
                "maxItems": len(route_ids),
                "items": output.ResolverRouteAssessmentOutput.schema(
                    {
                        "resolver_route_id": {"enum": list(route_ids)},
                        "assessment": {"type": "string", "minLength": 1},
                        "decision": {"enum": ["FITS", "DOES_NOT_FIT"]},
                    }
                ),
            },
            "resolver_route_basis": {"type": "string", "minLength": 1},
            "resolver_route_id": _nullable_enum(route_ids),
            "evidence_refs": _bounded_array(
                tuple(
                    dict.fromkeys(
                        (
                            task.task_ref,
                            *(item.canonical_option_id for item in task.canonical_options),
                            *(item.route_ref for item in task.resolver_routes),
                            *(candidate.candidate_ref for candidate in request.read_candidates),
                        )
                    )
                )
            ),
            "outcome": {
                "enum": [
                    "SELECTED",
                    "NO_CANONICAL_INTERPRETATION",
                    "NO_RESOLVER_ROUTE",
                ]
            },
        }
    )


def _bounded_array(values: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "array",
        "minItems": 0,
        "maxItems": len(values),
        "items": {"type": "string", "enum": list(values)},
    }


def _nullable_enum(values: tuple[str, ...]) -> dict[str, object]:
    return {
        "anyOf": [
            {"type": "string", "enum": list(values)},
            {"type": "null"},
        ]
    }


def _closed_object(properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


__all__ = ["build_semantic_read_eligibility_schema"]
