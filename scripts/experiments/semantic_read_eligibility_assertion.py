"""Assertions for the semantic Read Eligibility boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_semantic_read_eligibility_boundary import build_request
from fervis.lookup.read_eligibility.semantic import (
    IdentityRouteSelection,
    SemanticReadDecision,
)
from fervis.lookup.read_eligibility.semantic_parser import (
    parse_semantic_read_eligibility,
)


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    request = build_request(context)
    try:
        result = parse_semantic_read_eligibility(arguments, request=request)
    except ValueError as exc:
        return [f"semantic read eligibility parser rejected output: {exc}"]
    if context.get("kind") == "returned_identity":
        assessment = next(
            item
            for item in result.read_assessments
            if item.read_id == str(context["expected_read"])
        )
        if assessment.decision is not SemanticReadDecision.RETAIN:
            return ["the identity-bearing answer source was not retained"]
        retained_fields = set(assessment.relevant_field_refs)
        expected_field = str(context["expected_identity_field"])
        expected_value_field = str(context["expected_value_field"])
        if not {expected_field, expected_value_field} <= retained_fields:
            return [
                "retained fields do not include the expected identity and value fields: "
                f"{sorted(retained_fields)!r}"
            ]
        return []
    if context.get("kind") == "boolean_coverage":
        answer_assessment = next(
            item for item in result.read_assessments if item.read_id == "list_measurement"
        )
        if answer_assessment.decision is not SemanticReadDecision.RETAIN:
            return ["the answer source was not retained"]
        if result.identity_outcomes:
            return ["the scalar predicate produced an identity outcome"]
        return []
    if any(
        item.decision is not SemanticReadDecision.RETAIN
        for item in result.read_assessments
    ):
        return ["the answer source was not retained"]
    [outcome] = result.identity_outcomes
    if not isinstance(outcome, IdentityRouteSelection):
        return ["identity selection returned clarification"]
    route = next(
        item
        for task in request.identity_tasks
        for item in task.resolver_routes
        if item.route_ref == outcome.resolver_route_id
    )
    if route.option.candidate.resolver_read_id != str(context["expected_read"]):
        return ["the selected resolver route is not the expected read"]
    return []


__all__ = ["validate"]
