"""Outcome assertions for a captured production Read Eligibility turn."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    if context.get("identity_task_ref") is None and not context.get(
        "expected_read_decisions"
    ):
        return ["read eligibility assertion requires an expected decision"]
    errors = _identity_errors(arguments, context)
    expected_read_decisions = context.get("expected_read_decisions", {})
    if expected_read_decisions:
        requested_fact_ref = context.get("requested_fact_ref")
        assessments_by_fact = arguments.get(
            "read_assessments_by_requested_fact"
        )
        assessments = (
            assessments_by_fact.get(requested_fact_ref)
            if isinstance(assessments_by_fact, dict)
            and isinstance(requested_fact_ref, str)
            else None
        )
        for read_ref, expected in expected_read_decisions.items():
            assessment = (
                assessments.get(read_ref) if isinstance(assessments, dict) else None
            )
            actual = (
                assessment.get("decision") if isinstance(assessment, dict) else None
            )
            if actual != expected:
                errors.append(
                    f"{read_ref} decision is {actual!r}; expected {expected!r}"
                )
    return errors


def _identity_errors(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    task_ref = context.get("identity_task_ref")
    if task_ref is None:
        return []
    outcomes = arguments.get("identity_outcomes")
    if not isinstance(outcomes, dict):
        return ["identity_outcomes is missing"]
    outcome = outcomes.get(task_ref)
    if not isinstance(outcome, dict):
        return [f"identity outcome is missing for {task_ref!r}"]
    errors = []
    for field, context_key in (
        ("canonical_option_id", "expected_canonical_option_id"),
        ("resolver_route_id", "expected_resolver_route_id"),
    ):
        expected = context.get(context_key)
        if expected is not None and outcome.get(field) != expected:
            errors.append(f"{field} is {outcome.get(field)!r}; expected {expected!r}")
    return errors


__all__ = ["validate"]
