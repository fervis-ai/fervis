"""Outcome assertions for captured Conversation Resolution invocations."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    outcome = arguments.get("outcome")
    expected_kind = context.get("expected_outcome_kind", "resolved")
    if not isinstance(outcome, dict) or outcome.get("kind") != expected_kind:
        actual = outcome.get("kind") if isinstance(outcome, dict) else None
        return [f"outcome is {actual!r}; expected {expected_kind!r}"]
    if expected_kind != "resolved":
        return []

    expected_question = context.get("expected_contextualized_question")
    if (
        expected_question is not None
        and outcome.get("contextualized_question") != expected_question
    ):
        return [
            "contextualized question is "
            f"{outcome.get('contextualized_question')!r}; expected {expected_question!r}"
        ]
    contextualized_question = str(outcome.get("contextualized_question") or "")
    required_terms = set(context.get("required_contextualized_terms") or ())
    missing_terms = sorted(
        term for term in required_terms if str(term).casefold() not in contextualized_question.casefold()
    )
    if missing_terms:
        return [f"contextualized question omits {missing_terms!r}"]

    clauses = outcome.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        return ["resolved conversation lacks clauses"]

    expected_shape_source = context.get("expected_request_shape_source")
    if expected_shape_source is not None:
        actual_sources = [
            clause.get("request_shape_source")
            for clause in clauses
            if isinstance(clause, dict)
        ]
        if actual_sources != [expected_shape_source] * len(clauses):
            return [
                f"request shape sources are {actual_sources!r}; "
                f"expected {expected_shape_source!r}"
            ]

    expected_values = set(context.get("expected_resolved_values") or ())
    actual_values = {
        str(value.get("resolved_text"))
        for clause in clauses
        if isinstance(clause, dict)
        for value in clause.get("values") or ()
        if isinstance(value, dict)
    }
    missing_values = sorted(expected_values - actual_values)
    if missing_values:
        return [f"resolved values omit {missing_values!r}"]
    return []


__all__ = ["validate"]
