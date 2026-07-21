"""Validate the Question Contract boundary for an ordered extremum request."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    outcome = arguments.get("outcome") or {}
    requests = outcome.get("answer_requests") or []
    if len(requests) != 1:
        return [f"expected one answer request, got {len(requests)}"]
    expression = requests[0].get("answer_expression") or {}
    errors: list[str] = []
    if expression.get("family") not in {"list_rows", "grouped_aggregate"}:
        errors.append(f"unexpected family: {expression.get('family')!r}")
    if not expression.get("ordering"):
        errors.append("ordering is missing")
    if (expression.get("selection") or {}).get("kind") != "take_one":
        errors.append(f"unexpected selection: {expression.get('selection')!r}")
    result_limits = [
        item
        for item in outcome.get("question_inputs") or []
        if item.get("role") == "result_limit"
    ]
    if result_limits:
        errors.append("take_one invented a result_limit input")
    return errors
