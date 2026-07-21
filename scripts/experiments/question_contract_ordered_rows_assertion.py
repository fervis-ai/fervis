"""Validate ordered selection over qualifying rows without aggregation."""

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
    if expression.get("family") != "list_rows":
        errors.append(f"family is not list_rows: {expression.get('family')!r}")
    if not expression.get("ordering"):
        errors.append("ordering is missing")
    if (expression.get("selection") or {}).get("kind") != "take_one":
        errors.append(f"unexpected selection: {expression.get('selection')!r}")
    return errors
