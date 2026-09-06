"""Validate returned-value roles, then reuse the production frame assertion."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.experiments.question_frame.assertion import validate as validate_frame


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors = _value_kind_errors(arguments, context=context)
    production_arguments = deepcopy(arguments)
    _remove_value_kinds(production_arguments)
    return [*errors, *validate_frame(production_arguments, context)]


def _value_kind_errors(
    arguments: dict[str, Any], *, context: dict[str, Any]
) -> list[str]:
    expected = context.get("expected_returned_value_kinds") or ()
    if not expected:
        return []
    outcome = arguments.get("outcome")
    requests = outcome.get("answer_requests") if isinstance(outcome, dict) else None
    if not isinstance(requests, list):
        return ["answer_requests is missing"]
    actual = [_returned_value_kinds(item) for item in requests]
    if actual != expected:
        return [f"returned value kinds are {actual!r}; expected {expected!r}"]
    return []


def _returned_value_kinds(request: object) -> list[str]:
    if not isinstance(request, dict):
        return []
    body = request.get("request")
    if not isinstance(body, dict):
        return []
    result = body.get("result")
    if not isinstance(result, dict):
        return []
    projection = result.get("projection")
    if not isinstance(projection, dict):
        return []
    values = projection.get("explicitly_requested_values")
    if not isinstance(values, list):
        return []
    return [
        str(item.get("value_kind")) for item in values if isinstance(item, dict)
    ]


def _remove_value_kinds(value: object) -> None:
    if isinstance(value, list):
        for item in value:
            _remove_value_kinds(item)
        return
    if not isinstance(value, dict):
        return
    values = value.get("explicitly_requested_values")
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                item.pop("value_kind", None)
                item.pop("value_kind_basis", None)
    for item in value.values():
        _remove_value_kinds(item)


__all__ = ["validate"]
