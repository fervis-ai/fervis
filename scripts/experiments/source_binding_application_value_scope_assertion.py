"""Validate the compact Source Binding application-target contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    errors: list[str] = []
    for item in _objects(arguments):
        if "value_component" not in item or "match_basis_explanation" not in item:
            continue
        application_target_id = item.get("application_target_id")
        if not isinstance(application_target_id, str) or not application_target_id:
            errors.append(
                "resolved input application has no application_target_id"
            )
        stale_fields = tuple(
            field
            for field in ("target_kind", "target_id", "application_value_id")
            if field in item
        )
        if stale_fields:
            errors.append(
                "resolved input application uses stale independent fields: "
                + ", ".join(stale_fields)
            )
    return errors


def _objects(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)
