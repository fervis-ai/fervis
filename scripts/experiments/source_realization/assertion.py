"""Assert selected row, field, and relationship outcomes without rebuilding inputs."""
from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    expected_maps = {
        "set_bindings": ("expected_row_refs", "rows_ref"),
        "fact_bindings": ("expected_field_refs", "field_ref"),
        "association_bindings": ("expected_relation_refs", "realization_ref"),
    }
    if not any(context.get(key) for key, _ in expected_maps.values()):
        return ["source realization assertion requires expected outcomes"]
    errors = []
    for section, (expected_key, field) in expected_maps.items():
        actual = arguments.get(section)
        if not isinstance(actual, dict):
            errors.append(f"{section} is missing")
            continue
        for ref, expected in context.get(expected_key, {}).items():
            values = actual.get(ref)
            if not isinstance(values, list):
                errors.append(f"realization is missing: {ref}")
                continue
            observed = [value.get(field) for value in values if isinstance(value, dict)]
            if observed != expected:
                errors.append(f"{ref} selects {observed!r}; expected {expected!r}")
    return errors
