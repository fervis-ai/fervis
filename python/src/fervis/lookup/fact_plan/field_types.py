"""Shared source-field type predicates for fact planning."""

from __future__ import annotations

from typing import Any


def field_is_numeric(field: Any) -> bool:
    if field is None:
        return False
    return str(getattr(field, "type", "") or "").lower() in {
        "integer",
        "number",
        "decimal",
        "float",
        "double",
    }
