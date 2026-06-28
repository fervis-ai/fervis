"""Canonical source-binding parameter value helpers."""

from __future__ import annotations

from typing import Any


def canonical_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)
