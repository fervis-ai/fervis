"""Shared serialization helpers for memory payloads."""

from __future__ import annotations

from typing import Any


def without_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ("", None, {}, [], ())
    }
