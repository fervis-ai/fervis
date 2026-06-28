"""Shared source-binding parser payload helpers."""

from __future__ import annotations

from typing import Any


def _dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _required_dicts(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(_dict(item, label) for item in value)


def _dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("value must be a list")
    return tuple(_dict(item, "value") for item in value)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("value must be a list")
    return tuple(str(item) for item in value if str(item))


def _required_strings(value: Any, label: str) -> tuple[str, ...]:
    output = _strings(value)
    if not output:
        raise ValueError(f"{label} must contain at least one value")
    return output


def _text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("source binding requires non-empty text")
    return text


def _optional_text(value: Any) -> str:
    return str(value or "").strip()
