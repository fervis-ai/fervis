"""Shared JSON schema validation primitives for Fervis config versions."""

from __future__ import annotations

from collections.abc import Mapping


def reject_unknown_keys(
    payload: Mapping[str, object],
    *,
    allowed: set[str],
    prefix: str = "",
) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        dotted = ", ".join(f"{prefix}.{key}" if prefix else key for key in unknown)
        raise ValueError(f"unsupported keys: {dotted}")


def require_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object.")
    return value


def require_list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list.")
    return value


def require_string(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_blank: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    if not allow_blank and not value.strip():
        raise ValueError(f"{key} must not be blank.")
    return value


def require_string_list(payload: Mapping[str, object], key: str) -> list[str]:
    values = require_list(payload, key)
    if not values:
        raise ValueError(f"{key} must contain at least one value.")
    strings: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must contain only non-empty strings.")
        strings.append(value)
    return strings
