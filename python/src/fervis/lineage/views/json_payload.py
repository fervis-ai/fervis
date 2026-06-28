"""JSON payload helpers for lineage view dataclasses."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import StrEnum
from typing import Any


def view_json(value: Any) -> Any:
    if is_dataclass(value):
        return view_json(asdict(value))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {_camel_key(str(key)): view_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [view_json(item) for item in value]
    return value


def _camel_key(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])
