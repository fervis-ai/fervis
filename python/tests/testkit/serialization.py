from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any


def portable_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if is_dataclass(value) and not isinstance(value, type):
        return portable_value(asdict(value))
    if isinstance(value, tuple):
        return [portable_value(item) for item in value]
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): portable_value(item) for key, item in value.items()}
    return value
