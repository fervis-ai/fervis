"""Closed canonical serialization for runtime data and evidence identity."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
import json
import math
from typing import Any
from uuid import UUID


def canonical_runtime_json(value: object) -> str:
    return json.dumps(
        _canonical_runtime_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonical_runtime_value(value: object) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("runtime numbers must be finite")
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError("runtime decimals must be finite")
        return {"$decimal": format(value, "f")}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, time):
        return {"$time": value.isoformat()}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, Enum):
        return {
            "$enum": type(value).__name__,
            "value": _canonical_runtime_value(value.value),
        }
    if type(value) is tuple:
        return {"$tuple": [_canonical_runtime_value(item) for item in value]}
    if type(value) is list:
        return [_canonical_runtime_value(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("runtime object keys must be strings")
        return {
            str(key): _canonical_runtime_value(item)
            for key, item in value.items()
        }
    raise TypeError(f"unsupported runtime value {type(value).__name__}")
