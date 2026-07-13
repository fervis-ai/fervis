"""Closed canonical serialization for runtime data and evidence identity."""

from __future__ import annotations

from datetime import date, datetime, time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import json
import math
from typing import Any, Mapping, TypeAlias
from uuid import UUID


RuntimeScalar: TypeAlias = (
    None | bool | int | float | str | Decimal | date | datetime | time | UUID | Enum
)
RuntimeValue: TypeAlias = (
    RuntimeScalar
    | tuple["RuntimeValue", ...]
    | list["RuntimeValue"]
    | Mapping[str, "RuntimeValue"]
)


@dataclass(frozen=True)
class EntityKeyComponentValue:
    component_id: str
    value: RuntimeValue

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("entity key component requires component id")
        if self.value is None:
            raise ValueError("entity key component requires a non-null value")


@dataclass(frozen=True)
class EntityKeyValue:
    entity_kind: str
    key_id: str
    components: tuple[EntityKeyComponentValue, ...]

    def __post_init__(self) -> None:
        if not self.entity_kind or not self.key_id or not self.components:
            raise ValueError("entity key value is incomplete")
        component_ids = tuple(component.component_id for component in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("entity key value repeats a component")

    def component_values(self) -> dict[str, RuntimeValue]:
        return {
            component.component_id: component.value for component in self.components
        }


ResultValue: TypeAlias = RuntimeValue | EntityKeyValue


def parse_runtime_value(value: object) -> RuntimeValue:
    """Close one external value over the deterministic runtime algebra."""

    if value is None or isinstance(
        value,
        (bool, int, str, Decimal, date, datetime, time, UUID, Enum),
    ):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("runtime numbers must be finite")
        return value
    if isinstance(value, tuple):
        return tuple(parse_runtime_value(item) for item in value)
    if isinstance(value, list):
        return [parse_runtime_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("runtime object keys must be strings")
        return {
            key: parse_runtime_value(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    raise TypeError(f"unsupported runtime value {type(value).__name__}")


def canonical_runtime_json(value: object) -> str:
    return json.dumps(
        runtime_value_to_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def runtime_value_to_payload(value: object) -> Any:
    """Encode a closed runtime value into its canonical JSON transport shape."""

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
            "value": runtime_value_to_payload(value.value),
        }
    if type(value) is tuple:
        return {"$tuple": [runtime_value_to_payload(item) for item in value]}
    if type(value) is list:
        return [runtime_value_to_payload(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("runtime object keys must be strings")
        return {
            str(key): runtime_value_to_payload(item) for key, item in value.items()
        }
    raise TypeError(f"unsupported runtime value {type(value).__name__}")


def runtime_value_from_payload(value: object) -> RuntimeValue:
    """Decode a canonical runtime transport payload at a persistence boundary."""

    if isinstance(value, list):
        return [runtime_value_from_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return parse_runtime_value(value)
    if set(value) == {"$decimal"}:
        return Decimal(str(value["$decimal"]))
    if set(value) == {"$datetime"}:
        return datetime.fromisoformat(str(value["$datetime"]))
    if set(value) == {"$date"}:
        return date.fromisoformat(str(value["$date"]))
    if set(value) == {"$time"}:
        return time.fromisoformat(str(value["$time"]))
    if set(value) == {"$uuid"}:
        return UUID(str(value["$uuid"]))
    if set(value) == {"$tuple"}:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise TypeError("runtime tuple payload must contain an array")
        return tuple(runtime_value_from_payload(item) for item in items)
    if set(value) == {"$enum", "value"}:
        return runtime_value_from_payload(value["value"])
    return {
        str(key): runtime_value_from_payload(item) for key, item in value.items()
    }
