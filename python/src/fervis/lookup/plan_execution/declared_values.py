"""Single declared-type algebra for runtime values, comparisons, and keys."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Mapping
from uuid import UUID

from fervis.lookup.canonical_data import (
    RuntimeValue,
    canonical_runtime_json,
    parse_runtime_value,
)
from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.types.enums import StrEnum


MAX_RANK_LIMIT = 9_223_372_036_854_775_807


class DeclaredValueKind(StrEnum):
    RUNTIME = "runtime"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    DECIMAL = "decimal"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    UUID = "uuid"
    COLLECTION = "collection"


_DECLARED_KINDS = {
    "boolean": DeclaredValueKind.BOOLEAN,
    "bool": DeclaredValueKind.BOOLEAN,
    "integer": DeclaredValueKind.INTEGER,
    "int": DeclaredValueKind.INTEGER,
    "number": DeclaredValueKind.DECIMAL,
    "numeric": DeclaredValueKind.DECIMAL,
    "decimal": DeclaredValueKind.DECIMAL,
    "float": DeclaredValueKind.DECIMAL,
    "double": DeclaredValueKind.DECIMAL,
    "string": DeclaredValueKind.STRING,
    "choice": DeclaredValueKind.STRING,
    "path": DeclaredValueKind.STRING,
    "pk": DeclaredValueKind.STRING,
    "duration": DeclaredValueKind.STRING,
    "date": DeclaredValueKind.DATE,
    "datetime": DeclaredValueKind.DATETIME,
    "time": DeclaredValueKind.TIME,
    "uuid": DeclaredValueKind.UUID,
    "array": DeclaredValueKind.COLLECTION,
    "list": DeclaredValueKind.COLLECTION,
    "object": DeclaredValueKind.COLLECTION,
    "json": DeclaredValueKind.COLLECTION,
    "any": DeclaredValueKind.RUNTIME,
    "unknown": DeclaredValueKind.RUNTIME,
    "": DeclaredValueKind.RUNTIME,
}


@dataclass(frozen=True)
class DeclaredValue:
    kind: DeclaredValueKind
    value: RuntimeValue

    @property
    def equality_key(self) -> tuple[str, str]:
        return self.kind.value, canonical_runtime_json(self.value)

    @property
    def ordering_key(self) -> tuple[int, Decimal | str]:
        if self.value is None:
            return 0, ""
        if self.kind in {DeclaredValueKind.INTEGER, DeclaredValueKind.DECIMAL}:
            return 1, Decimal(str(self.value))
        if self.kind in {
            DeclaredValueKind.STRING,
            DeclaredValueKind.DATE,
            DeclaredValueKind.DATETIME,
            DeclaredValueKind.TIME,
            DeclaredValueKind.UUID,
        }:
            return 2, str(self.value)
        if self.kind is DeclaredValueKind.RUNTIME:
            if isinstance(self.value, bool):
                return 3, "true" if self.value else "false"
            if isinstance(self.value, (int, float, Decimal)):
                return 1, Decimal(str(self.value))
            if isinstance(self.value, (str, date, datetime, time, UUID)):
                return 2, str(self.value)
        raise RelationEngineError(f"{self.kind.value} value is not orderable")


def declared_kind(type_name: str | None) -> DeclaredValueKind:
    normalized = str(type_name or "").strip().casefold()
    try:
        return _DECLARED_KINDS[normalized]
    except KeyError as exc:
        raise RelationEngineError(
            f"unsupported declared value type {normalized}"
        ) from exc


def declared_types_compatible(left: str | None, right: str | None) -> bool:
    left_kind = declared_kind(left)
    right_kind = declared_kind(right)
    return (
        left_kind is DeclaredValueKind.RUNTIME
        or right_kind is DeclaredValueKind.RUNTIME
        or left_kind is right_kind
    )


def parse_declared_value(value: object, type_name: str | None) -> RuntimeValue:
    kind = declared_kind(type_name)
    if value is None:
        return None
    if kind is DeclaredValueKind.RUNTIME:
        return parse_runtime_value(value)
    if kind is DeclaredValueKind.BOOLEAN:
        if type(value) is bool:
            return value
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            return value.strip().casefold() == "true"
        raise RelationEngineError("declared boolean value must be boolean")
    if kind is DeclaredValueKind.INTEGER:
        return _integer(value)
    if kind is DeclaredValueKind.DECIMAL:
        return _decimal(value)
    if kind is DeclaredValueKind.STRING:
        if not isinstance(value, str):
            raise RelationEngineError("declared string value must be string")
        return value
    if kind is DeclaredValueKind.DATE:
        if isinstance(value, datetime):
            raise RelationEngineError("declared date value must not be datetime")
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:
                raise RelationEngineError("declared date value is invalid") from exc
        raise RelationEngineError("declared date value must be ISO date")
    if kind is DeclaredValueKind.DATETIME:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(_iso_utc_offset(value))
            except ValueError as exc:
                raise RelationEngineError("declared datetime value is invalid") from exc
        raise RelationEngineError("declared datetime value must be ISO datetime")
    if kind is DeclaredValueKind.TIME:
        if isinstance(value, time):
            return value
        if isinstance(value, str):
            try:
                return time.fromisoformat(_iso_utc_offset(value))
            except ValueError as exc:
                raise RelationEngineError("declared time value is invalid") from exc
        raise RelationEngineError("declared time value must be ISO time")
    if kind is DeclaredValueKind.UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            # Host catalogs use UUID as an identity domain even when test or legacy
            # adapters expose opaque identifiers that are not RFC-4122 spellings.
            try:
                return UUID(value)
            except ValueError:
                return value
        raise RelationEngineError("declared UUID value must be UUID text")
    if kind is DeclaredValueKind.COLLECTION:
        parsed = parse_runtime_value(value)
        if not isinstance(parsed, (tuple, list, Mapping)):
            raise RelationEngineError("declared collection value must be a collection")
        return parsed
    raise AssertionError("unreachable declared value kind")


def _iso_utc_offset(value: str) -> str:
    if value.endswith("Z"):
        return f"{value[:-1]}+00:00"
    return value


def declared_value(value: object, type_name: str | None) -> DeclaredValue:
    kind = declared_kind(type_name)
    parsed = parse_declared_value(value, type_name)
    if kind is DeclaredValueKind.RUNTIME:
        kind = _runtime_kind(parsed)
    return DeclaredValue(kind=kind, value=parsed)


def declared_equal(
    left: object,
    left_type: str | None,
    right: object,
    right_type: str | None,
) -> bool:
    if not declared_types_compatible(left_type, right_type):
        raise RelationEngineError(
            "comparison operands have incompatible declared types"
        )
    comparison_type = _comparison_type(left_type, right_type)
    return (
        declared_value(left, comparison_type).equality_key
        == declared_value(right, comparison_type).equality_key
    )


def declared_key(value: object, type_name: str | None) -> tuple[str, str]:
    return declared_value(value, type_name).equality_key


def declared_order_key(
    value: object,
    type_name: str | None,
) -> tuple[int, Decimal | str]:
    return declared_value(value, type_name).ordering_key


def declared_order_pair(
    left: object,
    left_type: str | None,
    right: object,
    right_type: str | None,
) -> tuple[tuple[int, Decimal | str], tuple[int, Decimal | str]]:
    if not declared_types_compatible(left_type, right_type):
        raise RelationEngineError(
            "comparison operands have incompatible declared types"
        )
    comparison_type = _comparison_type(left_type, right_type)
    return (
        declared_order_key(left, comparison_type),
        declared_order_key(right, comparison_type),
    )


def declared_number(value: object, type_name: str | None) -> Decimal:
    """Return an exact numeric value only when its declared type is numeric."""
    kind = declared_kind(type_name)
    if kind not in {
        DeclaredValueKind.RUNTIME,
        DeclaredValueKind.INTEGER,
        DeclaredValueKind.DECIMAL,
    }:
        raise RelationEngineError("aggregation requires a declared numeric field")
    parsed = declared_value(value, type_name)
    if parsed.kind not in {DeclaredValueKind.INTEGER, DeclaredValueKind.DECIMAL}:
        raise RelationEngineError("aggregation requires a numeric value")
    return Decimal(str(parsed.value))


def exact_positive_integer(value: object, *, maximum: int = MAX_RANK_LIMIT) -> int:
    numeric = _decimal(value)
    integral = numeric.to_integral_value()
    if numeric != integral or integral < 1 or integral > maximum:
        raise ValueError("rank requires positive integer limit within supported range")
    return int(integral)


def _comparison_type(left: str | None, right: str | None) -> str | None:
    if declared_kind(left) is not DeclaredValueKind.RUNTIME:
        return left
    if declared_kind(right) is not DeclaredValueKind.RUNTIME:
        return right
    return None


def _runtime_kind(value: RuntimeValue) -> DeclaredValueKind:
    if value is None:
        return DeclaredValueKind.RUNTIME
    if type(value) is bool:
        return DeclaredValueKind.BOOLEAN
    if type(value) is int:
        return DeclaredValueKind.INTEGER
    if isinstance(value, (float, Decimal)):
        return DeclaredValueKind.DECIMAL
    if isinstance(value, str):
        return DeclaredValueKind.STRING
    if isinstance(value, datetime):
        return DeclaredValueKind.DATETIME
    if isinstance(value, date):
        return DeclaredValueKind.DATE
    if isinstance(value, time):
        return DeclaredValueKind.TIME
    if isinstance(value, UUID):
        return DeclaredValueKind.UUID
    if isinstance(value, (tuple, list, Mapping)):
        return DeclaredValueKind.COLLECTION
    raise RelationEngineError("unsupported runtime comparison value")


def _integer(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text.startswith("-") and text[1:].isdigit())):
            return int(text)
    raise RelationEngineError("declared integer value must be exact integer")


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise RelationEngineError("declared numeric value must be numeric")
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RelationEngineError("declared numeric value is invalid") from exc
    if not numeric.is_finite():
        raise RelationEngineError("declared numeric value must be finite")
    return numeric
