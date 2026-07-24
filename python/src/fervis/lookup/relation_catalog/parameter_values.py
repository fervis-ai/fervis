"""Typed parsing for endpoint parameter values declared by a relation catalog."""

from __future__ import annotations

import json
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import TypeAlias
from uuid import UUID


CatalogParameterValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["CatalogParameterValue", ...]
    | dict[str, "CatalogParameterValue"]
)
CatalogScalarParameterValue: TypeAlias = bool | int | float | str


class CatalogParameterValueError(ValueError):
    pass


def parse_catalog_parameter_value(
    value: object,
    *,
    type_name: str,
    choices: tuple[str, ...] = (),
) -> CatalogParameterValue:
    """Parse one raw catalog value according to its declared endpoint type."""

    if value is None:
        return None
    normalized_type = type_name.strip().casefold()
    if normalized_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value in {"true", "false"}:
            return value == "true"
        raise CatalogParameterValueError("boolean value must be true or false")
    if normalized_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise CatalogParameterValueError("integer value must be an integer")
        return value
    if normalized_type in {"number", "double", "float"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            isinstance(value, float) and not isfinite(value)
        ):
            raise CatalogParameterValueError("numeric value must be finite")
        return value
    if normalized_type in {"array", "list"}:
        if not isinstance(value, (list, tuple)):
            raise CatalogParameterValueError("sequence value must be an array")
        return tuple(_parse_json_value(item) for item in value)
    if normalized_type in {"json", "object"}:
        if not isinstance(value, dict):
            raise CatalogParameterValueError("object value must be an object")
        return _parse_json_object(value)
    if normalized_type in {"date", "datetime", "decimal", "time"}:
        if not isinstance(value, str):
            raise CatalogParameterValueError("text value must be a string")
        _validate_text_scalar(value, type_name=normalized_type)
        return value
    if normalized_type in {"choice", "duration", "path", "pk", "string"}:
        if not isinstance(value, str):
            raise CatalogParameterValueError("text value must be a string")
        if choices and value not in choices:
            raise CatalogParameterValueError("value is not a declared choice")
        return value
    if normalized_type == "uuid":
        if not isinstance(value, str):
            raise CatalogParameterValueError("UUID value must be text")
        try:
            return str(UUID(value))
        except ValueError as exc:
            raise CatalogParameterValueError("UUID value is invalid") from exc
    if normalized_type in {"any", "unknown", ""}:
        return _parse_json_value(value)
    raise CatalogParameterValueError(f"unsupported catalog value type {type_name}")


def _validate_text_scalar(value: str, *, type_name: str) -> None:
    try:
        if type_name == "date":
            date.fromisoformat(value)
        elif type_name == "datetime":
            datetime.fromisoformat(value)
        elif type_name == "time":
            time.fromisoformat(value)
        else:
            number = Decimal(value)
            if not number.is_finite():
                raise CatalogParameterValueError("decimal value must be finite")
    except (ValueError, InvalidOperation) as exc:
        raise CatalogParameterValueError(
            f"{type_name} value has invalid syntax"
        ) from exc


def parse_catalog_parameter_text(
    text: str,
    *,
    type_name: str,
    choices: tuple[str, ...] = (),
) -> CatalogScalarParameterValue:
    """Compile supplied text into one declared scalar parameter value."""

    candidates: list[CatalogScalarParameterValue] = [text]
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, (bool, int, float, str)) and decoded != text:
        candidates.append(decoded)
    for candidate in candidates:
        try:
            parsed = parse_catalog_parameter_value(
                candidate,
                type_name=type_name,
                choices=choices,
            )
        except CatalogParameterValueError:
            continue
        if isinstance(parsed, (bool, int, float, str)):
            return parsed
    raise CatalogParameterValueError("text is incompatible with the scalar parameter")


def _parse_json_value(value: object) -> CatalogParameterValue:
    if isinstance(value, float) and not isfinite(value):
        raise CatalogParameterValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_parse_json_value(item) for item in value)
    if isinstance(value, dict):
        return _parse_json_object(value)
    raise CatalogParameterValueError("value must be JSON-compatible")


def _parse_json_object(value: dict[object, object]) -> dict[str, CatalogParameterValue]:
    if any(type(key) is not str for key in value):
        raise CatalogParameterValueError("object keys must be strings")
    return {str(key): _parse_json_value(item) for key, item in value.items()}
