"""Time intent validation."""

from __future__ import annotations

from typing import Any

from .contract import IntentField, IntentKind
from .contract import Policy, Unit
from .date_values import parse_date_value

MIN_YEAR = 1
MAX_YEAR = 9999
MIN_WEEKDAY = 0
MAX_WEEKDAY = 6
MIN_MONTH = 1
MAX_MONTH = 12
MIN_QUARTER = 1
MAX_QUARTER = 4


def validate_time_intent(intent: dict[str, Any]) -> None:
    kind = str(intent.get(IntentField.KIND) or "")
    if kind == IntentKind.POINT:
        _require_exact(intent, IntentField.PRECISION, Unit.DAY)
        populated = [
            field
            for field in (IntentField.VALUE, IntentField.RELATIVE, IntentField.NAMED)
            if field in intent
        ]
        if len(populated) != 1:
            raise ValueError(
                "point intent requires exactly one of value, relative, or named."
            )
        if IntentField.VALUE in populated:
            parse_date_value(intent.get(IntentField.VALUE))
        elif IntentField.RELATIVE in populated:
            relative = _mapping(intent.get(IntentField.RELATIVE), IntentField.RELATIVE)
            _require_exact(relative, IntentField.UNIT, Unit.DAY)
            _require_int(relative, IntentField.OFFSET)
        else:
            named = _mapping(intent.get(IntentField.NAMED), IntentField.NAMED)
            _require_int_range(
                named,
                IntentField.WEEKDAY,
                minimum=MIN_WEEKDAY,
                maximum=MAX_WEEKDAY,
            )
            if IntentField.OFFSET in named:
                _require_int(named, IntentField.OFFSET)
        return
    if kind == IntentKind.RANGE:
        parse_date_value(intent.get(IntentField.START))
        parse_date_value(intent.get(IntentField.END))
        return
    if kind == IntentKind.OPEN_RANGE:
        parse_date_value(intent.get(IntentField.START))
        return
    if kind == IntentKind.PERIOD:
        unit = _require_choice(
            intent,
            IntentField.UNIT,
            {Unit.DAY, Unit.WEEK, Unit.MONTH, Unit.QUARTER, Unit.YEAR},
        )
        _require_choice(intent, IntentField.MODE, {Policy.FULL, Policy.TO_DATE})
        has_named = IntentField.NAMED in intent
        has_relative = IntentField.RELATIVE in intent
        if has_named == has_relative:
            raise ValueError("period intent requires exactly one of named or relative.")
        if has_relative:
            relative = _mapping(intent.get(IntentField.RELATIVE), IntentField.RELATIVE)
            _require_int(relative, IntentField.OFFSET)
            return
        if unit not in {Unit.MONTH, Unit.QUARTER, Unit.YEAR}:
            raise ValueError("period.named requires month, quarter, or year unit.")
        named = _mapping(intent.get(IntentField.NAMED), IntentField.NAMED)
        if unit == Unit.YEAR and IntentField.YEAR in named:
            raise ValueError("period year intent must not include named.year.")
        value = _require_int(named, IntentField.VALUE)
        _validate_named_period_value(unit=unit, value=value)
        if IntentField.YEAR in named:
            _require_int_range(
                named,
                IntentField.YEAR,
                minimum=MIN_YEAR,
                maximum=MAX_YEAR,
            )
        return
    if kind == IntentKind.WINDOW:
        _require_choice(intent, IntentField.UNIT, {Unit.DAY, Unit.WEEK, Unit.MONTH})
        count = _require_int(intent, IntentField.COUNT)
        if count < 1:
            raise ValueError("window intent requires count >= 1.")
        _require_choice(intent, IntentField.DIRECTION, {Policy.PAST, Policy.FUTURE})
        return
    raise ValueError(f"Unsupported structured time intent: {kind}")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} requires object payload.")
    return value


def _require_exact(payload: dict[str, Any], field: str, expected: str) -> None:
    actual = str(payload.get(field) or "")
    if actual != expected:
        raise ValueError(f"{field} requires {expected}.")


def _require_choice(payload: dict[str, Any], field: str, allowed: set[str]) -> str:
    actual = str(payload.get(field) or "")
    if actual not in allowed:
        raise ValueError(f"{field} has unsupported value: {actual}")
    return actual


def _require_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} requires integer value.")
    return value


def _require_int_range(
    payload: dict[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _require_int(payload, field)
    if value < minimum or value > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return value


def _validate_named_period_value(*, unit: str, value: int) -> None:
    if unit == Unit.MONTH:
        if value < MIN_MONTH or value > MAX_MONTH:
            raise ValueError("period month must be between 1 and 12.")
        return
    if unit == Unit.QUARTER:
        if value < MIN_QUARTER or value > MAX_QUARTER:
            raise ValueError("period quarter must be between 1 and 4.")
        return
    if unit == Unit.YEAR:
        if value < MIN_YEAR or value > MAX_YEAR:
            raise ValueError("period year must be between 1 and 9999.")
