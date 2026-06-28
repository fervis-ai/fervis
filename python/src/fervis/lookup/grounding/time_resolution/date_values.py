"""Structured date values for time intents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from .contract import IntentField, Policy


@dataclass(frozen=True)
class StructuredDateValue:
    month: int
    day: int
    year: int | None = None
    year_policy: str = ""


def parse_date_value(raw: Any) -> StructuredDateValue:
    if not isinstance(raw, Mapping):
        raise ValueError("date value requires a structured date object.")
    month = _int_field(raw, IntentField.MONTH)
    day = _int_field(raw, IntentField.DAY)
    if month < 1 or month > 12:
        raise ValueError("date value requires month between 1 and 12.")
    year = raw.get(IntentField.YEAR)
    year_policy = str(raw.get(IntentField.YEAR_POLICY) or "")
    if year is None and not year_policy:
        raise ValueError("date value requires year or year_policy.")
    if year is not None and year_policy:
        raise ValueError("date value requires either year or year_policy, not both.")
    if year_policy and year_policy != Policy.MOST_RECENT:
        raise ValueError(f"Unsupported date year policy: {year_policy}")
    return StructuredDateValue(
        month=month,
        day=day,
        year=_strict_int(year, IntentField.YEAR) if year is not None else None,
        year_policy=year_policy,
    )


def resolve_date_value(anchor: date, raw: Any) -> date:
    value = parse_date_value(raw)
    if value.year is not None:
        return date(value.year, value.month, value.day)
    candidate = date(anchor.year, value.month, value.day)
    if candidate > anchor:
        candidate = date(anchor.year - 1, value.month, value.day)
    return candidate


def normalize_date_value(raw: Any, value: date) -> dict[str, int]:
    parse_date_value(raw)
    return {
        IntentField.MONTH: value.month,
        IntentField.DAY: value.day,
        IntentField.YEAR: value.year,
    }


def _int_field(raw: Mapping[str, Any], field: str) -> int:
    try:
        return _strict_int(raw[field], field)
    except KeyError as exc:
        raise ValueError(f"date value requires {field}.") from exc


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"date value requires integer {field}.")
    return value
