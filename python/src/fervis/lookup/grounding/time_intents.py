"""Model-authored time intent normalization for grounding."""

from __future__ import annotations

from typing import Any

from fervis.lookup.grounding.time_resolution import validate_time_intent
from fervis.lookup.grounding.provider_contract import FlatTimeIntentOutput

TIME_INTENT_FIELDS = (
    "time_shape",
    "unit",
    "mode",
    "year",
    "month",
    "day",
    "year_policy",
    "relative_offset",
    "named_value",
    "end_year",
    "end_month",
    "end_day",
    "end_year_policy",
    "count",
    "direction",
)


def normalize_grounding_date_intent(
    expression: str,
    intent: FlatTimeIntentOutput,
    *,
    path: str,
) -> dict[str, object]:
    _required_text(expression, path=f"{path}.expression")
    intent_fields = {
        "year": intent.year,
        "month": intent.month,
        "day": intent.day,
        "year_policy": intent.year_policy,
        "relative_offset": intent.relative_offset,
        "named_value": intent.named_value,
        "end_year": intent.end_year,
        "end_month": intent.end_month,
        "end_day": intent.end_day,
        "end_year_policy": intent.end_year_policy,
        "count": intent.count,
        "direction": intent.direction,
        "time_shape": intent.time_shape,
        "unit": intent.unit,
        "mode": intent.mode,
    }
    normalized = _canonical_time_intent_from_flat_fields(intent_fields)
    validate_time_intent(normalized)
    return normalized


def _canonical_time_intent_from_flat_fields(intent: dict[str, Any]) -> dict[str, Any]:
    _reject_unexpected_keys(intent, set(TIME_INTENT_FIELDS), "date_intent.intent")
    time_shape = _required_text(
        intent.get("time_shape"),
        path="date_intent.intent.time_shape",
    )
    unit = _required_text(intent.get("unit"), path="date_intent.intent.unit")
    mode = _required_text(intent.get("mode"), path="date_intent.intent.mode")
    direction = _required_text(
        intent.get("direction"),
        path="date_intent.intent.direction",
    )
    if time_shape == "point_date":
        if unit not in {"none", "day"}:
            raise ValueError("point_date requires none or day unit")
        _require_exact_flat(mode, "none", "mode")
        _require_time_shape_neutral_except(intent, used=("date",))
        return {
            "kind": "point",
            "precision": "day",
            "value": _flat_shared_date_value(intent),
        }
    if time_shape == "point_relative":
        _require_exact_flat(unit, "day", "unit")
        _require_exact_flat(mode, "none", "mode")
        _require_time_shape_neutral_except(intent, used=("relative",))
        return {
            "kind": "point",
            "precision": "day",
            "relative": {
                "unit": "day",
                "offset": _flat_int(intent, "relative_offset"),
            },
        }
    if time_shape == "period_relative":
        if unit not in {"day", "week", "month", "quarter", "year"}:
            raise ValueError("period_relative requires period unit")
        if mode not in {"full", "to_date"}:
            raise ValueError("period_relative requires full or to_date mode")
        _require_time_shape_neutral_except(intent, used=("relative",))
        return {
            "kind": "period",
            "unit": unit,
            "mode": mode,
            "relative": {
                "unit": unit,
                "offset": _flat_int(intent, "relative_offset"),
            },
        }
    if time_shape == "period_named":
        if unit not in {"month", "quarter", "year"}:
            raise ValueError("period_named requires month, quarter, or year unit")
        if mode not in {"full", "to_date"}:
            raise ValueError("period_named requires full or to_date mode")
        _require_time_shape_neutral_except(intent, used=("named",))
        return {
            "kind": "period",
            "unit": unit,
            "mode": mode,
            "named": _flat_shared_named_period(intent, unit=unit),
        }
    if time_shape == "range":
        _require_exact_flat(unit, "none", "unit")
        _require_exact_flat(mode, "none", "mode")
        _require_time_shape_neutral_except(intent, used=("date", "end"))
        return {
            "kind": "range",
            "start": _flat_shared_date_value(intent),
            "end": _flat_end_date_value(intent),
        }
    if time_shape == "open_range":
        _require_exact_flat(unit, "none", "unit")
        _require_exact_flat(mode, "none", "mode")
        _require_time_shape_neutral_except(intent, used=("date",))
        return {
            "kind": "open_range",
            "start": _flat_shared_date_value(intent),
        }
    if time_shape == "window":
        if unit not in {"day", "week", "month"}:
            raise ValueError("window requires day, week, or month unit")
        if direction not in {"past", "future"}:
            raise ValueError("window requires past or future direction")
        _require_exact_flat(mode, "none", "mode")
        _require_time_shape_neutral_except(intent, used=("count",))
        return {
            "kind": "window",
            "unit": unit,
            "count": _flat_int(intent, "count"),
            "direction": direction,
        }
    raise ValueError("unsupported time_shape")


def _flat_shared_date_value(intent: dict[str, Any]) -> dict[str, object]:
    year = _flat_int(intent, "year")
    month = _flat_int(intent, "month")
    day = _flat_int(intent, "day")
    year_policy = _required_text(
        intent.get("year_policy"),
        path="date_intent.intent.year_policy",
    )
    if month < 1 or day < 1:
        raise ValueError("date requires month and day")
    if year_policy == "most_recent":
        if year != 0:
            raise ValueError("date uses either year or year_policy")
        return {"month": month, "day": day, "year_policy": "most_recent"}
    _require_exact_flat(year_policy, "none", "year_policy")
    if year < 1:
        raise ValueError("date requires year or year_policy")
    return {"year": year, "month": month, "day": day}


def _flat_end_date_value(intent: dict[str, Any]) -> dict[str, object]:
    year = _flat_int(intent, "end_year")
    month = _flat_int(intent, "end_month")
    day = _flat_int(intent, "end_day")
    year_policy = _required_text(
        intent.get("end_year_policy"),
        path="date_intent.intent.end_year_policy",
    )
    if month < 1 or day < 1:
        raise ValueError("end date requires month and day")
    if year_policy == "most_recent":
        if year != 0:
            raise ValueError("end date uses either year or year_policy")
        return {"month": month, "day": day, "year_policy": "most_recent"}
    _require_exact_flat(year_policy, "none", "end_year_policy")
    if year < 1:
        raise ValueError("end date requires year or year_policy")
    return {"year": year, "month": month, "day": day}


def _flat_shared_named_period(
    intent: dict[str, Any],
    *,
    unit: str,
) -> dict[str, object]:
    value = _flat_int(intent, "named_value")
    year = _flat_int(intent, "year")
    year_policy = _required_text(
        intent.get("year_policy"),
        path="date_intent.intent.year_policy",
    )
    if value < 1:
        raise ValueError("named period requires named_value")
    output: dict[str, object] = {"value": value}
    if unit in {"month", "quarter"}:
        if year_policy == "most_recent":
            if year != 0:
                raise ValueError("named period uses either year or year_policy")
            output["year_policy"] = "most_recent"
            return output
        _require_exact_flat(year_policy, "none", "year_policy")
        if year:
            output["year"] = year
        return output
    _require_exact_flat(year_policy, "none", "year_policy")
    if year:
        raise ValueError("year period must not include year")
    return output


def _require_time_shape_neutral_except(
    intent: dict[str, Any],
    *,
    used: tuple[str, ...],
) -> None:
    used_fields = _time_shape_field_groups(used)
    neutral_values: dict[str, object] = {
        "year": 0,
        "month": 0,
        "day": 0,
        "year_policy": "none",
        "relative_offset": 0,
        "named_value": 0,
        "end_year": 0,
        "end_month": 0,
        "end_day": 0,
        "end_year_policy": "none",
        "count": 0,
        "direction": "none",
    }
    for field, neutral in neutral_values.items():
        if field in used_fields:
            continue
        if intent.get(field) != neutral:
            raise ValueError(f"unused time field must be neutral: {field}")


def _time_shape_field_groups(groups: tuple[str, ...]) -> set[str]:
    fields: set[str] = set()
    for group in groups:
        if group == "date":
            fields.update({"year", "month", "day", "year_policy"})
        elif group == "relative":
            fields.add("relative_offset")
        elif group == "named":
            fields.update({"named_value", "year", "year_policy"})
        elif group == "end":
            fields.update({"end_year", "end_month", "end_day", "end_year_policy"})
        elif group == "count":
            fields.update({"count", "direction"})
    return fields


def _flat_int(intent: dict[str, Any], field: str) -> int:
    value = intent.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"date_intent.intent.{field} must be an integer")
    return value


def _require_exact_flat(value: str, expected: str, field: str) -> None:
    if value != expected:
        raise ValueError(f"date_intent.intent.{field} requires {expected}")


def _required_text(value: object, *, path: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{path} requires non-empty text")
    return text


def _reject_unexpected_keys(
    raw: dict[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(f"{path} has unexpected keys: {', '.join(unexpected)}")
