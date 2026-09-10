"""Strict provider schema for one normalized temporal intent."""

from __future__ import annotations

from fervis.lookup.grounding.time_intents import TIME_INTENT_FIELDS


def time_intent_schema() -> dict[str, object]:
    return {
        "oneOf": [
            _variant(
                "point_date",
                ["none", "day"],
                ["none"],
                year={"type": "integer", "minimum": 1},
                month=_month(),
                day=_day(),
            ),
            _variant(
                "point_date",
                ["none", "day"],
                ["none"],
                year={"enum": [0]},
                month=_month(),
                day=_day(),
                year_policy={"enum": ["most_recent"]},
            ),
            _variant(
                "point_relative", ["day"], ["none"], relative_offset={"type": "integer"}
            ),
            _variant(
                "period_relative",
                ["day", "week", "month", "quarter", "year"],
                ["full", "to_date"],
                relative_offset={"type": "integer"},
            ),
            _variant(
                "period_named",
                ["month"],
                ["full", "to_date"],
                year={"type": "integer", "minimum": 1},
                year_policy={"enum": ["none"]},
                named_value=_month(),
            ),
            _variant(
                "period_named",
                ["month"],
                ["full", "to_date"],
                year={"enum": [0]},
                year_policy={"enum": ["most_recent"]},
                named_value=_month(),
            ),
            _variant(
                "period_named",
                ["quarter"],
                ["full", "to_date"],
                year={"type": "integer", "minimum": 1},
                year_policy={"enum": ["none"]},
                named_value={"type": "integer", "minimum": 1, "maximum": 4},
            ),
            _variant(
                "period_named",
                ["quarter"],
                ["full", "to_date"],
                year={"enum": [0]},
                year_policy={"enum": ["most_recent"]},
                named_value={"type": "integer", "minimum": 1, "maximum": 4},
            ),
            _variant(
                "period_named",
                ["year"],
                ["full", "to_date"],
                year={"enum": [0]},
                year_policy={"enum": ["none"]},
                named_value={"type": "integer", "minimum": 1},
            ),
            _variant(
                "range",
                ["none"],
                ["none"],
                year={"type": "integer", "minimum": 0},
                month=_month(),
                day=_day(),
                year_policy={"enum": ["none", "most_recent"]},
                end_year={"type": "integer", "minimum": 0},
                end_month=_month(),
                end_day=_day(),
                end_year_policy={"enum": ["none", "most_recent"]},
            ),
            _variant(
                "open_range",
                ["none"],
                ["none"],
                year={"type": "integer", "minimum": 0},
                month=_month(),
                day=_day(),
                year_policy={"enum": ["none", "most_recent"]},
            ),
            _variant(
                "window",
                ["day", "week", "month"],
                ["none"],
                count={"type": "integer", "minimum": 1},
                direction={"enum": ["past", "future"]},
            ),
        ]
    }


def _month() -> dict[str, object]:
    return {"type": "integer", "minimum": 1, "maximum": 12}


def _day() -> dict[str, object]:
    return {"type": "integer", "minimum": 1, "maximum": 31}


def _variant(
    time_shape: str,
    units: list[str],
    modes: list[str],
    **overrides: dict[str, object],
) -> dict[str, object]:
    properties: dict[str, object] = {
        "time_shape": {"enum": [time_shape]},
        "unit": {"enum": units},
        "mode": {"enum": modes},
        "year": {"enum": [0]},
        "month": {"enum": [0]},
        "day": {"enum": [0]},
        "year_policy": {"enum": ["none"]},
        "relative_offset": {"enum": [0]},
        "named_value": {"enum": [0]},
        "end_year": {"enum": [0]},
        "end_month": {"enum": [0]},
        "end_day": {"enum": [0]},
        "end_year_policy": {"enum": ["none"]},
        "count": {"enum": [0]},
        "direction": {"enum": ["none"]},
    }
    properties.update(overrides)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(TIME_INTENT_FIELDS),
    }


__all__ = ["time_intent_schema"]
