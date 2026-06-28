"""Executable fact-plan shape names."""

from __future__ import annotations

SINGLE_RELATION_PLAN_SHAPES = (
    "list_rows",
    "grouped_rows",
    "direct_field_value",
    "aggregate_scalar",
    "aggregate_by_group",
    "ranked_aggregate",
)
VALUE_PLAN_SHAPES = ("computed_scalar",)
MULTI_RELATION_PLAN_SHAPES = ("set_difference", "joined_rows")
SOURCE_BOUND_PLAN_SHAPES = (
    "list_rows",
    "grouped_rows",
    "direct_field_value",
    "aggregate_scalar",
    "aggregate_by_group",
    "ranked_aggregate",
)
ALL_PLAN_SHAPES = (
    *SINGLE_RELATION_PLAN_SHAPES,
    *VALUE_PLAN_SHAPES,
    *MULTI_RELATION_PLAN_SHAPES,
)
