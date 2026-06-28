"""Fact-planning schema fragments for row-list answers."""

from __future__ import annotations

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    SourceBoundPatternSchemaContext,
    field_selection_schema,
    source_bound_pattern_base,
    source_bound_pattern_required,
)
from fervis.lookup.fact_planning.schema_helpers import (
    non_empty_array_items,
    strict_object,
)


def list_rows_pattern_schema(
    context: SourceBoundPatternSchemaContext,
) -> dict[str, object]:
    field_schema = field_selection_schema(field_ids=context.field_ids)
    return strict_object(
        {
            **source_bound_pattern_base(context),
            "pattern": {"enum": ["list_rows"]},
            "output_fields": non_empty_array_items(field_schema),
        },
        required=source_bound_pattern_required(context, "output_fields"),
    )


def grouped_rows_pattern_schema(
    context: SourceBoundPatternSchemaContext,
) -> dict[str, object]:
    field_schema = field_selection_schema(field_ids=context.field_ids)
    return strict_object(
        {
            **source_bound_pattern_base(context),
            "pattern": {"enum": ["grouped_rows"]},
            "group_fields": non_empty_array_items(field_schema),
            "output_fields": non_empty_array_items(field_schema),
        },
        required=source_bound_pattern_required(
            context,
            "group_fields",
            "output_fields",
        ),
    )


SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS = {
    "list_rows": list_rows_pattern_schema,
    "grouped_rows": grouped_rows_pattern_schema,
}
