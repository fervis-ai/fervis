"""Fact-planning schema fragments for row-list answers."""

from __future__ import annotations

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    SourceBoundPatternSchemaContext,
    field_selection_schema,
    source_bound_pattern_base,
    source_bound_pattern_required,
)
from fervis.lookup.fact_planning.schema_helpers import (
    field_id_schema,
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


def ranked_rows_pattern_schema(
    context: SourceBoundPatternSchemaContext,
) -> dict[str, object]:
    field_schema = field_selection_schema(field_ids=context.field_ids)
    limit_value_id_schema: dict[str, object] = {"type": "string", "maxLength": 0}
    if context.rank_limit_value_ids:
        limit_value_id_schema = {
            "type": "string",
            "enum": list(context.rank_limit_value_ids),
        }
    rank_schema = strict_object(
        {
            "selection_basis": {"type": "string", "minLength": 1},
            "sort": {"enum": ["asc", "desc"]},
            "limit_value_id": limit_value_id_schema,
        },
        required=("selection_basis", "sort"),
    )
    order_field_schema = strict_object(
        {
            "selection_basis": {"type": "string", "minLength": 1},
            "field_id": field_id_schema(context.field_ids),
        },
        required=("selection_basis", "field_id"),
    )
    return strict_object(
        {
            **source_bound_pattern_base(context),
            "pattern": {"enum": ["ranked_rows"]},
            "output_fields": non_empty_array_items(field_schema),
            "order_field": order_field_schema,
            "rank": rank_schema,
        },
        required=source_bound_pattern_required(
            context,
            "output_fields",
            "order_field",
            "rank",
        ),
    )


SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS = {
    "list_rows": list_rows_pattern_schema,
    "ranked_rows": ranked_rows_pattern_schema,
    "grouped_rows": grouped_rows_pattern_schema,
}
