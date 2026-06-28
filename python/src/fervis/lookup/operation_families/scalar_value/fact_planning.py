"""Fact-planning schema fragments for direct scalar-value answers."""

from __future__ import annotations

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    SourceBoundPatternSchemaContext,
    field_selection_schema,
    source_bound_pattern_base,
    source_bound_pattern_required,
)
from fervis.lookup.fact_planning.schema_helpers import strict_object


def direct_field_value_pattern_schema(
    context: SourceBoundPatternSchemaContext,
) -> dict[str, object]:
    return strict_object(
        {
            **source_bound_pattern_base(context),
            "pattern": {"enum": ["direct_field_value"]},
            "output_field": field_selection_schema(field_ids=context.field_ids),
        },
        required=source_bound_pattern_required(context, "output_field"),
    )


SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS = {
    "direct_field_value": direct_field_value_pattern_schema,
}
