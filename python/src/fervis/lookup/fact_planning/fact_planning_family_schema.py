"""Turn-owned schema primitives for family fact-planning fragments."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.fact_planning.schema_helpers import (
    field_id_schema,
    handle_schema,
    non_empty_string_array,
    strict_object,
)


@dataclass(frozen=True)
class SourceBoundPatternSchemaContext:
    """Provider-schema context for one source-bound fact-plan pattern."""

    require_pattern: bool
    source_binding_id_schema: dict[str, object]
    source_binding_id: str | None
    field_ids: tuple[str, ...] | None
    answer_output_ids_schema: dict[str, object] | None = None
    requested_fact_id_schema: dict[str, object] | None = None
    include_source_binding_id: bool = True
    rank_limit_value_ids: tuple[str, ...] | None = None


def source_bound_pattern_base(
    context: SourceBoundPatternSchemaContext,
) -> dict[str, object]:
    base: dict[str, object] = {
        "requested_fact_id": context.requested_fact_id_schema or handle_schema(),
        "answer_output_ids": context.answer_output_ids_schema
        or non_empty_string_array(),
    }
    if context.include_source_binding_id:
        base["source_binding_id"] = context.source_binding_id_schema
    return base


def source_bound_pattern_required(
    context: SourceBoundPatternSchemaContext,
    *extra_required: str,
) -> tuple[str, ...]:
    source_required = (
        ("source_binding_id",) if context.include_source_binding_id else ()
    )
    return (
        "requested_fact_id",
        "answer_output_ids",
        "pattern",
        *source_required,
        *extra_required,
    )


def field_selection_schema(
    *,
    field_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    return strict_object(
        {
            "field_id": field_id_schema(field_ids),
        },
        required=("field_id",),
    )


def optional_pattern_schema(schema: dict[str, object]) -> dict[str, object]:
    required = schema.get("required")
    required_items = required if isinstance(required, (list, tuple)) else ()
    return {
        **schema,
        "required": [item for item in required_items if item != "pattern"],
    }


def source_bound_pattern_variant(
    context: SourceBoundPatternSchemaContext,
    schema: dict[str, object],
) -> dict[str, object]:
    if context.require_pattern:
        return schema
    return optional_pattern_schema(schema)
