"""Fact-planning schema fragments for joined-row answers."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    field_selection_schema,
    optional_pattern_schema,
)
from fervis.lookup.fact_planning.schema_helpers import (
    field_id_schema,
    handle_schema,
    non_empty_array_items,
    non_empty_string_array,
    strict_object,
)


JOINED_ROWS_PATTERN_NAMES = frozenset({"joined_rows"})


def joined_rows_generic_pattern_answer_variants(
    *,
    requested_fact_id_schema: dict[str, object] | None,
    require_pattern: bool,
) -> list[dict[str, object]]:
    schema = _joined_rows_generic_pattern_schema(
        requested_fact_id_schema=requested_fact_id_schema,
    )
    return [schema if require_pattern else optional_pattern_schema(schema)]


def joined_rows_pattern_answer_variants(
    *,
    requested_fact_id_schema: dict[str, object],
    answer_output_ids_schema: dict[str, object] | None,
    source_binding_ids: tuple[str, ...],
    source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    require_pattern: bool,
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    left_source_ids = source_binding_ids_by_requirement.get(
        "left",
        source_binding_ids,
    )
    right_source_ids = source_binding_ids_by_requirement.get(
        "right",
        source_binding_ids,
    )
    for left_source_id in left_source_ids:
        left_field_ids = field_ids_by_source_binding_id.get(left_source_id, ())
        if not left_field_ids:
            continue
        for right_source_id in right_source_ids:
            right_field_ids = field_ids_by_source_binding_id.get(right_source_id, ())
            if not right_field_ids:
                continue
            schema = _joined_rows_pattern_schema(
                requested_fact_id_schema=requested_fact_id_schema,
                answer_output_ids_schema=answer_output_ids_schema,
                left_source_id=left_source_id,
                left_field_ids=left_field_ids,
                right_source_id=right_source_id,
                right_field_ids=right_field_ids,
            )
            variants.append(
                schema if require_pattern else optional_pattern_schema(schema)
            )
    return variants


def _joined_rows_generic_pattern_schema(
    *,
    requested_fact_id_schema: dict[str, object] | None,
) -> dict[str, object]:
    return strict_object(
        {
            "requested_fact_id": requested_fact_id_schema or handle_schema(),
            "answer_output_ids": non_empty_string_array(),
            "pattern": {"enum": ["joined_rows"]},
            "left": _join_operand_schema(),
            "right": _join_operand_schema(),
            "join_keys": non_empty_array_items(_join_key_schema()),
            "output_fields": non_empty_array_items(_joined_field_schema()),
        },
        required=(
            "requested_fact_id",
            "answer_output_ids",
            "pattern",
            "left",
            "right",
            "join_keys",
            "output_fields",
        ),
    )


def _joined_rows_pattern_schema(
    *,
    requested_fact_id_schema: dict[str, object],
    answer_output_ids_schema: dict[str, object] | None,
    left_source_id: str,
    left_field_ids: tuple[str, ...],
    right_source_id: str,
    right_field_ids: tuple[str, ...],
) -> dict[str, object]:
    return strict_object(
        {
            "requested_fact_id": requested_fact_id_schema,
            "answer_output_ids": answer_output_ids_schema or non_empty_string_array(),
            "pattern": {"enum": ["joined_rows"]},
            "left": _join_operand_schema_for_source(
                source_binding_id=left_source_id,
                field_ids=left_field_ids,
            ),
            "right": _join_operand_schema_for_source(
                source_binding_id=right_source_id,
                field_ids=right_field_ids,
            ),
            "join_keys": non_empty_array_items(
                _join_key_schema_for_sources(
                    left_field_ids=left_field_ids,
                    right_field_ids=right_field_ids,
                )
            ),
            "output_fields": non_empty_array_items(
                _joined_field_schema_for_sources(
                    left_field_ids=left_field_ids,
                    right_field_ids=right_field_ids,
                )
            ),
        },
        required=(
            "requested_fact_id",
            "answer_output_ids",
            "pattern",
            "left",
            "right",
            "join_keys",
            "output_fields",
        ),
    )


def _join_operand_schema() -> dict[str, object]:
    return strict_object(
        {
            "source_binding_id": handle_schema(),
            "fields": non_empty_array_items(field_selection_schema()),
        },
        required=("source_binding_id", "fields"),
    )


def _join_operand_schema_for_source(
    *,
    source_binding_id: str,
    field_ids: tuple[str, ...],
) -> dict[str, object]:
    return strict_object(
        {
            "source_binding_id": {"enum": [source_binding_id]},
            "fields": non_empty_array_items(
                field_selection_schema(field_ids=field_ids)
            ),
        },
        required=("source_binding_id", "fields"),
    )


def _join_key_schema() -> dict[str, object]:
    return strict_object(
        {
            "left_field_id": field_id_schema(),
            "right_field_id": field_id_schema(),
        },
        required=("left_field_id", "right_field_id"),
    )


def _join_key_schema_for_sources(
    *,
    left_field_ids: tuple[str, ...],
    right_field_ids: tuple[str, ...],
) -> dict[str, object]:
    return strict_object(
        {
            "left_field_id": field_id_schema(left_field_ids),
            "right_field_id": field_id_schema(right_field_ids),
        },
        required=("left_field_id", "right_field_id"),
    )


def _joined_field_schema() -> dict[str, object]:
    return strict_object(
        {
            "side": {"enum": ["left", "right"]},
            "field_id": field_id_schema(),
        },
        required=("side", "field_id"),
    )


def _joined_field_schema_for_sources(
    *,
    left_field_ids: tuple[str, ...],
    right_field_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "oneOf": [
            strict_object(
                {
                    "side": {"enum": ["left"]},
                    "field_id": field_id_schema(left_field_ids),
                },
                required=("side", "field_id"),
            ),
            strict_object(
                {
                    "side": {"enum": ["right"]},
                    "field_id": field_id_schema(right_field_ids),
                },
                required=("side", "field_id"),
            ),
        ]
    }
