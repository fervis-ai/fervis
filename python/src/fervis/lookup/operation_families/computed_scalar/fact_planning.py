"""Fact-planning schema fragments for computed scalar answers."""

from __future__ import annotations

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    optional_pattern_schema,
)
from fervis.lookup.fact_planning.schema_helpers import (
    field_id_schema,
    handle_schema,
    non_empty_array_items,
    non_empty_string_array,
    strict_object,
)


COMPUTED_SCALAR_PATTERN_NAMES = frozenset({"computed_scalar"})


def computed_scalar_pattern_answer_variants(
    *,
    requested_fact_id_schema: dict[str, object] | None,
    require_pattern: bool,
) -> list[dict[str, object]]:
    schema = _computed_scalar_pattern_schema(
        requested_fact_id_schema=requested_fact_id_schema,
    )
    return [schema if require_pattern else optional_pattern_schema(schema)]


def _computed_scalar_pattern_schema(
    *,
    requested_fact_id_schema: dict[str, object] | None,
) -> dict[str, object]:
    return strict_object(
        {
            "requested_fact_id": requested_fact_id_schema or handle_schema(),
            "answer_output_ids": non_empty_string_array(),
            "pattern": {"enum": ["computed_scalar"]},
            "scalar_inputs": non_empty_array_items(_source_scalar_input_schema()),
            "expression": non_empty_array_items(_compute_expression_token_schema()),
            "output": _scalar_output_schema(),
        },
        required=(
            "requested_fact_id",
            "answer_output_ids",
            "pattern",
            "scalar_inputs",
            "expression",
            "output",
        ),
    )


def _source_scalar_input_schema() -> dict[str, object]:
    return strict_object(
        {
            "input_id": field_id_schema(),
            "source_binding_id": handle_schema(),
        },
        required=("input_id", "source_binding_id"),
    )


def _compute_expression_token_schema() -> dict[str, object]:
    return {
        "oneOf": [
            strict_object(
                {"input_id": field_id_schema()},
                required=("input_id",),
            ),
            strict_object(
                {
                    "operator": {
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide",
                            "negate",
                        ]
                    }
                },
                required=("operator",),
            ),
        ]
    }


def _scalar_output_schema() -> dict[str, object]:
    return strict_object(
        {
            "scalar_id": field_id_schema(),
            "label": {"type": "string"},
        },
        required=("scalar_id",),
    )
