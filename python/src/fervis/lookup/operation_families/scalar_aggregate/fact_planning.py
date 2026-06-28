"""Fact-planning schema fragments for scalar-aggregate answers."""

from __future__ import annotations

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    SourceBoundPatternSchemaContext,
    source_bound_pattern_base,
    source_bound_pattern_required,
)
from fervis.lookup.fact_planning.schema_helpers import strict_object


def scalar_aggregate_pattern_answer_variants(
    *,
    requested_fact_id_schema: dict[str, object],
    choices: tuple[dict[str, object], ...],
    require_pattern: bool,
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for choice in choices:
        metric_variants = _metric_variants(choice.get("metric_candidates") or ())
        function_variants = _function_variants(choice.get("function_candidates") or ())
        if not metric_variants or not function_variants:
            continue
        schema = strict_object(
            {
                "requested_fact_id": requested_fact_id_schema,
                "answer_output_ids": _answer_output_ids_schema(choice),
                "pattern": {"enum": ["aggregate_scalar"]},
                "source_binding_id": {
                    "enum": [str(choice.get("source_binding_id") or "")]
                },
                "metric": {"oneOf": metric_variants},
                "function": {"oneOf": function_variants},
            },
            required=(
                "requested_fact_id",
                "answer_output_ids",
                "pattern",
                "source_binding_id",
                "metric",
                "function",
            ),
        )
        variants.append(schema if require_pattern else _without_pattern(schema))
    return variants


def aggregate_scalar_pattern_schema(
    context: SourceBoundPatternSchemaContext,
) -> dict[str, object]:
    return strict_object(
        {
            **source_bound_pattern_base(context),
            "pattern": {"enum": ["aggregate_scalar"]},
            "metric": _generic_metric_schema(),
            "function": _generic_function_schema(),
        },
        required=source_bound_pattern_required(context, "metric", "function"),
    )


SOURCE_BOUND_PATTERN_SCHEMA_BUILDERS = {
    "aggregate_scalar": aggregate_scalar_pattern_schema,
}


def _metric_variants(candidates: object) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for candidate in candidates if isinstance(candidates, (tuple, list)) else ():
        if not isinstance(candidate, dict):
            continue
        properties: dict[str, object] = {
            "selection_basis": {"type": "string", "minLength": 1},
            "id": {"enum": [str(candidate.get("id") or "")]},
            "kind": {"enum": [str(candidate.get("kind") or "")]},
        }
        required = ["selection_basis", "id", "kind"]
        if candidate.get("field_id"):
            properties["field_id"] = {"enum": [str(candidate["field_id"])]}
            required.append("field_id")
        variants.append(strict_object(properties, required=tuple(required)))
    return variants


def _function_variants(candidates: object) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for candidate in candidates if isinstance(candidates, (tuple, list)) else ():
        if not isinstance(candidate, dict):
            continue
        variants.append(
            strict_object(
                {
                    "selection_basis": {"type": "string", "minLength": 1},
                    "id": {"enum": [str(candidate.get("id") or "")]},
                    "value": {"enum": [str(candidate.get("value") or "")]},
                },
                required=("selection_basis", "id", "value"),
            )
        )
    return variants


def _answer_output_ids_schema(choice: dict[str, object]) -> dict[str, object]:
    answer_output_ids = tuple(
        dict.fromkeys(
            str(metric.get("answer_output_id") or "")
            for metric in choice.get("metric_candidates") or ()
            if isinstance(metric, dict) and str(metric.get("answer_output_id") or "")
        )
    )
    return {
        "type": "array",
        "items": {"type": "string", "enum": list(answer_output_ids)},
        "minItems": len(answer_output_ids),
        "maxItems": len(answer_output_ids),
    }


def _generic_metric_schema() -> dict[str, object]:
    return strict_object(
        {
            "selection_basis": {"type": "string", "minLength": 1},
            "id": {"type": "string", "minLength": 1},
            "kind": {"type": "string", "minLength": 1},
            "field_id": {"type": "string"},
        },
        required=("selection_basis", "id", "kind"),
    )


def _generic_function_schema() -> dict[str, object]:
    return strict_object(
        {
            "selection_basis": {"type": "string", "minLength": 1},
            "id": {"type": "string", "minLength": 1},
            "value": {"type": "string", "minLength": 1},
        },
        required=("selection_basis", "id", "value"),
    )


def _without_pattern(schema: dict[str, object]) -> dict[str, object]:
    return {
        **schema,
        "required": [item for item in schema.get("required", ()) if item != "pattern"],
    }
