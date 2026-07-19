"""Fact-planning schema fragments for grouped aggregate choices."""

from __future__ import annotations

from fervis.lookup.fact_planning.fact_planning_family_schema import (
    optional_pattern_schema,
)
from fervis.lookup.fact_planning.schema_helpers import (
    field_id_schema,
    strict_object,
)


GROUPED_AGGREGATE_PATTERN_NAMES = frozenset(
    {
        "aggregate_by_group",
    }
)


def grouped_aggregate_pattern_answer_variants(
    *,
    plan_shape: str,
    requested_fact_id_schema: dict[str, object],
    choices: tuple[dict[str, object], ...],
    require_pattern: bool,
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for choice in choices:
        if not _valid_choice(choice):
            continue
        schema = _grouped_aggregate_pattern_answer_schema(
            plan_shape=plan_shape,
            requested_fact_id_schema=requested_fact_id_schema,
            choice=choice,
        )
        variants.append(schema if require_pattern else optional_pattern_schema(schema))
    return variants


def _grouped_aggregate_pattern_answer_schema(
    *,
    plan_shape: str,
    requested_fact_id_schema: dict[str, object],
    choice: dict[str, object],
) -> dict[str, object]:
    properties: dict[str, object] = {
        "requested_fact_id": requested_fact_id_schema,
        "pattern": {"enum": [plan_shape]},
        "source_binding_id": {"enum": [_text(choice.get("source_binding_id"))]},
        "metric": _candidate_selection_schema(
            choice.get("metric_candidates"),
            kind="metric",
        ),
        "function": _candidate_selection_schema(
            choice.get("function_candidates"),
            kind="function",
        ),
    }
    if choice.get("ordering_required"):
        group = choice.get("group")
        group_payload = group if isinstance(group, dict) else {}
        ordering_field_ids = tuple(
            str(field_id)
            for field_id in group_payload.get("field_ids", ())
        )
        if ordering_field_ids:
            properties["ordering_field"] = strict_object(
                {
                    "selection_basis": {"type": "string", "minLength": 1},
                    "field_id": field_id_schema(ordering_field_ids),
                },
                required=("selection_basis", "field_id"),
            )
    required: tuple[str, ...] = (
        "requested_fact_id",
        "pattern",
        "source_binding_id",
        "metric",
        "function",
    )
    return strict_object(properties, required=required)


def _candidate_selection_schema(
    raw_candidates: object, *, kind: str
) -> dict[str, object]:
    variants = [
        _candidate_schema(candidate, kind=kind)
        for candidate in _dict_candidates(raw_candidates)
    ]
    if len(variants) == 1:
        return variants[0]
    return {"oneOf": variants}


def _candidate_schema(candidate: dict[str, object], *, kind: str) -> dict[str, object]:
    properties: dict[str, object] = {
        "selection_basis": {"type": "string", "minLength": 1},
        "id": {"enum": [_text(candidate.get("id"))]},
    }
    required: tuple[str, ...] = ("selection_basis", "id")
    if kind == "metric":
        metric_kind = _text(candidate.get("kind"))
        properties["kind"] = {"enum": [metric_kind]}
        required = (*required, "kind")
        if metric_kind == "aggregate_field":
            properties["field_id"] = {"enum": [_text(candidate.get("field_id"))]}
            required = (*required, "field_id")
    elif kind == "function":
        properties["value"] = {"enum": [_text(candidate.get("value"))]}
        required = (*required, "value")
    else:
        raise ValueError(f"unsupported grouped aggregate candidate kind: {kind}")
    return strict_object(properties, required=required)


def _valid_choice(choice: dict[str, object]) -> bool:
    return bool(
        _text(choice.get("source_binding_id"))
        and isinstance(choice.get("group"), dict)
        and _dict_candidates(choice.get("metric_candidates"))
        and _dict_candidates(choice.get("function_candidates"))
    )


def _dict_candidates(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, tuple):
        if not isinstance(value, list):
            return ()
    return tuple(item for item in value if isinstance(item, dict))


def _text(value: object) -> str:
    return str(value or "").strip()
