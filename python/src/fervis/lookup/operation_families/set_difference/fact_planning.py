"""Fact-planning schema fragments for set-difference answers."""

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


SET_DIFFERENCE_PATTERN_NAMES = frozenset({"set_difference"})


def set_difference_generic_pattern_answer_variants(
    *,
    requested_fact_id_schema: dict[str, object] | None,
    require_pattern: bool,
) -> list[dict[str, object]]:
    schema = _set_difference_generic_pattern_schema(
        requested_fact_id_schema=requested_fact_id_schema,
    )
    return [schema if require_pattern else optional_pattern_schema(schema)]


def set_difference_pattern_answer_variants(
    *,
    requested_fact_id_schema: dict[str, object],
    answer_output_ids_schema: dict[str, object] | None,
    source_binding_ids: tuple[str, ...],
    source_binding_ids_by_requirement: Mapping[str, tuple[str, ...]],
    field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    identity_field_ids_by_source_binding_id: Mapping[str, tuple[str, ...]],
    require_pattern: bool,
) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    candidate_source_ids = source_binding_ids_by_requirement.get(
        "candidate_set",
        source_binding_ids,
    )
    observed_source_ids = source_binding_ids_by_requirement.get(
        "observed_set",
        source_binding_ids,
    )
    for candidate_source_id in candidate_source_ids:
        candidate_field_ids = field_ids_by_source_binding_id.get(
            candidate_source_id, ()
        )
        candidate_identity_field_ids = identity_field_ids_by_source_binding_id.get(
            candidate_source_id, ()
        )
        if not candidate_field_ids or not candidate_identity_field_ids:
            continue
        for observed_source_id in observed_source_ids:
            observed_field_ids = field_ids_by_source_binding_id.get(
                observed_source_id,
                (),
            )
            observed_identity_field_ids = identity_field_ids_by_source_binding_id.get(
                observed_source_id, ()
            )
            if not observed_field_ids or not observed_identity_field_ids:
                continue
            schema = _set_difference_pattern_schema(
                requested_fact_id_schema=requested_fact_id_schema,
                answer_output_ids_schema=answer_output_ids_schema,
                candidate_source_id=candidate_source_id,
                candidate_field_ids=candidate_field_ids,
                candidate_identity_field_ids=candidate_identity_field_ids,
                observed_source_id=observed_source_id,
                observed_field_ids=observed_field_ids,
                observed_identity_field_ids=observed_identity_field_ids,
            )
            variants.append(
                schema if require_pattern else optional_pattern_schema(schema)
            )
    return variants


def _set_difference_generic_pattern_schema(
    *,
    requested_fact_id_schema: dict[str, object] | None,
) -> dict[str, object]:
    return strict_object(
        {
            "requested_fact_id": requested_fact_id_schema or handle_schema(),
            "answer_output_ids": non_empty_string_array(),
            "pattern": {"enum": ["set_difference"]},
            "candidate": _set_difference_candidate_schema(),
            "observed": _set_difference_observed_schema(),
        },
        required=(
            "requested_fact_id",
            "answer_output_ids",
            "pattern",
            "candidate",
            "observed",
        ),
    )


def _set_difference_pattern_schema(
    *,
    requested_fact_id_schema: dict[str, object],
    answer_output_ids_schema: dict[str, object] | None,
    candidate_source_id: str,
    candidate_field_ids: tuple[str, ...],
    candidate_identity_field_ids: tuple[str, ...],
    observed_source_id: str,
    observed_field_ids: tuple[str, ...],
    observed_identity_field_ids: tuple[str, ...],
) -> dict[str, object]:
    return strict_object(
        {
            "requested_fact_id": requested_fact_id_schema,
            "answer_output_ids": answer_output_ids_schema or non_empty_string_array(),
            "pattern": {"enum": ["set_difference"]},
            "candidate": _set_difference_candidate_schema_for_source(
                source_binding_id=candidate_source_id,
                field_ids=candidate_field_ids,
                identity_field_ids=candidate_identity_field_ids,
            ),
            "observed": _set_difference_observed_schema_for_source(
                source_binding_id=observed_source_id,
                identity_field_ids=observed_identity_field_ids,
            ),
        },
        required=(
            "requested_fact_id",
            "answer_output_ids",
            "pattern",
            "candidate",
            "observed",
        ),
    )


def _set_difference_candidate_schema() -> dict[str, object]:
    return strict_object(
        {
            "source_binding_id": handle_schema(),
            "identity_fields": non_empty_string_array(),
            "output_fields": non_empty_array_items(field_selection_schema()),
        },
        required=("source_binding_id", "identity_fields", "output_fields"),
    )


def _set_difference_candidate_schema_for_source(
    *,
    source_binding_id: str,
    field_ids: tuple[str, ...],
    identity_field_ids: tuple[str, ...],
) -> dict[str, object]:
    return strict_object(
        {
            "source_binding_id": {"enum": [source_binding_id]},
            "identity_fields": _non_empty_field_id_array(identity_field_ids),
            "output_fields": non_empty_array_items(
                field_selection_schema(field_ids=field_ids)
            ),
        },
        required=("source_binding_id", "identity_fields", "output_fields"),
    )


def _set_difference_observed_schema() -> dict[str, object]:
    return strict_object(
        {
            "source_binding_id": handle_schema(),
            "identity_fields": non_empty_string_array(),
        },
        required=("source_binding_id", "identity_fields"),
    )


def _set_difference_observed_schema_for_source(
    *,
    source_binding_id: str,
    identity_field_ids: tuple[str, ...],
) -> dict[str, object]:
    return strict_object(
        {
            "source_binding_id": {"enum": [source_binding_id]},
            "identity_fields": _non_empty_field_id_array(identity_field_ids),
        },
        required=("source_binding_id", "identity_fields"),
    )


def _non_empty_field_id_array(field_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "array",
        "items": field_id_schema(field_ids),
        "minItems": 1,
    }
