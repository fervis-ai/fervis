"""Provider schema for fact-local plan selection."""

from __future__ import annotations

from fervis.lookup.plan_selection import provider_contract as provider_output
from fervis.lookup.plan_selection.model import SourceAlignment


def build_plan_selection_schema(
    *,
    requested_fact_ids: tuple[str, ...],
    source_candidate_ids_by_requested_fact_id: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    outcome_schema = provider_output.SourceAlignmentReviewsOutput.schema(
        {
            "kind": {"enum": ["source_alignment_reviews"]},
            "reviews_by_requested_fact": _source_alignment_reviews_schema(
                requested_fact_ids=requested_fact_ids,
                source_candidate_ids_by_requested_fact_id=(
                    source_candidate_ids_by_requested_fact_id
                ),
            ),
        },
    )
    schema = provider_output.PlanSelectionOutput.schema({"outcome": outcome_schema})
    return {
        **schema,
        "modelSchemas": {"outcome": outcome_schema},
    }


def _source_alignment_reviews_schema(
    *,
    requested_fact_ids: tuple[str, ...],
    source_candidate_ids_by_requested_fact_id: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    fact_schemas: dict[str, object] = {}
    for requested_fact_id in requested_fact_ids:
        fact_schemas[requested_fact_id] = _fact_source_alignment_schema(
            requested_fact_id=requested_fact_id,
            source_candidate_ids=source_candidate_ids_by_requested_fact_id.get(
                requested_fact_id,
                (),
            ),
        )
    return _strict_object(fact_schemas, required=requested_fact_ids)


def _fact_source_alignment_schema(
    *,
    requested_fact_id: str,
    source_candidate_ids: tuple[str, ...],
) -> dict[str, object]:
    del requested_fact_id
    return _strict_object(
        {
            source_candidate_id: _source_alignment_review_schema(source_candidate_id)
            for source_candidate_id in source_candidate_ids
        },
        required=source_candidate_ids,
    )


def _source_alignment_review_schema(source_candidate_id: str) -> dict[str, object]:
    return provider_output.SourceAlignmentReviewOutput.schema(
        {
            "source_candidate_id": {"enum": [source_candidate_id]},
            "basis": {"type": "string", "minLength": 1},
            "source_alignment": {"enum": [item.value for item in SourceAlignment]},
        },
    )


def _strict_object(
    properties: dict[str, object],
    *,
    required: tuple[str, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }
