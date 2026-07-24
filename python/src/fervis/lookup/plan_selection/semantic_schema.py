"""Strict provider schema for semantic Plan Selection."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.plan_selection import semantic_provider_contract as output
from fervis.lookup.plan_selection.semantic import SemanticPlanSelectionRequest


def build_semantic_plan_selection_schema(
    request: SemanticPlanSelectionRequest,
) -> dict[str, object]:
    source_refs = tuple(source.id for source in request.source_catalog.sources)
    assessments = {
        index.requested_fact_id: _closed_object(
            {
                source_ref: output.SourceAlignmentAssessmentOutput.schema(
                    {
                        "basis": {"type": "string", "minLength": 1},
                        "alignment": {
                            "enum": [
                                item.value
                                for item in request.allowed_alignments(
                                    requested_fact_id=index.requested_fact_id,
                                    source_ref=source_ref,
                                )
                            ]
                        },
                    }
                )
                for source_ref in source_refs
            }
        )
        for index in request.indexes
    }
    return output.SemanticPlanSelectionOutput.schema(
        {"source_assessments_by_requested_fact": _closed_object(assessments)}
    )


def _closed_object(properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


__all__ = ["build_semantic_plan_selection_schema"]
