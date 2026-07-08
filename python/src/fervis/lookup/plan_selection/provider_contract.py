"""Provider-output DTOs for plan selection."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


PlanSelectionOutput = provider_output_type("PlanSelectionOutput", ("outcome",))
SourceAlignmentReviewsOutput = provider_output_type(
    "SourceAlignmentReviewsOutput",
    ("kind", "reviews_by_requested_fact"),
)
SourceAlignmentReviewOutput = provider_output_type(
    "SourceAlignmentReviewOutput",
    ("source_candidate_id", "basis", "source_alignment"),
)
