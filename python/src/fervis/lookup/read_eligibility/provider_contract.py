"""Provider-output DTOs for read eligibility."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


ReadEligibilityOutput = provider_output_type(
    "ReadEligibilityOutput",
    ("requested_fact_assessments",),
)
RequestedFactAssessmentOutput = provider_output_type(
    "RequestedFactAssessmentOutput",
    ("requested_fact_id", "read_candidate_reviews"),
)
ReadCandidateReviewOutput = provider_output_type(
    "ReadCandidateReviewOutput",
    (
        "source_candidate_id",
        "read_id",
        "relevant_row_path_tokens",
        "relevant_field_tokens",
        "retention_basis",
        "retention_decision",
    ),
)
