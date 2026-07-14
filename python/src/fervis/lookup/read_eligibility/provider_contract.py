"""Typed provider-output contracts for read eligibility."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class ReadCandidateReviewOutput(ProviderOutput):
    source_candidate_id: str
    read_id: str
    relevant_row_path_tokens: tuple[str, ...]
    relevant_field_tokens: tuple[str, ...]
    retention_basis: str
    retention_decision: str


@dataclass(frozen=True)
class RequestedFactAssessmentOutput(ProviderOutput):
    requested_fact_id: str
    read_candidate_reviews: tuple[ReadCandidateReviewOutput, ...]


@dataclass(frozen=True)
class ReadEligibilityOutput(ProviderOutput):
    requested_fact_assessments: tuple[RequestedFactAssessmentOutput, ...]
