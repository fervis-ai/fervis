"""Typed provider-output contracts for read eligibility."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class CanonicalInputSelectionOutput(ProviderOutput):
    interpretation_question: str
    because: str
    canonical_option_id: str


@dataclass(frozen=True)
class RetainedReadReviewOutput(ProviderOutput):
    relevant_row_path_tokens: tuple[str, ...]
    relevant_field_tokens: tuple[str, ...]
    retention_basis: str
    retention_decision: str


@dataclass(frozen=True)
class DroppedReadReviewOutput(ProviderOutput):
    retention_basis: str
    retention_decision: str


@dataclass(frozen=True)
class RequestedFactAssessmentOutput(ProviderOutput):
    canonical_inputs: dict[str, CanonicalInputSelectionOutput]
    read_candidate_reviews: dict[str, ProviderObject]


@dataclass(frozen=True)
class ReadEligibilityOutput(ProviderOutput):
    requested_fact_assessments: dict[str, RequestedFactAssessmentOutput]
