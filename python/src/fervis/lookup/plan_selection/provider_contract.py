"""Typed provider-output contracts for plan selection."""

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class SourceAlignmentReviewOutput(ProviderOutput):
    source_candidate_id: str
    basis: str
    source_alignment: str


@dataclass(frozen=True)
class SourceAlignmentReviewsOutput(ProviderOutput):
    kind: str
    reviews_by_requested_fact: dict[str, ProviderObject]


@dataclass(frozen=True)
class PlanSelectionOutput(ProviderOutput):
    outcome: SourceAlignmentReviewsOutput
