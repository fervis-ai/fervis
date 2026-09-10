"""Provider DTOs for semantic Read Eligibility."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class ReadRequirementAssessmentOutput(ProviderOutput):
    assessment_basis: str
    relevant_field_refs: tuple[str, ...]
    decision: str


@dataclass(frozen=True)
class CanonicalOptionAssessmentOutput(ProviderOutput):
    canonical_option_id: str
    assessment: str
    decision: str


@dataclass(frozen=True)
class ResolverRouteAssessmentOutput(ProviderOutput):
    resolver_route_id: str
    assessment: str
    decision: str


@dataclass(frozen=True)
class IdentityRouteOutcomeOutput(ProviderOutput):
    canonical_option_assessments: tuple[CanonicalOptionAssessmentOutput, ...]
    canonical_option_basis: str
    canonical_option_id: str | None
    resolver_route_assessments: tuple[ResolverRouteAssessmentOutput, ...]
    resolver_route_basis: str
    resolver_route_id: str | None
    evidence_refs: tuple[str, ...]
    outcome: str


@dataclass(frozen=True)
class SemanticReadEligibilityOutput(ProviderOutput):
    read_assessments_by_requested_fact: dict[
        str, dict[str, ReadRequirementAssessmentOutput]
    ]
    identity_outcomes: dict[str, IdentityRouteOutcomeOutput]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
