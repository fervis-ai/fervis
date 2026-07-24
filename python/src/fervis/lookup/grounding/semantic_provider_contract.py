"""Provider output for semantic identity-route compatibility."""

from dataclasses import dataclass

from fervis.lookup.grounding.time_resolution.provider_contract import (
    KnownTimeResolutionOutput,
)
from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class ResolverMechanicsOutput(ProviderOutput):
    decision: str
    lookup_request_params: tuple[str, ...]
    returned_identity_verification_fields: tuple[str, ...]


@dataclass(frozen=True)
class IdentityRouteReviewOutput(ProviderOutput):
    assessment_basis: str
    resolution: ResolverMechanicsOutput


@dataclass(frozen=True)
class ResourceTypeReviewOutput(ProviderOutput):
    compatibility_basis: str
    compatibility: str
    route_reviews: dict[str, IdentityRouteReviewOutput]


@dataclass(frozen=True)
class IdentityGroundingReviewOutput(ProviderOutput):
    identifier_kind_basis: str
    identifier_kind: str
    purpose: str
    resource_type_reviews: dict[str, ResourceTypeReviewOutput]


@dataclass(frozen=True)
class SemanticGroundingOutput(ProviderOutput):
    time_resolutions: dict[str, KnownTimeResolutionOutput]
    reference_reviews: dict[str, ProviderObject]


__all__ = tuple(name for name in globals() if not name.startswith("_"))
