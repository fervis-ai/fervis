"""Public read-eligibility candidate surface."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.read_eligibility.candidate_scope import (
    ReadEligibilityCandidateScope,
    read_eligibility_candidate_scopes_from_cards,
)
from fervis.lookup.read_eligibility.cards import build_read_eligibility_cards
from fervis.lookup.read_eligibility.model import (
    CanonicalInputOption,
    ReadEligibilityRequest,
)


@dataclass(frozen=True)
class ReadEligibilityCandidateSurface:
    card_payload: dict[str, object]
    candidate_scopes: tuple[ReadEligibilityCandidateScope, ...]
    canonical_options: tuple[CanonicalInputOption, ...]


def read_eligibility_candidate_surface(
    request: ReadEligibilityRequest,
) -> ReadEligibilityCandidateSurface:
    cards = build_read_eligibility_cards(
        requested_facts=request.requested_facts,
        catalog_selection=request.catalog_selection,
        resolver_catalog=request.resolver_catalog,
        binding_tasks=request.binding_tasks,
        compatible_reference_bindings=request.compatible_reference_bindings,
        canonical_values=request.canonical_values,
    )
    return ReadEligibilityCandidateSurface(
        card_payload=cards.payload,
        candidate_scopes=read_eligibility_candidate_scopes_from_cards(
            request,
            card_payload=cards.payload,
        ),
        canonical_options=cards.canonical_options,
    )
