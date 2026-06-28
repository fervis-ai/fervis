"""Public read-eligibility candidate surface."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.read_eligibility.candidate_scope import (
    ReadEligibilityCandidateScope,
    read_eligibility_candidate_scopes_from_cards,
)
from fervis.lookup.read_eligibility.cards import read_eligibility_cards_payload
from fervis.lookup.read_eligibility.model import ReadEligibilityRequest


@dataclass(frozen=True)
class ReadEligibilityCandidateSurface:
    card_payload: dict[str, object]
    candidate_scopes: tuple[ReadEligibilityCandidateScope, ...]


def read_eligibility_candidate_surface(
    request: ReadEligibilityRequest,
) -> ReadEligibilityCandidateSurface:
    card_payload = read_eligibility_cards_payload(
        requested_facts=request.requested_facts,
        catalog_selection=request.catalog_selection,
        available_values=request.available_values,
    )
    return ReadEligibilityCandidateSurface(
        card_payload=card_payload,
        candidate_scopes=read_eligibility_candidate_scopes_from_cards(
            request,
            card_payload=card_payload,
        ),
    )
