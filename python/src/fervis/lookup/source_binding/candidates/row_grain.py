"""Relational grain checks for selectable source fulfillment."""

from __future__ import annotations

from fervis.lookup.source_binding.candidates.contracts import (
    CandidateKeyEvidence,
    EntityEvidence,
    EntityReferenceEvidence,
    FulfillmentSupportSet,
)
from fervis.lookup.source_binding.candidates.model import SourceCandidate


def fulfillment_preserves_row_grain(
    candidate: SourceCandidate,
    fulfillment_support_set_id: str,
) -> bool:
    support_set = _support_set(candidate, fulfillment_support_set_id)
    if support_set is None:
        return False
    entity_evidence = tuple(
        evidence
        for slot in support_set.fulfillment_slots
        for evidence in slot.entity_evidence
    )
    return all(
        _entity_evidence_preserves_row_grain(evidence, candidate=candidate)
        for evidence in entity_evidence
    )


def _support_set(
    candidate: SourceCandidate,
    fulfillment_support_set_id: str,
) -> FulfillmentSupportSet | None:
    return next(
        (
            support_set
            for support_set in candidate.fulfillment_support_sets
            if fulfillment_support_set_id
            in {
                support_set.fulfillment_support_set_id,
                support_set.fulfillment_choice_id,
            }
        ),
        None,
    )


def _entity_evidence_preserves_row_grain(
    evidence: EntityEvidence,
    *,
    candidate: SourceCandidate,
) -> bool:
    if isinstance(evidence, CandidateKeyEvidence):
        return True
    return _entity_reference_is_row_unique(evidence, candidate=candidate)


def _entity_reference_is_row_unique(
    reference: EntityReferenceEvidence,
    *,
    candidate: SourceCandidate,
) -> bool:
    reference_field_ids = {component.field_id for component in reference.components}
    return any(
        key.row_source_id == reference.row_source_id
        and {component.field_id for component in key.components}
        <= reference_field_ids
        for key in candidate.evidence_items
        if isinstance(key, CandidateKeyEvidence)
    )
