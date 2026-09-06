"""Declared relationship realizations coupled to their endpoint row identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fervis.lookup.question_contract import AssociationTerm, FactLocalRef
from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind

if TYPE_CHECKING:
    from .model import SemanticSourceBindingRequest


@dataclass(frozen=True)
class AssociationChoice:
    from_rows_ref: str
    to_rows_ref: str
    realization_ref: str
    reference_from_set_ref: str | None = None


def association_endpoints(
    request: SemanticSourceBindingRequest, ref: str
) -> tuple[str, str]:
    term = request.index.term_by_ref[FactLocalRef.from_token(ref)]
    assert isinstance(term, AssociationTerm)
    return (
        request.index.fact_local_ref_by_local_id[term.from_set_ref].token,
        request.index.fact_local_ref_by_local_id[term.to_set_ref].token,
    )


def endpoint_realizations(
    request: SemanticSourceBindingRequest,
    left: str,
    right: str,
) -> tuple[tuple[str, bool | None], ...]:
    """Evidence and source-end orientation; None means orientation is unambiguous."""
    identities = {
        item.identity_ref: item for item in request.source_catalog.identity_evidence
    }
    left_id, right_id = identities.get(left), identities.get(right)
    left_source = left_id.source_ref if left_id else left
    right_source = right_id.source_ref if right_id else right
    result: list[tuple[str, bool | None]] = []
    if left_source == right_source and any(
        identity is not None and identity.kind is RowSourceIdentityKind.ENTITY_REFERENCE
        for identity in (left_id, right_id)
    ):
        result.append((left_source, None))
    for edge in request.source_catalog.relation_evidence:
        pair = (edge.left_source_ref, edge.right_source_ref)
        if pair == (left_source, right_source):
            if left_source == right_source:
                result.extend(((edge.evidence_ref, True), (edge.evidence_ref, False)))
            else:
                result.append((edge.evidence_ref, None))
        elif pair == (right_source, left_source):
            result.append((edge.evidence_ref, None))
    return tuple(result)


def association_choices(
    request: SemanticSourceBindingRequest, ref: str
) -> tuple[AssociationChoice, ...]:
    left_set, right_set = association_endpoints(request, ref)
    return tuple(
        AssociationChoice(
            left,
            right,
            evidence,
            (left_set if orientation else right_set)
            if orientation is not None
            else None,
        )
        for left in request.row_references_for_set(left_set)
        for right in request.row_references_for_set(right_set)
        for evidence, orientation in endpoint_realizations(request, left, right)
    )
