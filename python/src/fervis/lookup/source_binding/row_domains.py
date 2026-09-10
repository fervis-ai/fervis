"""Remove row candidates lacking a declared way to connect their logical roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fervis.lookup.question_contract import AssociationTerm, SetTerm
from .association_choices import endpoint_realizations

if TYPE_CHECKING:
    from .model import SemanticSourceBindingRequest


def connected_row_domains(
    request: SemanticSourceBindingRequest,
) -> dict[str, tuple[str, ...]]:
    domains = {
        ref.token: request._local_row_references_for_set(ref.token)
        for ref in request.index.source_requirement_refs
        if isinstance(request.index.term_by_ref[ref], SetTerm)
    }
    associations = []
    for ref in request.index.association_requirement_refs:
        term = request.index.term_by_ref[ref]
        assert isinstance(term, AssociationTerm)
        associations.append(
            tuple(
                request.index.fact_local_ref_by_local_id[local].token
                for local in (term.from_set_ref, term.to_set_ref)
            )
        )
    # Arc consistency removes only candidates with no possible neighbor. It does
    # not choose a semantic mapping or assert that a complete assignment is valid.
    changed = True
    while changed:
        changed = False
        for left, right in associations:
            for owner, neighbor in ((left, right), (right, left)):
                remaining = tuple(
                    candidate
                    for candidate in domains[owner]
                    if any(
                        endpoint_realizations(request, candidate, other)
                        for other in domains[neighbor]
                    )
                )
                if remaining != domains[owner]:
                    domains[owner] = remaining
                    changed = True
    return domains
