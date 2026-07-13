"""Blocked-fact shared helpers for fact-plan verification."""

from ._shared import (
    CatalogFactAvailability,
    RowSourceCatalog,
)


def _policy_blocked_evidence_refs(
    row_sources: RowSourceCatalog,
) -> frozenset[str]:
    return frozenset(
        ref
        for source in row_sources.sources
        for fact in source.blocked_facts
        if fact.availability == CatalogFactAvailability.POLICY_BLOCKED
        for ref in fact.proof_refs
    )
