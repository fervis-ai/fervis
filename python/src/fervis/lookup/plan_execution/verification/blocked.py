"""Blocked-fact shared helpers for fact-plan verification."""

from ._shared import (
    CatalogFactAvailability,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
    RowSourceCatalog,
    RowSourceKind,
)


def _required_reviewed_read_ids(
    requested_fact_id: str,
    *,
    row_sources: RowSourceCatalog,
    catalog_selection: CatalogSelectionResult | None,
) -> tuple[str, ...] | None:
    if catalog_selection is None:
        return tuple(
            dict.fromkeys(
                source.read_id
                for source in row_sources.sources
                if source.kind == RowSourceKind.API_READ and source.read_id
            )
        )
    selection = _catalog_selection_for_fact(
        requested_fact_id,
        catalog_selection=catalog_selection,
    )
    if selection is None:
        return None
    return tuple(selection.selected_read_ids)


def _catalog_selection_for_fact(
    requested_fact_id: str,
    *,
    catalog_selection: CatalogSelectionResult | None,
) -> RequestedFactCatalogSelection | None:
    if catalog_selection is None:
        return None
    for selection in catalog_selection.requested_fact_selections:
        if selection.requested_fact_id == requested_fact_id:
            return selection
    return None


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
