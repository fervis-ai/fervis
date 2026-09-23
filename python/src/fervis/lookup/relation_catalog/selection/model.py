"""Typed catalog-selection contract for Lookup planning."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog import RelationCatalog


@dataclass(frozen=True)
class CatalogSelectionRanking:
    read_id: str
    score: int
    matched_terms: tuple[str, ...] = ()
    matched_fact_refs: tuple[str, ...] = ()
    matched_field_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestedFactCatalogSelection:
    requested_fact_id: str
    query_terms: tuple[str, ...]
    rankings: tuple[CatalogSelectionRanking, ...]
    selected_read_ids: tuple[str, ...]
    unselected_positive_read_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogSelectionResult:
    relation_catalog: RelationCatalog
    requested_fact_selections: tuple[RequestedFactCatalogSelection, ...]
    selected_read_ids: tuple[str, ...]


def catalog_selection_evidence_ref(*, requested_fact_id: str) -> str:
    return f"catalog_selection:{requested_fact_id}"
