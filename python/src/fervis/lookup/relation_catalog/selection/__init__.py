"""Lookup catalog selection public boundary."""

from fervis.lookup.relation_catalog.selection.model import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
    catalog_selection_evidence_ref,
)
from fervis.lookup.relation_catalog.selection.results import (
    dedupe_read_ids,
    relation_catalog_for_read_ids,
    selected_read_ids_from_fact_selections,
)
from fervis.lookup.relation_catalog.selection.selector import (
    DEFAULT_MAX_CATALOG_READS_PER_FACT,
    SemanticCatalogSelectionRequest,
    select_resolver_reads,
    select_semantic_relation_catalog,
)

__all__ = [
    "CatalogSelectionRanking",
    "CatalogSelectionResult",
    "DEFAULT_MAX_CATALOG_READS_PER_FACT",
    "SemanticCatalogSelectionRequest",
    "RequestedFactCatalogSelection",
    "catalog_selection_evidence_ref",
    "dedupe_read_ids",
    "relation_catalog_for_read_ids",
    "select_resolver_reads",
    "select_semantic_relation_catalog",
    "selected_read_ids_from_fact_selections",
]
