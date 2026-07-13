"""Lookup catalog selection public boundary."""

from fervis.lookup.relation_catalog.selection.model import (
    AnswerOutputResourceLineage,
    CatalogSelectionRanking,
    CatalogSelectionRequest,
    CatalogSelectionResult,
    EntityTargetCatalogSearchTerms,
    EntityTargetResolverSelection,
    ResolverCatalogSelectionRequest,
    ResolverCatalogSelectionResult,
    RequestedFactCatalogSelection,
    RequestedFactResourceNameMatches,
    catalog_selection_evidence_ref,
)
from fervis.lookup.relation_catalog.selection.results import (
    dedupe_read_ids,
    relation_catalog_for_read_ids,
    selected_read_ids_from_fact_selections,
)
from fervis.lookup.relation_catalog.selection.selector import (
    DEFAULT_MAX_CATALOG_READS_PER_FACT,
    select_resolver_relation_catalog,
    select_relation_catalog,
)

__all__ = [
    "AnswerOutputResourceLineage",
    "CatalogSelectionRanking",
    "CatalogSelectionRequest",
    "CatalogSelectionResult",
    "DEFAULT_MAX_CATALOG_READS_PER_FACT",
    "EntityTargetCatalogSearchTerms",
    "EntityTargetResolverSelection",
    "ResolverCatalogSelectionRequest",
    "ResolverCatalogSelectionResult",
    "RequestedFactCatalogSelection",
    "RequestedFactResourceNameMatches",
    "catalog_selection_evidence_ref",
    "dedupe_read_ids",
    "relation_catalog_for_read_ids",
    "select_resolver_relation_catalog",
    "select_relation_catalog",
    "selected_read_ids_from_fact_selections",
]
