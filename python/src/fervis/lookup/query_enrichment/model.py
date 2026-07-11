"""Typed model for catalog query enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import (
    EntityTargetCatalogSearchTerms,
    RequestedFactResourceNameMatches,
)
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.turn_prompts.context import HostPromptContext


QUERY_ENRICHMENT_MAX_CATALOG_SEARCH_TERMS = 5


@dataclass(frozen=True)
class QueryEnrichmentRequest:
    question: str
    conversation_context: dict[str, object]
    requested_facts: tuple[RequestedFact, ...]
    relation_catalog: RelationCatalog
    host: HostPromptContext = field(default_factory=HostPromptContext)


@dataclass(frozen=True)
class QueryEnrichmentResult:
    requested_fact_resource_name_matches: tuple[RequestedFactResourceNameMatches, ...]
    entity_target_catalog_search_terms: tuple[EntityTargetCatalogSearchTerms, ...] = ()


def query_enrichment_resource_names(request: QueryEnrichmentRequest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                resource_name
                for read in request.relation_catalog.reads
                for resource_name in read.resource_names
                if resource_name
            }
        )
    )


def query_enrichment_endpoint_names(request: QueryEnrichmentRequest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                read.endpoint_name
                for read in request.relation_catalog.reads
                if read.endpoint_name
            }
        )
    )
