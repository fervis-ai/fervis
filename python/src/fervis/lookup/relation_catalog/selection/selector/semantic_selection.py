"""Deterministic source recall from semantic requirement resource matches."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.query_enrichment import (
    RecallBucketMatch,
    semantic_recall_buckets,
)
from fervis.lookup.query_enrichment.semantic import SemanticRecallBucket
from fervis.lookup.question_contract import RequestedFactSemanticIndex
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog import CatalogFact
from fervis.lookup.relation_catalog.selection.model import (
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.relation_catalog.selection.results import (
    relation_catalog_for_read_ids,
    selected_read_ids_from_fact_selections,
)

from .fact_selection import (
    catalog_candidate_limit,
    merge_positive_rankings,
    merge_selected_rankings,
    resource_name_selection,
    unselected_positive_read_ids,
)
from .terms import _catalog_facts_by_read, _explicit_catalog_search_query_terms


@dataclass(frozen=True)
class FactRecallBuckets:
    requested_fact_id: str
    buckets: tuple[SemanticRecallBucket, ...]


@dataclass(frozen=True)
class SemanticCatalogSelectionRequest:
    relation_catalog: RelationCatalog
    indexes: tuple[RequestedFactSemanticIndex, ...]
    resource_matches: tuple[RecallBucketMatch, ...]
    max_reads_per_fact: int
    fact_buckets: tuple[FactRecallBuckets, ...] = ()

    @property
    def facts(self):
        return self.fact_buckets or tuple(FactRecallBuckets(index.requested_fact_id,semantic_recall_buckets(index))
                                         for index in self.indexes)


def select_semantic_relation_catalog(
    request: SemanticCatalogSelectionRequest,
) -> CatalogSelectionResult:
    if request.max_reads_per_fact < 1 or not request.facts:
        raise ValueError(
            "semantic catalog selection requires facts and a positive limit"
        )
    known_buckets = {
        bucket.bucket_ref
        for fact in request.facts
        for bucket in fact.buckets
    }
    matches = {item.bucket_ref: item for item in request.resource_matches}
    if len(matches) != len(request.resource_matches) or set(matches) != known_buckets:
        raise ValueError(
            "semantic resource matches must cover every recall bucket once"
        )
    read_facts = _catalog_facts_by_read(request.relation_catalog)
    selections = tuple(
        _select_semantic_fact(
            fact,
            request=request,
            matches=matches,
            read_facts=read_facts,
        )
        for fact in request.facts
    )
    selected_ids = selected_read_ids_from_fact_selections(selections)
    return CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            request.relation_catalog,
            read_ids=selected_ids,
        ),
        requested_fact_selections=selections,
        selected_read_ids=selected_ids,
    )


def _select_semantic_fact(
    fact: FactRecallBuckets,
    *,
    request: SemanticCatalogSelectionRequest,
    matches: dict[str, RecallBucketMatch],
    read_facts: dict[str, tuple[CatalogFact, ...]],
) -> RequestedFactCatalogSelection:
    buckets = fact.buckets
    recall_selections = tuple(
        resource_name_selection(
            resource_names=matches[bucket.bucket_ref].exhaustive_resource_names,
            relation_catalog=request.relation_catalog,
            read_facts=read_facts,
        )
        for bucket in buckets
    )
    selected = merge_selected_rankings(
        recall_selections,
        candidate_limit=catalog_candidate_limit(request.max_reads_per_fact),
    )
    positive = merge_positive_rankings(
        tuple(selection.positive_rankings for selection in recall_selections)
    )
    resource_names = tuple(
        dict.fromkeys(
            resource_name
            for bucket in buckets
            for resource_name in matches[bucket.bucket_ref].exhaustive_resource_names
        )
    )
    return RequestedFactCatalogSelection(
        requested_fact_id=fact.requested_fact_id,
        query_terms=_explicit_catalog_search_query_terms(resource_names),
        rankings=selected,
        selected_read_ids=tuple(item.read_id for item in selected),
        unselected_positive_read_ids=unselected_positive_read_ids(
            positive,
            selected_rankings=selected,
        ),
    )


__all__ = ["SemanticCatalogSelectionRequest", "select_semantic_relation_catalog"]
