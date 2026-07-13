"""Resolver catalog selection."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog import (
    EndpointRead,
    RelationCatalog,
    primary_stable_key_entity_kinds,
    read_has_primary_stable_key,
)
from fervis.lookup.relation_catalog.selection.model import (
    EntityTargetResolverSelection,
    ResolverCatalogSelectionRequest,
    ResolverCatalogSelectionResult,
)

from .constants import _RESOLVER_ENDPOINT_STOPWORDS
from .terms import (
    _ordered_terms,
    _resolver_endpoint_name_terms,
    _resolver_query_term_weights,
)


def select_resolver_relation_catalog(
    request: ResolverCatalogSelectionRequest,
) -> ResolverCatalogSelectionResult:
    selected_read_ids: list[str] = []
    selections: list[EntityTargetResolverSelection] = []
    reads_by_id = {read.id: read for read in request.relation_catalog.reads}
    for item in request.entity_target_catalog_search_terms:
        catalog_search_terms = tuple(term for term in item.catalog_search_terms if term)
        selected = tuple(
            read.id
            for read in _resolver_reads_for_endpoint_terms(
                request.relation_catalog,
                catalog_search_terms=catalog_search_terms,
                limit=request.max_reads_per_target,
            )
        )
        for read_id in selected:
            if read_id not in selected_read_ids:
                selected_read_ids.append(read_id)
        selections.append(
            EntityTargetResolverSelection(
                target_id=item.target_id,
                catalog_search_terms=catalog_search_terms,
                selected_read_ids=selected,
            )
        )
    return ResolverCatalogSelectionResult(
        relation_catalog=RelationCatalog(
            reads=tuple(reads_by_id[read_id] for read_id in selected_read_ids)
        ),
        entity_target_selections=tuple(selections),
        selected_read_ids=tuple(selected_read_ids),
    )


def _resolver_reads_for_endpoint_terms(
    catalog: RelationCatalog,
    *,
    catalog_search_terms: tuple[str, ...],
    limit: int,
) -> tuple[EndpointRead, ...]:
    if not catalog_search_terms or limit <= 0:
        return ()
    term_weights = _resolver_query_term_weights(catalog_search_terms)
    if not term_weights:
        return ()
    ranked: list[_ResolverReadRanking] = []
    order = 0
    for read in catalog.reads:
        order += 1
        if read.method.upper() != "GET":
            continue
        if not _has_stable_identity_field(read):
            continue
        identity_terms = set(_resolver_identity_terms(read))
        identity_score = sum(
            weight for term, weight in term_weights.items() if term in identity_terms
        )
        endpoint_terms = set(_resolver_endpoint_name_terms(read.endpoint_name))
        endpoint_score = sum(
            weight for term, weight in term_weights.items() if term in endpoint_terms
        )
        matched_terms = tuple(
            term
            for term in term_weights
            if term in identity_terms or term in endpoint_terms
        )
        if not matched_terms:
            continue
        ranked.append(
            _ResolverReadRanking(
                read=read,
                matched_terms=matched_terms,
                identity_score=identity_score,
                endpoint_score=endpoint_score,
                order=order,
            )
        )
    ranked.sort(key=_resolver_ranking_key)
    return _resolver_selection_with_term_coverage(
        ranked,
        terms=tuple(term_weights),
        limit=limit,
    )


@dataclass(frozen=True)
class _ResolverReadRanking:
    read: EndpointRead
    matched_terms: tuple[str, ...]
    identity_score: int
    endpoint_score: int
    order: int


def _resolver_ranking_key(item: _ResolverReadRanking) -> tuple[int, int, int, str]:
    return (-item.identity_score, -item.endpoint_score, item.order, item.read.id)


def _resolver_term_coverage_key(
    item: _ResolverReadRanking,
    *,
    term: str,
) -> tuple[int, int, int, int, int, int, str]:
    resource_terms = set(_resolver_exact_resource_terms(item.read))
    identity_terms = set(_resolver_identity_terms(item.read))
    endpoint_terms = set(_resolver_endpoint_name_terms(item.read.endpoint_name))
    return (
        -(1 if term in resource_terms else 0),
        len(resource_terms),
        -(1 if term in identity_terms else 0),
        -(1 if term in endpoint_terms else 0),
        -item.identity_score,
        item.order,
        item.read.id,
    )


def _resolver_selection_with_term_coverage(
    ranked: list[_ResolverReadRanking],
    *,
    terms: tuple[str, ...],
    limit: int,
) -> tuple[EndpointRead, ...]:
    selected: list[_ResolverReadRanking] = []
    selected_ids: set[str] = set()
    for term in terms:
        candidates = sorted(
            (
                item
                for item in ranked
                if term in item.matched_terms and item.read.id not in selected_ids
            ),
            key=lambda item: _resolver_term_coverage_key(item, term=term),
        )
        candidate = candidates[0] if candidates else None
        if candidate is None or candidate.read.id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.read.id)
        if len(selected) >= limit:
            return tuple(item.read for item in selected)
    for item in ranked:
        if item.read.id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.read.id)
        if len(selected) >= limit:
            break
    return tuple(item.read for item in selected)


def _has_stable_identity_field(read: EndpointRead) -> bool:
    return read_has_primary_stable_key(read)


def _resolver_identity_terms(read: EndpointRead) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(read.resource_names)
    values.extend(primary_stable_key_entity_kinds(read))
    for key in read.candidate_keys:
        if key.primary and key.stable:
            values.extend(component.id for component in key.components)
    return _ordered_terms(tuple(values), stopwords=_RESOLVER_ENDPOINT_STOPWORDS)


def _resolver_exact_resource_terms(read: EndpointRead) -> tuple[str, ...]:
    return tuple(
        term
        for term in (
            _resource_name_key(resource_name) for resource_name in read.resource_names
        )
        if term and term not in _RESOLVER_ENDPOINT_STOPWORDS
    )


def _resource_name_key(value: object) -> str:
    terms = _ordered_terms(
        (value,),
        stopwords=_RESOLVER_ENDPOINT_STOPWORDS,
    )
    return "_".join(terms)
