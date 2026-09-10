"""Resolver catalog selection."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog import (
    EndpointRead,
    RelationCatalog,
    primary_stable_key_entity_kinds,
)

from .constants import _RESOLVER_ENDPOINT_STOPWORDS
from .terms import (
    _ordered_terms,
    _resolver_endpoint_name_terms,
    _resolver_query_term_weights,
)


def select_resolver_reads(
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
    return _resolver_selection_with_primary_class_routes(
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


def _resolver_selection_with_primary_class_routes(
    ranked: list[_ResolverReadRanking],
    *,
    terms: tuple[str, ...],
    limit: int,
) -> tuple[EndpointRead, ...]:
    selected: list[_ResolverReadRanking] = []
    selected_ids: set[str] = set()
    for term_index, term in enumerate(terms):
        candidates = sorted(
            (
                item
                for item in ranked
                if term in item.matched_terms and item.read.id not in selected_ids
            ),
            key=lambda item: _resolver_term_coverage_key(item, term=term),
        )
        term_limit = 2 if term_index == 0 else 1
        for candidate in candidates[:term_limit]:
            selected.append(candidate)
            selected_ids.add(candidate.read.id)
            if len(selected) >= limit:
                return tuple(item.read for item in selected)
    for item in ranked:
        if item.read.id in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return tuple(item.read for item in selected)


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
