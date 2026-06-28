"""Catalog selection term extraction."""

from __future__ import annotations

import json

from fervis.lookup.relation_catalog import (
    CatalogFact,
    CatalogField,
    IdentityMetadata,
    RelationCatalog,
)

from .constants import (
    _ENGLISH_STOPWORDS,
    _RESOLVER_ENDPOINT_STOPWORDS,
    _TOKEN_RE,
)


def _resolver_query_term_weights(
    catalog_search_terms: tuple[str, ...],
) -> dict[str, int]:
    terms = _ordered_terms(
        catalog_search_terms,
        stopwords=_RESOLVER_ENDPOINT_STOPWORDS,
    )
    return {term: len(terms) - index for index, term in enumerate(terms)}


def _resolver_endpoint_name_terms(endpoint_name: str) -> tuple[str, ...]:
    return _ordered_terms((endpoint_name,), stopwords=_RESOLVER_ENDPOINT_STOPWORDS)


def _explicit_catalog_search_query_terms(
    catalog_search_terms: tuple[str, ...],
) -> tuple[str, ...]:
    return _ordered_terms(catalog_search_terms, stopwords=frozenset())


def _catalog_facts_by_read(
    catalog: RelationCatalog,
) -> dict[str, tuple[CatalogFact, ...]]:
    facts_by_read: dict[str, list[CatalogFact]] = {}
    for fact in catalog.facts:
        if not fact.read_id:
            continue
        facts_by_read.setdefault(fact.read_id, []).append(fact)
    return {read_id: tuple(facts) for read_id, facts in facts_by_read.items()}


def _field_parts(field: CatalogField) -> tuple[object, ...]:
    return (
        field.ref,
        field.path,
        field.row_path_id,
        field.choices,
        field.metadata or {},
        _identity_parts(field.identity),
    )


def _fact_parts(fact: CatalogFact) -> tuple[object, ...]:
    return (
        fact.ref,
        fact.availability.value,
        fact.field_ref,
        fact.read_id,
        fact.proof_refs,
    )


def _identity_parts(identity: IdentityMetadata | None) -> tuple[object, ...]:
    if identity is None:
        return ()
    return (
        identity.entity_ref,
        identity.identity_field,
        identity.display_fields,
    )


def _ordered_terms(
    items: tuple[object, ...],
    *,
    stopwords: frozenset[str] = _ENGLISH_STOPWORDS,
) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        for token in _tokens(item, stopwords=stopwords):
            if token in seen:
                continue
            seen.add(token)
            output.append(token)
    return tuple(output)


def _tokens(
    value: object, *, stopwords: frozenset[str] = _ENGLISH_STOPWORDS
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value
    else:
        raw = json.dumps(value, sort_keys=True, default=str)
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(raw.lower()):
        token = match.group(0)
        if token.isdigit():
            continue
        if token in stopwords:
            continue
        for variant in _token_variants(token):
            if variant not in stopwords:
                tokens.append(variant)
    return tuple(tokens)


def _token_variants(token: str) -> tuple[str, ...]:
    if len(token) < 2:
        return ()
    if token.endswith("ies") and len(token) > 4:
        return (f"{token[:-3]}y",)
    if token.endswith(("ches", "shes", "sses", "xes", "zes")) and len(token) > 4:
        return (token[:-2],)
    if (
        token.endswith("s")
        and len(token) > 3
        and not token.endswith(("ss", "us", "is"))
    ):
        return (token[:-1],)
    return (token,)
