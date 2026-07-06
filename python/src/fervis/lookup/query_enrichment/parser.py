"""Parse and validate catalog query-enrichment output."""

from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog.selection import (
    AnswerOutputResourceLineage,
    EntityTargetCatalogSearchTerms,
    RequestedFactResourceNameMatches,
)
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.query_enrichment.model import (
    QUERY_ENRICHMENT_MAX_CATALOG_SEARCH_TERMS,
    QueryEnrichmentRequest,
    QueryEnrichmentResult,
    query_enrichment_endpoint_names,
    query_enrichment_resource_names,
)
from fervis.lookup.query_enrichment import provider_contract as provider_output


def parse_query_enrichment(
    payload: dict[str, Any],
    *,
    request: QueryEnrichmentRequest,
) -> QueryEnrichmentResult:
    parsed = provider_output.QueryEnrichmentOutput.parse(payload)
    raw_items = parsed.requested_fact_resource_name_matches
    if not isinstance(raw_items, list):
        raise ValueError("requested_fact_resource_name_matches must be an array")
    requested_fact_ids = {fact.id for fact in request.requested_facts}
    endpoint_names = set(query_enrichment_endpoint_names(request))
    resource_names = set(query_enrichment_resource_names(request))
    output: list[RequestedFactResourceNameMatches] = []
    seen_facts: set[str] = set()
    for raw in raw_items:
        item = provider_output.RequestedFactResourceNameMatchesOutput.parse(raw)
        requested_fact_id = _text(item.requested_fact_id)
        if requested_fact_id not in requested_fact_ids:
            raise ValueError("query enrichment references unknown requested fact")
        if requested_fact_id in seen_facts:
            raise ValueError("duplicate resource name matches for requested fact")
        raw_matches = item.answer_output_resource_lineage
        if not isinstance(raw_matches, list):
            raise ValueError("answer_output_resource_lineage must be an array")
        matches = _answer_output_resource_lineage(
            raw_matches,
            answer_output_ids={
                output.id
                for fact in request.requested_facts
                if fact.id == requested_fact_id
                for output in fact.answer_outputs
            },
            endpoint_names=endpoint_names,
            resource_names=resource_names,
        )
        output.append(
            RequestedFactResourceNameMatches(
                requested_fact_id=requested_fact_id,
                answer_output_resource_lineage=matches,
            )
        )
        seen_facts.add(requested_fact_id)
    missing = requested_fact_ids - seen_facts
    if missing:
        raise ValueError("query enrichment missing requested fact")
    entity_terms = _entity_target_catalog_search_terms(
        parsed.entity_target_catalog_search_terms,
        request=request,
    )
    return QueryEnrichmentResult(
        requested_fact_resource_name_matches=tuple(output),
        entity_target_catalog_search_terms=entity_terms,
    )


def _answer_output_resource_lineage(
    raw_matches: list[Any],
    *,
    answer_output_ids: set[str],
    endpoint_names: set[str],
    resource_names: set[str],
) -> tuple[AnswerOutputResourceLineage, ...]:
    rows: dict[tuple[str, str, str], list[str]] = {}
    for raw in raw_matches:
        item = provider_output.AnswerOutputResourceLineageOutput.parse(raw)
        answer_output_id = _text(item.answer_output_id)
        if answer_output_id not in answer_output_ids:
            raise ValueError("query enrichment references unknown answer output")
        support_role = _text(item.support_role)
        if support_role not in ANSWER_OUTPUT_SUPPORT_ROLE_VALUES:
            raise ValueError("query enrichment references unknown support role")
        source_text = _text(item.source_text)
        raw_terms = item.matching_resource_names
        if not isinstance(raw_terms, list):
            raise ValueError("matching_resource_names must be an array")
        terms = _matching_resource_names(
            raw_terms,
            endpoint_names=endpoint_names,
            resource_names=resource_names,
        )
        if not terms:
            raise ValueError(
                "answer_output_resource_lineage requires matching_resource_names"
            )
        row_key = (answer_output_id, support_role, source_text)
        row_terms = rows.setdefault(row_key, [])
        row_terms.extend(term for term in terms if term not in row_terms)
    return tuple(
        AnswerOutputResourceLineage(
            answer_output_id=answer_output_id,
            support_role=support_role,
            source_text=source_text,
            matching_resource_names=tuple(terms),
        )
        for (answer_output_id, support_role, source_text), terms in rows.items()
    )


def _entity_target_catalog_search_terms(
    raw_items: object,
    *,
    request: QueryEnrichmentRequest,
) -> tuple[EntityTargetCatalogSearchTerms, ...]:
    if not isinstance(raw_items, list):
        raise ValueError("entity_target_catalog_search_terms must be an array")
    entity_target_ids = {
        known.id
        for fact in request.requested_facts
        for known in fact.known_inputs
        if known.is_reference_value
    }
    endpoint_names = set(query_enrichment_endpoint_names(request))
    resource_names = set(query_enrichment_resource_names(request))
    output: list[EntityTargetCatalogSearchTerms] = []
    seen_targets: set[str] = set()
    for raw in raw_items:
        item = provider_output.EntityTargetCatalogSearchTermsOutput.parse(raw)
        target_id = _text(item.target_id)
        if target_id not in entity_target_ids:
            raise ValueError("query enrichment references unknown entity target")
        if target_id in seen_targets:
            raise ValueError("duplicate entity target catalog search terms")
        raw_terms = item.catalog_search_terms
        if not isinstance(raw_terms, list):
            raise ValueError("catalog_search_terms must be an array")
        terms = _entity_catalog_search_terms(
            raw_terms,
            endpoint_names=endpoint_names,
            resource_names=resource_names,
            max_terms=QUERY_ENRICHMENT_MAX_CATALOG_SEARCH_TERMS,
        )
        output.append(
            EntityTargetCatalogSearchTerms(
                target_id=target_id,
                catalog_search_terms=terms,
            )
        )
        seen_targets.add(target_id)
    missing = entity_target_ids - seen_targets
    if missing:
        raise ValueError("query enrichment missing entity target")
    return tuple(output)


def _entity_catalog_search_terms(
    raw_terms: list[Any],
    *,
    endpoint_names: set[str],
    resource_names: set[str],
    max_terms: int,
) -> tuple[str, ...]:
    if len(raw_terms) > max_terms:
        raise ValueError(f"catalog_search_terms must contain at most {max_terms} terms")
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in raw_terms:
        if not isinstance(raw_term, dict):
            raise ValueError("entity catalog_search_terms items must be objects")
        item = provider_output.CatalogSearchTermOutput.parse(raw_term)
        _text(item.basis)
        term = _text(item.term)
        if term in endpoint_names:
            raise ValueError("catalog_search_terms must not contain endpoint names")
        if term not in resource_names:
            raise ValueError("catalog_search_terms must be resource names")
        if term not in seen:
            terms.append(term)
        seen.add(term)
    return tuple(terms)


def _matching_resource_names(
    raw_terms: list[Any],
    *,
    endpoint_names: set[str],
    resource_names: set[str],
) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in raw_terms:
        term = _text(raw_term)
        if term in endpoint_names:
            raise ValueError("matching_resource_names must not contain endpoint names")
        if term not in resource_names:
            raise ValueError("matching_resource_names must be resource names")
        if term not in seen:
            terms.append(term)
        seen.add(term)
    return tuple(terms)


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("query enrichment text values must be strings")
    text = value.strip()
    if not text:
        raise ValueError("query enrichment requires non-empty text")
    return text
