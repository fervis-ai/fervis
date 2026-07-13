"""Parse and validate catalog query-enrichment output."""

from __future__ import annotations

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
    payload: dict[str, object],
    *,
    request: QueryEnrichmentRequest,
) -> QueryEnrichmentResult:
    parsed = provider_output.QueryEnrichmentOutput.parse(payload)
    items = parsed.requested_fact_resource_name_matches
    requested_fact_ids = {fact.id for fact in request.requested_facts}
    endpoint_names = set(query_enrichment_endpoint_names(request))
    resource_names = set(query_enrichment_resource_names(request))
    output: list[RequestedFactResourceNameMatches] = []
    seen_facts: set[str] = set()
    for item in items:
        requested_fact_id = _text(item.requested_fact_id)
        if requested_fact_id not in requested_fact_ids:
            raise ValueError("query enrichment references unknown requested fact")
        if requested_fact_id in seen_facts:
            raise ValueError("duplicate resource name matches for requested fact")
        matches = _answer_output_resource_lineage(
            item.answer_output_resource_lineage,
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
    items: tuple[provider_output.AnswerOutputResourceLineageOutput, ...],
    *,
    answer_output_ids: set[str],
    endpoint_names: set[str],
    resource_names: set[str],
) -> tuple[AnswerOutputResourceLineage, ...]:
    rows: dict[tuple[str, str, str], list[str]] = {}
    for item in items:
        answer_output_id = _text(item.answer_output_id)
        if answer_output_id not in answer_output_ids:
            raise ValueError("query enrichment references unknown answer output")
        support_role = _text(item.support_role)
        if support_role not in ANSWER_OUTPUT_SUPPORT_ROLE_VALUES:
            raise ValueError("query enrichment references unknown support role")
        source_text = _text(item.source_text)
        terms = _matching_resource_names(
            item.matching_resource_names,
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
    items: tuple[provider_output.EntityTargetCatalogSearchTermsOutput, ...],
    *,
    request: QueryEnrichmentRequest,
) -> tuple[EntityTargetCatalogSearchTerms, ...]:
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
    for item in items:
        target_id = _text(item.target_id)
        if target_id not in entity_target_ids:
            raise ValueError("query enrichment references unknown entity target")
        if target_id in seen_targets:
            raise ValueError("duplicate entity target catalog search terms")
        terms = _entity_catalog_search_terms(
            item.catalog_search_terms,
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
    items: tuple[provider_output.CatalogSearchTermOutput, ...],
    *,
    endpoint_names: set[str],
    resource_names: set[str],
    max_terms: int,
) -> tuple[str, ...]:
    if len(items) > max_terms:
        raise ValueError(f"catalog_search_terms must contain at most {max_terms} terms")
    terms: list[str] = []
    seen: set[str] = set()
    for item in items:
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
    raw_terms: tuple[str, ...],
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
