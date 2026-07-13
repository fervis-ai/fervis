"""Requested-fact catalog selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from fervis.lookup.relation_catalog import (
    CatalogFact,
    CatalogField,
    EndpointRead,
    EntityKeyComponentTarget,
    ParamSource,
)
from fervis.lookup.relation_catalog.selection.model import (
    CatalogSelectionRanking,
    CatalogSelectionRequest,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
    RequestedFactResourceNameMatches,
)
from fervis.lookup.relation_catalog.selection.results import (
    relation_catalog_for_read_ids,
    selected_read_ids_from_fact_selections,
)
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
)

from .constants import (
    _CATALOG_TERM_SCORE,
    _RESOURCE_TERM_SCORE,
    MIN_CATALOG_READS_PER_FACT,
)

from .terms import (
    _catalog_facts_by_read,
    _fact_parts,
    _field_parts,
    _explicit_catalog_search_query_terms,
    _ordered_terms,
)

def select_relation_catalog(
    request: CatalogSelectionRequest,
) -> CatalogSelectionResult:
    if request.max_reads_per_fact < 1:
        raise ValueError("catalog selection requires positive max_reads_per_fact")
    if not request.requested_facts:
        raise ValueError("catalog selection requires requested facts")

    read_facts = _catalog_facts_by_read(request.relation_catalog)
    resource_matches_by_fact = _resource_name_matches_by_fact(request)
    fact_selections = tuple(
        _select_for_requested_fact(
            fact,
            request=request,
            read_facts=read_facts,
            resource_name_matches=resource_matches_by_fact[fact.id],
        )
        for fact in request.requested_facts
    )
    selected_ids = selected_read_ids_from_fact_selections(fact_selections)
    return CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            request.relation_catalog,
            read_ids=selected_ids,
        ),
        requested_fact_selections=fact_selections,
        selected_read_ids=selected_ids,
    )


def _select_for_requested_fact(
    fact: RequestedFact,
    *,
    request: CatalogSelectionRequest,
    read_facts: dict[str, tuple[CatalogFact, ...]],
    resource_name_matches: RequestedFactResourceNameMatches,
) -> RequestedFactCatalogSelection:
    candidate_limit = _candidate_limit(request.max_reads_per_fact)
    source_text_selections = _source_text_selections(
        fact,
        request=request,
        read_facts=read_facts,
        resource_name_matches=resource_name_matches,
    )
    selected_rankings = _merge_selected_rankings(
        source_text_selections,
        candidate_limit=candidate_limit,
    )
    positive_rankings = _merge_positive_rankings(
        tuple(selection.positive_rankings for selection in source_text_selections)
    )
    query_terms = _merge_ordered_query_terms(
        tuple(selection.query_terms for selection in source_text_selections)
    )
    return RequestedFactCatalogSelection(
        requested_fact_id=fact.id,
        query_terms=query_terms,
        rankings=selected_rankings,
        selected_read_ids=tuple(item.read_id for item in selected_rankings),
        unselected_positive_read_ids=_unselected_positive_read_ids(
            positive_rankings,
            selected_rankings=selected_rankings,
        ),
    )


def _candidate_limit(max_reads_per_fact: int) -> int:
    return max(max_reads_per_fact, MIN_CATALOG_READS_PER_FACT)


def _resource_name_matches_by_fact(
    request: CatalogSelectionRequest,
) -> dict[str, RequestedFactResourceNameMatches]:
    requested_fact_ids = {fact.id for fact in request.requested_facts}
    output: dict[str, RequestedFactResourceNameMatches] = {}
    for item in request.resource_name_matches:
        if item.requested_fact_id not in requested_fact_ids:
            raise ValueError("resource name matches reference unknown requested fact")
        if item.requested_fact_id in output:
            raise ValueError("duplicate resource name matches for requested fact")
        output[item.requested_fact_id] = item
    missing = requested_fact_ids - set(output)
    if missing:
        raise ValueError("resource name matches missing requested fact")
    return output


@dataclass(frozen=True)
class _SourceTextSelection:
    query_terms: tuple[str, ...]
    positive_rankings: tuple[CatalogSelectionRanking, ...]
    exact_rankings: tuple[CatalogSelectionRanking, ...]


@dataclass(frozen=True)
class _ResourceNameRanking:
    is_exact_resource_name: bool
    ranking: CatalogSelectionRanking


def _source_text_selections(
    fact: RequestedFact,
    *,
    request: CatalogSelectionRequest,
    read_facts: dict[str, tuple[CatalogFact, ...]],
    resource_name_matches: RequestedFactResourceNameMatches,
) -> tuple[_SourceTextSelection, ...]:
    resource_name_requests = _resource_name_requests(resource_name_matches)
    return tuple(
        _rank_for_source_text(
            resource_name_request,
            request=request,
            read_facts=read_facts,
        )
        for resource_name_request in resource_name_requests
    )


@dataclass(frozen=True)
class _ResourceNameRequest:
    answer_output_id: str
    support_role: str
    source_text: str
    resource_names: tuple[str, ...]


def _resource_name_requests(
    resource_name_matches: RequestedFactResourceNameMatches,
) -> tuple[_ResourceNameRequest, ...]:
    output: list[_ResourceNameRequest] = []
    for item in resource_name_matches.answer_output_resource_lineage:
        resource_names: list[str] = []
        seen_in_item: set[str] = set()
        for resource_name in item.matching_resource_names:
            if resource_name in seen_in_item:
                continue
            seen_in_item.add(resource_name)
            resource_names.append(resource_name)
        if resource_names:
            output.append(
                _ResourceNameRequest(
                    answer_output_id=item.answer_output_id,
                    support_role=item.support_role,
                    source_text=item.source_text,
                    resource_names=tuple(resource_names),
                ),
            )
    return tuple(output)


def _rank_for_source_text(
    resource_name_request: _ResourceNameRequest,
    *,
    request: CatalogSelectionRequest,
    read_facts: dict[str, tuple[CatalogFact, ...]],
) -> _SourceTextSelection:
    query_terms = _explicit_catalog_search_query_terms(
        resource_name_request.resource_names,
    )
    reads_by_id = {read.id: read for read in request.relation_catalog.reads}
    positive_groups = tuple(
        _positive_rankings(
            _rankings_for_resource_name(
                resource_name,
                request=request,
                read_facts=read_facts,
            ),
            reads_by_id=reads_by_id,
            available_values=request.available_values,
        )
        for resource_name in resource_name_request.resource_names
    )
    exact_rankings = _round_robin_rankings(
        tuple(
            tuple(item.ranking for item in group if item.is_exact_resource_name)
            for group in positive_groups
        )
    )
    positive_rankings = _round_robin_rankings(
        tuple(tuple(item.ranking for item in group) for group in positive_groups)
    )
    return _SourceTextSelection(
        query_terms=query_terms,
        positive_rankings=positive_rankings,
        exact_rankings=exact_rankings,
    )


def _merge_exact_rankings(
    selections: tuple[_SourceTextSelection, ...],
    *,
    candidate_limit: int,
) -> tuple[CatalogSelectionRanking, ...]:
    return _round_robin_unique_rankings(
        tuple(selection.exact_rankings for selection in selections),
        limit=candidate_limit,
    )


def _remaining_positive_rankings(
    selections: tuple[_SourceTextSelection, ...],
    *,
    selected_read_ids: set[str],
) -> tuple[tuple[CatalogSelectionRanking, ...], ...]:
    return tuple(
        tuple(
            ranking
            for ranking in selection.positive_rankings
            if ranking.read_id not in selected_read_ids
        )
        for selection in selections
    )


def _merge_selected_rankings(
    selections: tuple[_SourceTextSelection, ...],
    *,
    candidate_limit: int,
) -> tuple[CatalogSelectionRanking, ...]:
    exact_rankings = _merge_exact_rankings(
        selections,
        candidate_limit=candidate_limit,
    )
    remaining_limit = max(candidate_limit - len(exact_rankings), 0)
    if remaining_limit == 0:
        return exact_rankings
    selected_read_ids = {ranking.read_id for ranking in exact_rankings}
    remaining_rankings = _round_robin_unique_rankings(
        _remaining_positive_rankings(
            selections,
            selected_read_ids=selected_read_ids,
        ),
        limit=remaining_limit,
    )
    return (*exact_rankings, *remaining_rankings)


def _rankings_for_resource_name(
    resource_name: str,
    *,
    request: CatalogSelectionRequest,
    read_facts: dict[str, tuple[CatalogFact, ...]],
) -> tuple[_ResourceNameRanking, ...]:
    ranked_matches: list[_ResourceNameRanking] = []
    for read in request.relation_catalog.reads:
        match = _resource_name_match(
            requested_resource_name=resource_name,
            read_resource_names=read.resource_names,
        )
        if match is None:
            ranking = _catalog_evidence_ranking(
                read,
                requested_resource_name=resource_name,
                read_facts=read_facts.get(read.id, ()),
            )
            if ranking is not None:
                ranked_matches.append(
                    _ResourceNameRanking(
                        is_exact_resource_name=False,
                        ranking=ranking,
                    )
                )
            continue
        ranked_matches.append(
            _ResourceNameRanking(
                is_exact_resource_name=match.is_exact_resource_name,
                ranking=_resource_name_match_ranking(
                    _rank_resource_read(
                        read,
                        match.query_terms,
                        read_facts=read_facts.get(read.id, ()),
                    ),
                    match=match,
                ),
            )
        )
    return tuple(
        item
        for item in sorted(
            ranked_matches,
            key=lambda item: (
                not item.is_exact_resource_name,
                -item.ranking.score,
                item.ranking.read_id,
            ),
        )
    )


def _catalog_evidence_ranking(
    read: EndpointRead,
    *,
    requested_resource_name: str,
    read_facts: tuple[CatalogFact, ...],
) -> CatalogSelectionRanking | None:
    query_terms = _explicit_catalog_search_query_terms((requested_resource_name,))
    ranking = _rank_resource_read(read, query_terms, read_facts=read_facts)
    evidence_count = len(ranking.matched_fact_refs) + len(ranking.matched_field_refs)
    if evidence_count == 0:
        return None
    return replace(ranking, score=_CATALOG_TERM_SCORE * evidence_count)


def _round_robin_rankings(
    ranking_groups: tuple[tuple[CatalogSelectionRanking, ...], ...],
) -> tuple[CatalogSelectionRanking, ...]:
    return _round_robin_unique_rankings(ranking_groups)


def _round_robin_unique_rankings(
    ranking_groups: tuple[tuple[CatalogSelectionRanking, ...], ...],
    *,
    limit: int | None = None,
) -> tuple[CatalogSelectionRanking, ...]:
    selected: list[CatalogSelectionRanking] = []
    seen: set[str] = set()
    max_rankings = max((len(group) for group in ranking_groups), default=0)
    for index in range(max_rankings):
        for group in ranking_groups:
            if index >= len(group):
                continue
            ranking = group[index]
            if ranking.read_id in seen:
                continue
            seen.add(ranking.read_id)
            selected.append(ranking)
            if limit is not None and len(selected) >= limit:
                return tuple(selected)
    return tuple(selected)


def _resource_name_match_ranking(
    ranking: CatalogSelectionRanking,
    *,
    match: _ResourceNameMatch,
) -> CatalogSelectionRanking:
    if match.is_exact_resource_name:
        return ranking
    resource_score = _RESOURCE_TERM_SCORE * max(1, len(match.query_terms))
    return replace(ranking, score=max(1, ranking.score - resource_score))


def _merge_positive_rankings(
    rankings_by_source_text: tuple[tuple[CatalogSelectionRanking, ...], ...],
) -> tuple[CatalogSelectionRanking, ...]:
    return _dedupe_rankings(
        ranking for rankings in rankings_by_source_text for ranking in rankings
    )


def _dedupe_rankings(
    rankings: Iterable[CatalogSelectionRanking],
) -> tuple[CatalogSelectionRanking, ...]:
    seen: set[str] = set()
    output: list[CatalogSelectionRanking] = []
    for ranking in rankings:
        if ranking.read_id in seen:
            continue
        seen.add(ranking.read_id)
        output.append(ranking)
    return tuple(output)


def _merge_ordered_query_terms(
    term_groups: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for terms in term_groups:
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            output.append(term)
    return tuple(output)


def _unselected_positive_read_ids(
    positive_rankings: tuple[CatalogSelectionRanking, ...],
    *,
    selected_rankings: tuple[CatalogSelectionRanking, ...],
) -> tuple[str, ...]:
    selected_ids = {ranking.read_id for ranking in selected_rankings}
    return tuple(
        ranking.read_id
        for ranking in positive_rankings
        if ranking.read_id not in selected_ids
    )


def _positive_rankings(
    rankings: tuple[_ResourceNameRanking, ...],
    *,
    reads_by_id: dict[str, EndpointRead],
    available_values: tuple[FactValue, ...] = (),
) -> tuple[_ResourceNameRanking, ...]:
    positive = tuple(item for item in rankings if item.ranking.score > 0)
    positive = _drop_unbound_required_reads_with_open_alternatives(
        positive,
        reads_by_id=reads_by_id,
        available_values=available_values,
    )
    return positive


def _drop_unbound_required_reads_with_open_alternatives(
    rankings: tuple[_ResourceNameRanking, ...],
    *,
    reads_by_id: dict[str, EndpointRead],
    available_values: tuple[FactValue, ...] = (),
) -> tuple[_ResourceNameRanking, ...]:
    open_rankings = tuple(
        ranking
        for ranking in rankings
        if not _requires_unbound_required_input(
            reads_by_id[ranking.ranking.read_id],
            available_values=available_values,
        )
    )
    if not open_rankings:
        return rankings
    return open_rankings


def _requires_unbound_required_input(
    read: EndpointRead,
    *,
    available_values: tuple[FactValue, ...] = (),
) -> bool:
    for param in read.params:
        if not param.required or param.default is not None:
            continue
        if param.entity_target is not None and _has_matching_entity_value(
            param.entity_target,
            available_values,
        ):
            continue
        if param.source == ParamSource.PATH or param.entity_target is not None:
            return True
    return False


def _has_matching_entity_value(
    target: EntityKeyComponentTarget,
    values: tuple[FactValue, ...],
) -> bool:
    for value in values:
        payload = value.payload
        if not isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload)):
            continue
        if (
            payload.entity_kind == target.entity_kind
            and payload.key_id == target.key_id
            and target.component_id in _identity_component_ids(payload)
        ):
            return True
    return False


def _identity_component_ids(
    payload: IdentityValuePayload | IdentitySetValuePayload,
) -> frozenset[str]:
    key = payload.key if isinstance(payload, IdentityValuePayload) else payload.keys[0]
    return frozenset(component.component_id for component in key.components)


@dataclass(frozen=True)
class _ResourceNameMatch:
    query_terms: tuple[str, ...]
    is_exact_resource_name: bool = False


def _resource_name_match(
    *,
    requested_resource_name: str,
    read_resource_names: tuple[str, ...],
) -> _ResourceNameMatch | None:
    is_exact_resource_name = False
    matched_names: list[str] = []
    requested_terms = frozenset(
        _explicit_catalog_search_query_terms((requested_resource_name,))
    )
    if not requested_terms:
        return None
    for read_resource_name in read_resource_names:
        read_terms = frozenset(
            _explicit_catalog_search_query_terms((read_resource_name,))
        )
        if requested_resource_name == read_resource_name:
            is_exact_resource_name = True
            matched_names.append(requested_resource_name)
            break
        if _requested_terms_match_read_resource(requested_terms, read_terms):
            matched_names.append(requested_resource_name)
            break
    query_terms = _explicit_catalog_search_query_terms(tuple(matched_names))
    if not query_terms:
        return None
    return _ResourceNameMatch(
        query_terms=query_terms,
        is_exact_resource_name=is_exact_resource_name,
    )


def _requested_terms_match_read_resource(
    requested_terms: frozenset[str],
    read_terms: frozenset[str],
) -> bool:
    return bool(requested_terms and read_terms and requested_terms < read_terms)


def _rank_resource_read(
    read: EndpointRead,
    query_terms: tuple[str, ...],
    *,
    read_facts: tuple[CatalogFact, ...] = (),
) -> CatalogSelectionRanking:
    matched_fact_refs = _matched_catalog_fact_refs(
        (*read.facts, *read_facts),
        query_terms=query_terms,
    )
    matched_field_refs = _matched_catalog_field_refs(
        read.fields,
        query_terms=query_terms,
    )
    score = _RESOURCE_TERM_SCORE * max(1, len(query_terms))
    score += _CATALOG_TERM_SCORE * len(matched_fact_refs)
    score += _CATALOG_TERM_SCORE * len(matched_field_refs)
    return CatalogSelectionRanking(
        read_id=read.id,
        score=score,
        matched_terms=query_terms,
        matched_fact_refs=matched_fact_refs,
        matched_field_refs=matched_field_refs,
    )


def _matched_catalog_fact_refs(
    facts: tuple[CatalogFact, ...],
    *,
    query_terms: tuple[str, ...],
) -> tuple[str, ...]:
    query = set(query_terms)
    refs: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        if fact.ref in seen:
            continue
        if query.intersection(_ordered_terms(_fact_parts(fact))):
            refs.append(fact.ref)
            seen.add(fact.ref)
    return tuple(refs)


def _matched_catalog_field_refs(
    fields: tuple[CatalogField, ...],
    *,
    query_terms: tuple[str, ...],
) -> tuple[str, ...]:
    query = set(query_terms)
    refs: list[str] = []
    for field in fields:
        if query.intersection(_ordered_terms(_field_parts(field))):
            refs.append(field.ref)
    return tuple(refs)
