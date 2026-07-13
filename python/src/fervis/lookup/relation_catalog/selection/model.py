"""Typed catalog-selection contract for Lookup planning."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.answer_program.values import FactValue


@dataclass(frozen=True)
class AnswerOutputResourceLineage:
    answer_output_id: str
    support_role: str
    source_text: str
    matching_resource_names: tuple[str, ...]


@dataclass(frozen=True)
class RequestedFactResourceNameMatches:
    requested_fact_id: str
    answer_output_resource_lineage: tuple[AnswerOutputResourceLineage, ...]


@dataclass(frozen=True)
class EntityTargetCatalogSearchTerms:
    target_id: str
    catalog_search_terms: tuple[str, ...]


@dataclass(frozen=True)
class CatalogSelectionRequest:
    relation_catalog: RelationCatalog
    requested_facts: tuple[RequestedFact, ...]
    max_reads_per_fact: int
    resource_name_matches: tuple[RequestedFactResourceNameMatches, ...]
    available_values: tuple[FactValue, ...] = ()


@dataclass(frozen=True)
class CatalogSelectionRanking:
    read_id: str
    score: int
    matched_terms: tuple[str, ...] = ()
    matched_fact_refs: tuple[str, ...] = ()
    matched_field_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestedFactCatalogSelection:
    requested_fact_id: str
    query_terms: tuple[str, ...]
    rankings: tuple[CatalogSelectionRanking, ...]
    selected_read_ids: tuple[str, ...]
    unselected_positive_read_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogSelectionResult:
    relation_catalog: RelationCatalog
    requested_fact_selections: tuple[RequestedFactCatalogSelection, ...]
    selected_read_ids: tuple[str, ...]


@dataclass(frozen=True)
class EntityTargetResolverSelection:
    target_id: str
    catalog_search_terms: tuple[str, ...]
    selected_read_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolverCatalogSelectionRequest:
    relation_catalog: RelationCatalog
    entity_target_catalog_search_terms: tuple[EntityTargetCatalogSearchTerms, ...]
    max_reads_per_target: int = 3


@dataclass(frozen=True)
class ResolverCatalogSelectionResult:
    relation_catalog: RelationCatalog
    entity_target_selections: tuple[EntityTargetResolverSelection, ...]
    selected_read_ids: tuple[str, ...]


def catalog_selection_evidence_ref(*, requested_fact_id: str) -> str:
    return f"catalog_selection:{requested_fact_id}"
