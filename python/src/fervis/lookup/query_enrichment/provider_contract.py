"""Typed provider-output contracts for query enrichment."""

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderOutput


@dataclass(frozen=True)
class CatalogSearchTermOutput(ProviderOutput):
    basis: str
    term: str


@dataclass(frozen=True)
class EntityTargetCatalogSearchTermsOutput(ProviderOutput):
    target_id: str
    catalog_search_terms: tuple[CatalogSearchTermOutput, ...]


@dataclass(frozen=True)
class AnswerOutputResourceLineageOutput(ProviderOutput):
    answer_output_id: str
    support_role: str
    source_text: str
    matching_resource_names: tuple[str, ...]


@dataclass(frozen=True)
class RequestedFactResourceNameMatchesOutput(ProviderOutput):
    requested_fact_id: str
    answer_output_resource_lineage: tuple[AnswerOutputResourceLineageOutput, ...]


@dataclass(frozen=True)
class QueryEnrichmentOutput(ProviderOutput):
    requested_fact_resource_name_matches: tuple[
        RequestedFactResourceNameMatchesOutput, ...
    ]
    entity_target_catalog_search_terms: tuple[EntityTargetCatalogSearchTermsOutput, ...]
