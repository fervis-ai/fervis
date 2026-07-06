"""Provider-output DTOs for query enrichment."""

from __future__ import annotations

from fervis.lookup.provider_contract import provider_output_type


QueryEnrichmentOutput = provider_output_type(
    "QueryEnrichmentOutput",
    ("requested_fact_resource_name_matches", "entity_target_catalog_search_terms"),
)
RequestedFactResourceNameMatchesOutput = provider_output_type(
    "RequestedFactResourceNameMatchesOutput",
    ("requested_fact_id", "answer_output_resource_lineage"),
)
AnswerOutputResourceLineageOutput = provider_output_type(
    "AnswerOutputResourceLineageOutput",
    ("answer_output_id", "support_role", "source_text", "matching_resource_names"),
)
EntityTargetCatalogSearchTermsOutput = provider_output_type(
    "EntityTargetCatalogSearchTermsOutput",
    ("target_id", "catalog_search_terms"),
)
CatalogSearchTermOutput = provider_output_type(
    "CatalogSearchTermOutput",
    ("basis", "term"),
)
