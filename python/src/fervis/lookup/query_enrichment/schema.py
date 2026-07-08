"""Provider schema for catalog query enrichment."""

from __future__ import annotations

from fervis.lookup.query_enrichment import provider_contract as provider_output
from fervis.lookup.query_enrichment.model import (
    QUERY_ENRICHMENT_MAX_CATALOG_SEARCH_TERMS,
)
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)


def build_query_enrichment_schema(
    *,
    resource_names: tuple[str, ...] = (),
) -> dict[str, object]:
    max_terms = QUERY_ENRICHMENT_MAX_CATALOG_SEARCH_TERMS if resource_names else 0
    term_schema: dict[str, object] = (
        {"enum": list(resource_names)}
        if resource_names
        else {"type": "string", "minLength": 1}
    )
    matching_resource_names_schema: dict[str, object] = {
        "type": "array",
        "items": term_schema,
    }
    if resource_names:
        matching_resource_names_schema["minItems"] = 1
    answer_output_resource_lineage_schema: dict[str, object] = {
        "type": "array",
        "items": provider_output.AnswerOutputResourceLineageOutput.schema(
            {
                "answer_output_id": {"type": "string", "minLength": 1},
                "support_role": {"enum": list(ANSWER_OUTPUT_SUPPORT_ROLE_VALUES)},
                "source_text": {"type": "string", "minLength": 1},
                "matching_resource_names": matching_resource_names_schema,
            },
        ),
    }
    if not resource_names:
        answer_output_resource_lineage_schema["maxItems"] = 0
    return provider_output.QueryEnrichmentOutput.schema(
        {
            "requested_fact_resource_name_matches": {
                "type": "array",
                "items": provider_output.RequestedFactResourceNameMatchesOutput.schema(
                    {
                        "requested_fact_id": {"type": "string", "minLength": 1},
                        "answer_output_resource_lineage": (
                            answer_output_resource_lineage_schema
                        ),
                    },
                ),
            },
            "entity_target_catalog_search_terms": {
                "type": "array",
                "items": provider_output.EntityTargetCatalogSearchTermsOutput.schema(
                    {
                        "target_id": {"type": "string", "minLength": 1},
                        "catalog_search_terms": {
                            "type": "array",
                            "maxItems": max_terms,
                            "items": provider_output.CatalogSearchTermOutput.schema(
                                {
                                    "basis": {"type": "string", "minLength": 1},
                                    "term": term_schema,
                                },
                            ),
                        },
                    },
                ),
            },
        },
    )
