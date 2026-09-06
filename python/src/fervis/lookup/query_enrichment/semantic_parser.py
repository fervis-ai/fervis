"""Parse the semantic Query Enrichment result mechanically."""

from __future__ import annotations

from fervis.lookup.query_enrichment import semantic_provider_contract as output
from fervis.lookup.query_enrichment.semantic import (
    InputResourceSearchTerms,
    SemanticQueryEnrichmentRequest,
    SemanticQueryEnrichmentResult,
    RecallBucketMatch,
    validate_semantic_query_enrichment_result,
)


def parse_semantic_query_enrichment(
    payload: dict[str, object],
    *,
    request: SemanticQueryEnrichmentRequest,
) -> SemanticQueryEnrichmentResult:
    parsed = output.SemanticQueryEnrichmentOutput.parse(payload)
    result = SemanticQueryEnrichmentResult(
        recall_bucket_matches=tuple(
            RecallBucketMatch(
                bucket_ref=item.bucket_ref,
                exhaustive_resource_names=tuple(
                    dict.fromkeys(
                        (
                            *item.exhaustive_resource_names,
                            *item.matching_resource_names,
                        )
                    )
                ),
                matching_resource_names=tuple(
                    dict.fromkeys(item.matching_resource_names)
                ),
            )
            for item in parsed.recall_bucket_matches
        ),
        input_resource_search_terms=tuple(
            InputResourceSearchTerms(
                input_use_ref=item.input_use_ref,
                catalog_search_terms=tuple(dict.fromkeys(item.catalog_search_terms)),
            )
            for item in parsed.input_resource_search_terms
        ),
    )
    return validate_semantic_query_enrichment_result(
        result,
        recall_buckets=request.recall_buckets,
        reference_tasks=request.reference_tasks,
        resource_names=request.resource_names,
    )


__all__ = ["parse_semantic_query_enrichment"]
