"""Strict provider schema for semantic Query Enrichment."""

from __future__ import annotations

from fervis.lookup.query_enrichment import semantic_provider_contract as output
from fervis.lookup.query_enrichment.semantic import SemanticQueryEnrichmentRequest


def build_semantic_query_enrichment_schema(
    request: SemanticQueryEnrichmentRequest,
) -> dict[str, object]:
    resource_name_definition = (
        {"enum": list(request.resource_names)}
        if request.resource_names
        else {"type": "string", "maxLength": 0}
    )
    resource_name = {"$ref": "#/$defs/resource_name"}
    schema = output.SemanticQueryEnrichmentOutput.schema(
        {
            "recall_bucket_matches": {
                "type": "array",
                "minItems": len(request.recall_buckets),
                "maxItems": len(request.recall_buckets),
                "items": output.RecallBucketMatchOutput.schema(
                    {
                        "bucket_ref": {
                            "enum": [
                                item.bucket_ref for item in request.recall_buckets
                            ]
                        },
                        "exhaustive_resource_names": {
                            "type": "array",
                            "items": resource_name,
                        },
                        "matching_resource_names": {
                            "type": "array",
                            "items": resource_name,
                        },
                    }
                ),
            },
            "input_resource_search_terms": {
                "type": "array",
                "minItems": len(request.reference_tasks),
                "maxItems": len(request.reference_tasks),
                "items": output.InputResourceSearchTermsOutput.schema(
                    {
                        "input_use_ref": {
                            "enum": [item.input_use_ref for item in request.reference_tasks]
                        },
                        "catalog_search_terms": {
                            "type": "array",
                            "items": resource_name,
                        },
                    }
                ),
            },
        }
    )
    schema["$defs"] = {"resource_name": resource_name_definition}
    return schema


__all__ = ["build_semantic_query_enrichment_schema"]
