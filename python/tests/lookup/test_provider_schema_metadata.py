from __future__ import annotations

from typing import Any

from fervis.lookup.query_enrichment.semantic import SemanticQueryEnrichmentRequest
from fervis.lookup.query_enrichment.semantic_schema import (
    build_semantic_query_enrichment_schema,
)
from fervis.lookup.question_contract.semantic_schema import (
    build_semantic_question_frame_schema,
)


def test_lookup_provider_schemas_do_not_emit_internal_model_schemas_metadata():
    for schema in (
        build_semantic_question_frame_schema(),
        build_semantic_query_enrichment_schema(
            SemanticQueryEnrichmentRequest(
                recall_buckets=(),
                reference_tasks=(),
                resource_names=(),
            )
        ),
    ):
        assert not _contains_key(schema, "modelSchemas")


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_key(item, key) for item in value)
    return False
