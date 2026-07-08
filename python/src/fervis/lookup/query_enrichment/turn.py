"""Model turn for catalog query enrichment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.lookup.conversation_resolution import (
    conversation_resolution_query_enrichment_prompt_payload,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
)
from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.query_enrichment.model import (
    QueryEnrichmentRequest,
    QueryEnrichmentResult,
)
from fervis.lookup.query_enrichment.parser import parse_query_enrichment
from fervis.lookup.query_enrichment.prompt import QueryEnrichmentTurnPrompt


@dataclass(frozen=True)
class QueryEnrichmentTurnResult:
    result: QueryEnrichmentResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class QueryEnrichmentGenerationError(LookupModelTurnError):
    pass


def generate_query_enrichment(
    *,
    request: QueryEnrichmentRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> QueryEnrichmentTurnResult:
    invocation = QueryEnrichmentTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
            conversation_resolution_overlay=conversation_resolution_query_enrichment_prompt_payload(
                request.conversation_resolution_overlay
            ),
        )
    )
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message="query enrichment prompt budget exceeded",
            model_error_message="query enrichment model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise QueryEnrichmentGenerationError(**generation_error_kwargs(exc)) from exc
    artifact = replace(
        output.artifact,
        derived_payload=lineage_explanation_metadata(
            (
                "entity_target_catalog_search_terms",
                "*",
                "catalog_search_terms",
                "*",
                "basis",
            ),
        ),
    )
    try:
        result = parse_query_enrichment(output.arguments, request=request)
    except Exception as exc:
        raise QueryEnrichmentGenerationError(
            message="query enrichment parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=artifact,
        ) from exc
    return QueryEnrichmentTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
    )
