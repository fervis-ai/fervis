"""Model turn for pre-plan grounding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.lookup.grounding.model import (
    GroundingRequest,
    GroundingSelectionResult,
)
from fervis.lookup.grounding.parser import parse_grounding_compatibility
from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
from fervis.lookup.grounding.prompt import (
    GroundingTurnPrompt,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context


@dataclass(frozen=True)
class GroundingTurnResult:
    result: GroundingSelectionResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class GroundingGenerationError(LookupModelTurnError):
    pass


def generate_grounding(
    *,
    request: GroundingRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> GroundingTurnResult:
    invocation = GroundingTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
        )
    )
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message="grounding prompt budget exceeded",
            model_error_message="grounding model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise GroundingGenerationError(**generation_error_kwargs(exc)) from exc
    artifact = replace(
        output.artifact,
        derived_payload=lineage_explanation_metadata(
            (
                "known_input_bindings",
                "*",
                "selection_basis",
            ),
        ),
    )
    try:
        result = parse_grounding_compatibility(output.arguments, request=request)
    except Exception as exc:
        raise GroundingGenerationError(
            message="grounding parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=artifact,
        ) from exc
    return GroundingTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
    )
