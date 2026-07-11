"""Conversation-resolution model turn."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.conversation_resolution.model import (
    ConversationResolutionRequest,
    ConversationResolutionResult,
)
from fervis.lookup.conversation_resolution.parser import (
    parse_conversation_resolution,
)
from fervis.lookup.conversation_resolution.prompt import (
    ConversationResolutionTurnPrompt,
    conversation_resolution_context_frames,
    conversation_resolution_context_sources,
)
from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
)


@dataclass(frozen=True)
class ConversationResolutionTurnResult:
    result: ConversationResolutionResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class ConversationResolutionGenerationError(LookupModelTurnError):
    pass


def generate_conversation_resolution(
    *,
    request: ConversationResolutionRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> ConversationResolutionTurnResult:
    invocation = ConversationResolutionTurnPrompt(request).to_model_invocation()
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message=(
                "conversation resolution prompt budget exceeded"
            ),
            model_error_message="conversation resolution model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise ConversationResolutionGenerationError(
            **generation_error_kwargs(exc)
        ) from exc
    try:
        result = parse_conversation_resolution(
            tool_name=output.artifact.selected_tool_name or "",
            payload=output.arguments,
            current_question=request.question,
            context_sources=conversation_resolution_context_sources(request),
            context_frames=conversation_resolution_context_frames(request),
        )
    except Exception as exc:
        raise ConversationResolutionGenerationError(
            message="conversation resolution parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=output.artifact,
        ) from exc
    artifact = replace(
        output.artifact,
        parsed_payload=result.outcome.to_model_dict(),
        derived_payload=_derived_payload(result),
    )
    return ConversationResolutionTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
    )


def _derived_payload(result: ConversationResolutionResult) -> dict[str, Any]:
    return result.outcome.activation_payload()
