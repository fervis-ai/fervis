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
from fervis.model_io.structured_output.errors import ModelValidationKind
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
    validation_failure_observer=None,
) -> ConversationResolutionTurnResult:
    from fervis.lookup.turn_prompts.correction import run_with_correction
    return run_with_correction(ConversationResolutionTurnPrompt(request),
        lambda prompt: _generate(request=request, prompt=prompt, model_port=model_port,
            provider=provider, max_thinking_tokens=max_thinking_tokens),
        validation_failure_observer)


def _generate(*, request, prompt, model_port, provider, max_thinking_tokens):
    from fervis.lookup.turn_prompts import build_turn_prompt_context
    invocation = prompt.to_model_invocation(build_turn_prompt_context(
        current_question=request.question, conversation_context=request.conversation_context, host=request.host))
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
            error_context={"exception_class": type(exc).__name__, "message": str(exc)},
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=output.artifact,
            validation_kind=ModelValidationKind.SEMANTIC if isinstance(exc,ValueError) else None,
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
