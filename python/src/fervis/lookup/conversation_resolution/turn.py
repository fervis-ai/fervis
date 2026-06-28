"""Conversation-resolution model turn."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from fervis.lookup.errors import ErrorCode
from fervis.lookup.conversation_resolution.model import (
    ConversationResolutionRequest,
    ConversationResolutionResult,
)
from fervis.lookup.conversation_resolution.parser import (
    parse_conversation_resolution,
)
from fervis.lookup.conversation_resolution.overlay import (
    conversation_resolution_overlay_from,
    conversation_resolution_query_enrichment_prompt_payload,
)
from fervis.lookup.conversation_resolution.prompt import (
    ConversationResolutionTurnPrompt,
    conversation_resolution_context_frames,
    conversation_resolution_context_sources,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
    model_turn_artifact,
)
from fervis.model_io.structured_output.errors import RequiredToolOutputError
from fervis.model_io.structured_output.generation import (
    generate_one_of_tool_output,
)
from fervis.model_io.telemetry import (
    ModelTurnPromptBudgetError,
    enforce_model_turn_prompt_budget,
)


@dataclass(frozen=True)
class ConversationResolutionTurnResult:
    result: ConversationResolutionResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class ConversationResolutionGenerationError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def generate_conversation_resolution(
    *,
    request: ConversationResolutionRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> ConversationResolutionTurnResult:
    invocation = ConversationResolutionTurnPrompt(request).to_model_invocation()
    prompt = invocation.prompt_text
    system_prompt = invocation.system_prompt
    schema = invocation.provider_schema
    tool_specs = invocation.tool_specs
    try:
        enforce_model_turn_prompt_budget(prompt=prompt, tool_specs=tool_specs)
    except ModelTurnPromptBudgetError as exc:
        raise ConversationResolutionGenerationError(
            message="conversation resolution prompt budget exceeded",
            usage={},
            duration_ms=0,
            artifact=ModelTurnArtifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=schema,
                tool_specs=tool_specs,
                submitted_payload={},
            ),
        ) from exc
    started = time.monotonic()
    try:
        output = generate_one_of_tool_output(
            model_port=model_port,
            provider=provider,
            system_prompt=system_prompt,
            prompt=prompt,
            max_thinking_tokens=max_thinking_tokens,
            tool_specs=tool_specs,
        )
    except RequiredToolOutputError as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        raise ConversationResolutionGenerationError(
            message="conversation resolution model turn failed",
            usage=dict(exc.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=model_turn_artifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=schema,
                tool_specs=tool_specs,
                submitted_payload=exc.arguments,
                raw_output=exc.raw_output,
            ),
            error_code=exc.error_code or ErrorCode.PLANNING_FAILED,
            error_context=dict(exc.error_context or {}),
        ) from exc
    duration_ms = int((time.monotonic() - started) * 1000)
    try:
        result = parse_conversation_resolution(
            tool_name=output.tool_spec.name,
            payload=output.arguments,
            current_question=request.question,
            context_sources=conversation_resolution_context_sources(request),
            context_frames=conversation_resolution_context_frames(request),
        )
    except Exception as exc:
        artifact = model_turn_artifact(
            system_prompt=system_prompt,
            prompt_text=prompt,
            provider_schema=schema,
            tool_specs=tool_specs,
            submitted_payload=output.arguments,
            raw_output=output.raw_output,
            selected_tool_name=output.tool_spec.name,
        )
        raise ConversationResolutionGenerationError(
            message="conversation resolution parse failed",
            usage=dict(output.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=artifact,
        ) from exc
    artifact = model_turn_artifact(
        system_prompt=system_prompt,
        prompt_text=prompt,
        provider_schema=schema,
        tool_specs=tool_specs,
        submitted_payload=output.arguments,
        raw_output=output.raw_output,
        parsed_payload=result.outcome.to_model_dict(),
        derived_payload=_derived_payload(result),
        selected_tool_name=output.tool_spec.name,
    )
    return ConversationResolutionTurnResult(
        result=result,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=artifact,
    )


def _derived_payload(result: ConversationResolutionResult) -> dict[str, Any]:
    output = dict(result.outcome.activation_payload())
    value_frame_payload = conversation_resolution_query_enrichment_prompt_payload(
        conversation_resolution_overlay_from(result.outcome)
    )
    if value_frame_payload is not None:
        output.update(value_frame_payload)
    return output
