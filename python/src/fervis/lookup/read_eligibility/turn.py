"""Model turn for API read retention."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from fervis.lookup.errors import ErrorCode
from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.lookup.conversation_resolution import (
    conversation_resolution_source_binding_prompt_payload,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
    model_turn_artifact,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.read_eligibility.model import (
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.parser import parse_read_eligibility
from fervis.lookup.read_eligibility.prompt import ReadEligibilityTurnPrompt
from fervis.model_io.structured_output.provider_budget import (
    provider_budget_tool_specs,
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
class ReadEligibilityTurnResult:
    result: ReadEligibilityResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class ReadEligibilityGenerationError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def generate_read_eligibility(
    *,
    request: ReadEligibilityRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> ReadEligibilityTurnResult:
    invocation = ReadEligibilityTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
            conversation_resolution_overlay=conversation_resolution_source_binding_prompt_payload(
                request.conversation_resolution_overlay
            ),
        )
    )
    prompt = invocation.prompt_text
    system_prompt = invocation.system_prompt
    schema = invocation.provider_schema
    tool_specs = invocation.tool_specs
    try:
        enforce_model_turn_prompt_budget(
            prompt=prompt,
            tool_specs=provider_budget_tool_specs(
                provider=provider,
                tool_specs=tool_specs,
            ),
        )
    except ModelTurnPromptBudgetError as exc:
        raise ReadEligibilityGenerationError(
            message="read eligibility prompt budget exceeded",
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
        raise ReadEligibilityGenerationError(
            message="read eligibility model turn failed",
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
    artifact = model_turn_artifact(
        system_prompt=system_prompt,
        prompt_text=prompt,
        provider_schema=schema,
        tool_specs=tool_specs,
        submitted_payload=output.arguments,
        raw_output=output.raw_output,
        derived_payload=lineage_explanation_metadata(
            ("read_candidate_reviews", "*", "retention_basis"),
        ),
        selected_tool_name=output.tool_spec.name,
    )
    try:
        result = parse_read_eligibility(output.arguments, request=request)
    except Exception as exc:
        raise ReadEligibilityGenerationError(
            message="read eligibility parse failed",
            usage=dict(output.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=artifact,
            error_context={"reason": str(exc)},
        ) from exc
    return ReadEligibilityTurnResult(
        result=result,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=artifact,
    )
