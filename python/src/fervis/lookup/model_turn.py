"""Shared mechanics for lookup model turns."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from fervis.lookup.errors import ErrorCode
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
class ModelTurnOutput:
    arguments: dict[str, Any]
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class LookupModelTurnError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class ModelTurnGenerationFailure(LookupModelTurnError):
    pass


def generation_error_kwargs(
    failure: ModelTurnGenerationFailure,
) -> dict[str, Any]:
    return {
        "message": failure.message,
        "usage": failure.usage,
        "duration_ms": failure.duration_ms,
        "artifact": failure.artifact,
        "error_code": failure.error_code,
        "error_context": failure.error_context,
    }


def run_one_of_tool_model_turn(
    *,
    invocation: Any,
    model_port: Any,
    provider: str,
    max_thinking_tokens: int,
    prompt_budget_error_message: str,
    model_error_message: str,
    prompt_budget_tool_specs: Any | None = None,
) -> ModelTurnOutput:
    prompt = invocation.prompt_text
    system_prompt = invocation.system_prompt
    provider_schema = invocation.provider_schema
    tool_specs = invocation.tool_specs
    try:
        enforce_model_turn_prompt_budget(
            prompt=prompt,
            tool_specs=prompt_budget_tool_specs or tool_specs,
        )
    except ModelTurnPromptBudgetError as exc:
        raise ModelTurnGenerationFailure(
            message=prompt_budget_error_message,
            usage={},
            duration_ms=0,
            artifact=ModelTurnArtifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=provider_schema,
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
        raise ModelTurnGenerationFailure(
            message=model_error_message,
            usage=dict(exc.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=model_turn_artifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=provider_schema,
                tool_specs=tool_specs,
                submitted_payload=exc.arguments,
                raw_output=exc.raw_output,
            ),
            error_code=exc.error_code or ErrorCode.PLANNING_FAILED,
            error_context=dict(exc.error_context or {}),
        ) from exc
    duration_ms = int((time.monotonic() - started) * 1000)
    return ModelTurnOutput(
        arguments=output.arguments,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=model_turn_artifact(
            system_prompt=system_prompt,
            prompt_text=prompt,
            provider_schema=provider_schema,
            tool_specs=tool_specs,
            submitted_payload=output.arguments,
            raw_output=output.raw_output,
            selected_tool_name=output.tool_spec.name,
        ),
    )
