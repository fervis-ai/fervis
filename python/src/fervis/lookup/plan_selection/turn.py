"""Model turn for fact-local plan selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from fervis.lookup.errors import ErrorCode
from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
    model_turn_artifact,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.plan_selection.model import (
    PlanSelectionRequest,
    PlanSelectionResult,
)
from fervis.lookup.plan_selection.parser import parse_plan_selection
from fervis.lookup.plan_selection.prompt import PlanSelectionTurnPrompt
from fervis.model_io.structured_output.errors import RequiredToolOutputError
from fervis.model_io.structured_output.generation import (
    generate_one_of_tool_output,
)
from fervis.model_io.telemetry import (
    ModelTurnPromptBudgetError,
    enforce_model_turn_prompt_budget,
)


@dataclass(frozen=True)
class PlanSelectionTurnResult:
    result: PlanSelectionResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class PlanSelectionGenerationError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def generate_plan_selection(
    *,
    request: PlanSelectionRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> PlanSelectionTurnResult:
    invocation = PlanSelectionTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
        )
    )
    prompt = invocation.prompt_text
    system_prompt = invocation.system_prompt
    schema = invocation.provider_schema
    tool_specs = invocation.tool_specs
    try:
        enforce_model_turn_prompt_budget(prompt=prompt, tool_specs=tool_specs)
    except ModelTurnPromptBudgetError as exc:
        raise PlanSelectionGenerationError(
            message="plan selection prompt budget exceeded",
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
        raise PlanSelectionGenerationError(
            message="plan selection model turn failed",
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
            ("outcome", "reviews_by_requested_fact", "*", "*", "basis"),
        ),
        selected_tool_name=output.tool_spec.name,
    )
    try:
        result = parse_plan_selection(output.arguments, request=request)
    except Exception as exc:
        raise PlanSelectionGenerationError(
            message="plan selection parse failed",
            usage=dict(output.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=artifact,
            error_context={"reason": str(exc)},
        ) from exc
    return PlanSelectionTurnResult(
        result=result,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=artifact,
    )
