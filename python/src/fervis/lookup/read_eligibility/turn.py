"""Model turn for API read retention."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
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
from fervis.lookup.read_eligibility.model import (
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.parser import parse_read_eligibility
from fervis.lookup.read_eligibility.prompt import ReadEligibilityTurnPrompt
from fervis.model_io.structured_output.provider_budget import (
    provider_budget_tool_specs,
)


@dataclass(frozen=True)
class ReadEligibilityTurnResult:
    result: ReadEligibilityResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class ReadEligibilityGenerationError(LookupModelTurnError):
    pass


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
        )
    )
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message="read eligibility prompt budget exceeded",
            model_error_message="read eligibility model turn failed",
            prompt_budget_tool_specs=provider_budget_tool_specs(
                provider=provider,
                tool_specs=invocation.tool_specs,
            ),
        )
    except ModelTurnGenerationFailure as exc:
        raise ReadEligibilityGenerationError(**generation_error_kwargs(exc)) from exc
    artifact = replace(
        output.artifact,
        derived_payload=lineage_explanation_metadata(
            ("read_candidate_reviews", "*", "retention_basis"),
        ),
    )
    try:
        result = parse_read_eligibility(output.arguments, request=request)
    except Exception as exc:
        raise ReadEligibilityGenerationError(
            message="read eligibility parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=artifact,
            error_context={"reason": str(exc)},
        ) from exc
    return ReadEligibilityTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
    )
