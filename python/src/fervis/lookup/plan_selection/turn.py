"""Model turn for fact-local plan selection."""

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
from fervis.lookup.plan_selection.model import (
    PlanSelectionRequest,
    PlanSelectionResult,
)
from fervis.lookup.plan_selection.parser import parse_plan_selection
from fervis.lookup.plan_selection.prompt import PlanSelectionTurnPrompt


@dataclass(frozen=True)
class PlanSelectionTurnResult:
    result: PlanSelectionResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class PlanSelectionGenerationError(LookupModelTurnError):
    pass


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
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message="plan selection prompt budget exceeded",
            model_error_message="plan selection model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise PlanSelectionGenerationError(**generation_error_kwargs(exc)) from exc
    artifact = replace(
        output.artifact,
        derived_payload=lineage_explanation_metadata(
            ("outcome", "reviews_by_requested_fact", "*", "*", "basis"),
        ),
    )
    try:
        result = parse_plan_selection(output.arguments, request=request)
    except Exception as exc:
        raise PlanSelectionGenerationError(
            message="plan selection parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=artifact,
            error_context={"reason": str(exc)},
        ) from exc
    return PlanSelectionTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
    )
