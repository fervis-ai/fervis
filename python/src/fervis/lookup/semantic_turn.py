"""One execution boundary for semantic Lookup model turns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
from fervis.lookup.turn_prompts import TurnPromptBase, TurnPromptContext
from fervis.model_io.turn_artifacts import ModelTurnArtifact


T = TypeVar("T")


@dataclass(frozen=True)
class SemanticTurnResult(Generic[T]):
    result: T
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class SemanticTurnGenerationError(LookupModelTurnError):
    pass


def generate_semantic_turn(
    *,
    prompt: TurnPromptBase,
    context: TurnPromptContext,
    parse: Callable[[dict[str, object]], T],
    model_port: Any,
    provider: str,
    max_thinking_tokens: int,
) -> SemanticTurnResult[T]:
    """Run and parse one required strict semantic tool call."""

    invocation = prompt.to_model_invocation(context)
    turn_name = prompt.turn_name
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message=f"{turn_name} prompt budget exceeded",
            model_error_message=f"{turn_name} model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise SemanticTurnGenerationError(**generation_error_kwargs(exc)) from exc
    try:
        result = parse(output.arguments)
    except Exception as exc:
        raise SemanticTurnGenerationError(
            message=f"{turn_name} parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=output.artifact,
            error_context={"exception_class": type(exc).__name__, "message": str(exc)},
        ) from exc
    return SemanticTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=output.artifact,
    )


__all__ = [
    "SemanticTurnGenerationError",
    "SemanticTurnResult",
    "generate_semantic_turn",
]
