"""Question-contract model turn."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
)
from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
from fervis.lookup.question_contract.model import (
    QuestionContract,
    QuestionContractRequest,
    QuestionContractResult,
    validate_question_contract_against_question,
)
from fervis.lookup.question_contract.parser import parse_question_contract
from fervis.lookup.question_contract.prompt import (
    QuestionContractTurnPrompt,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.turn_prompts.context import active_clarification_context


@dataclass(frozen=True)
class QuestionContractTurnResult:
    result: QuestionContractResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


class QuestionContractGenerationError(LookupModelTurnError):
    pass


def generate_question_contract(
    *,
    request: QuestionContractRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> QuestionContractTurnResult:
    prompt_request = QuestionContractRequest(
        current_question=request.current_question,
        conversation_context=request.conversation_context,
        conversation_resolution=request.conversation_resolution,
        host=request.host,
    )
    invocation = QuestionContractTurnPrompt(prompt_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.current_question,
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
            prompt_budget_error_message="question contract prompt budget exceeded",
            model_error_message="question contract model turn failed",
        )
    except ModelTurnGenerationFailure as exc:
        raise QuestionContractGenerationError(**generation_error_kwargs(exc)) from exc
    try:
        result = parse_question_contract(
            tool_name=output.artifact.selected_tool_name or "",
            payload=output.arguments,
            question_context=request.current_question,
            question_context_texts=_question_contract_context_texts(request),
            current_question_context_texts=(
                _question_contract_active_clarification_texts(request)
            ),
            conversation_resolution=request.conversation_resolution,
        )
        if isinstance(result.outcome, QuestionContract):
            validate_question_contract_against_question(
                result.outcome,
                question=request.current_question,
                context_texts=_question_contract_context_texts(request),
            )
    except Exception as exc:
        raise QuestionContractGenerationError(
            message="question contract parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=output.artifact,
        ) from exc
    artifact = replace(
        output.artifact,
        parsed_payload=result.outcome.to_model_dict(),
    )
    return QuestionContractTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
    )


def _question_contract_context_texts(
    request: QuestionContractRequest,
) -> tuple[str, ...]:
    output = list(
        request.conversation_resolution.context_texts()
        if request.conversation_resolution is not None
        else ()
    )
    active = active_clarification_context(
        request.conversation_context,
        current_question=request.current_question,
    )
    if active is not None:
        output.append(active.original_question)
        for exchange in active.exchanges:
            output.extend(exchange.questions)
            output.append(exchange.answer)
    return tuple(dict.fromkeys(text for text in output if str(text or "").strip()))


def _question_contract_active_clarification_texts(
    request: QuestionContractRequest,
) -> tuple[str, ...]:
    active = active_clarification_context(
        request.conversation_context,
        current_question=request.current_question,
    )
    if active is None:
        return ()
    output: list[str] = [active.original_question]
    for exchange in active.exchanges:
        output.extend(exchange.questions)
        output.append(exchange.answer)
    return tuple(dict.fromkeys(text for text in output if str(text or "").strip()))
