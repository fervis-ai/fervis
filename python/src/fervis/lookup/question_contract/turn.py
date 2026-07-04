"""Question-contract model turn."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from fervis.lookup.errors import ErrorCode
from fervis.lookup.conversation_resolution import (
    conversation_resolution_question_contract_context_texts,
    conversation_resolution_question_contract_prompt_payload,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
    model_turn_artifact,
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
from fervis.model_io.structured_output.errors import RequiredToolOutputError
from fervis.model_io.structured_output.generation import (
    generate_one_of_tool_output,
)
from fervis.model_io.telemetry import (
    ModelTurnPromptBudgetError,
    enforce_model_turn_prompt_budget,
)


@dataclass(frozen=True)
class QuestionContractTurnResult:
    result: QuestionContractResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class QuestionContractGenerationError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


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
        conversation_resolution_overlay=request.conversation_resolution_overlay,
        host=request.host,
    )
    invocation = QuestionContractTurnPrompt(prompt_request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.current_question,
            conversation_context=request.conversation_context,
            host=request.host,
            conversation_resolution_overlay=conversation_resolution_question_contract_prompt_payload(
                request.conversation_resolution_overlay
            ),
        )
    )
    prompt = invocation.prompt_text
    system_prompt = invocation.system_prompt
    question_contract_provider_schema = invocation.provider_schema
    tool_specs = invocation.tool_specs
    try:
        enforce_model_turn_prompt_budget(prompt=prompt, tool_specs=tool_specs)
    except ModelTurnPromptBudgetError as exc:
        raise QuestionContractGenerationError(
            message="question contract prompt budget exceeded",
            usage={},
            duration_ms=0,
            artifact=ModelTurnArtifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=question_contract_provider_schema,
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
        raise QuestionContractGenerationError(
            message="question contract model turn failed",
            usage=dict(exc.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=model_turn_artifact(
                system_prompt=system_prompt,
                prompt_text=prompt,
                provider_schema=question_contract_provider_schema,
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
        provider_schema=question_contract_provider_schema,
        tool_specs=tool_specs,
        submitted_payload=output.arguments,
        raw_output=output.raw_output,
        selected_tool_name=output.tool_spec.name,
    )
    try:
        result = parse_question_contract(
            tool_name=output.tool_spec.name,
            payload=output.arguments,
            question_context=request.current_question,
            question_context_texts=_question_contract_context_texts(request),
            conversation_resolution_overlay=request.conversation_resolution_overlay,
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
            usage=dict(output.output.get("usage") or {}),
            duration_ms=duration_ms,
            artifact=artifact,
        ) from exc
    artifact = model_turn_artifact(
        system_prompt=system_prompt,
        prompt_text=prompt,
        provider_schema=question_contract_provider_schema,
        tool_specs=tool_specs,
        submitted_payload=output.arguments,
        raw_output=output.raw_output,
        parsed_payload=result.outcome.to_model_dict(),
        selected_tool_name=output.tool_spec.name,
    )
    return QuestionContractTurnResult(
        result=result,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=artifact,
    )


def _question_contract_context_texts(
    request: QuestionContractRequest,
) -> tuple[str, ...]:
    output = list(
        conversation_resolution_question_contract_context_texts(
            request.conversation_resolution_overlay
        )
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
