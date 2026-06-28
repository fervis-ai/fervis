"""Model turn for source binding."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from fervis.lookup.fact_plan.fact_plan import PlanImpossible
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.source_binding.model import (
    SourceBindingRequest,
    SourceBindingResult,
)
from fervis.lookup.source_binding.parser import (
    parse_source_binding,
)
from fervis.lookup.source_binding.prompt import (
    SOURCE_BINDING_TOOL_NAME,
    SourceBindingTurnPrompt,
)
from fervis.lookup.source_binding.terminal_outcomes import (
    backend_impossible_without_answer_candidates,
)
from fervis.model_io.structured_output.errors import RequiredToolOutputError
from fervis.model_io.structured_output.generation import (
    generate_one_of_tool_output,
)
from fervis.model_io.structured_output.provider_budget import (
    provider_budget_tool_specs,
)
from fervis.model_io.telemetry import (
    ModelTurnPromptBudgetError,
    enforce_model_turn_prompt_budget,
)


@dataclass(frozen=True)
class SourceBindingTurnResult:
    result: SourceBindingResult
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    subturns: tuple["SourceBindingSubturnResult", ...] = ()


@dataclass(frozen=True)
class SourceBindingSubturnResult:
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


@dataclass(frozen=True)
class SourceBindingGenerationError(Exception):
    message: str
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact
    error_code: str = ErrorCode.PLANNING_FAILED
    error_context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


def generate_source_binding(
    *,
    request: SourceBindingRequest,
    model_port: Any,
    provider: str,
    model_key: str,
    max_thinking_tokens: int,
) -> SourceBindingTurnResult:
    prompt = SourceBindingTurnPrompt(request)
    backend_impossible = backend_impossible_without_answer_candidates(request)
    if backend_impossible is not None:
        return _backend_impossible_turn_result(
            backend_impossible,
            request=request,
            prompt=prompt,
        )
    output = _run_source_binding_model_turn(
        request=request,
        prompt=prompt,
        model_port=model_port,
        provider=provider,
        max_thinking_tokens=max_thinking_tokens,
        error_label="source binding",
    )
    try:
        result = parse_source_binding(
            output.raw_arguments,
            request=request,
        )
    except Exception as exc:
        raise SourceBindingGenerationError(
            message="source binding parse failed",
            usage=output.usage,
            duration_ms=output.duration_ms,
            artifact=output.artifact,
            error_context={"reason": str(exc)},
        ) from exc
    artifact = replace(
        output.artifact,
        parsed_payload=output.raw_arguments,
        derived_payload=lineage_explanation_metadata(
            (
                "outcome",
                "source_invocations",
                "*",
                "finite_choice_param_reviews",
                "*",
                "role_selection_basis",
            ),
            (
                "outcome",
                "source_invocations",
                "*",
                "finite_choice_param_reviews",
                "*",
                "choice_reviews",
                "*",
                "choice_inclusion_basis",
            ),
        ),
    )
    return SourceBindingTurnResult(
        result=result,
        usage=output.usage,
        duration_ms=output.duration_ms,
        artifact=artifact,
        subturns=(
            SourceBindingSubturnResult(
                usage=output.usage,
                duration_ms=output.duration_ms,
                artifact=artifact,
            ),
        ),
    )


def _backend_impossible_turn_result(
    outcome: PlanImpossible,
    *,
    request: SourceBindingRequest,
    prompt: SourceBindingTurnPrompt,
) -> SourceBindingTurnResult:
    invocation = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
            memory_payload=request.memory_inputs,
            conversation_resolution_overlay=conversation_resolution_source_binding_prompt_payload(
                request.conversation_resolution_overlay
            ),
        )
    )
    submitted_payload = {
        "outcome": {
            "kind": "impossible",
            "blocked_facts": [
                {
                    "requested_fact_id": blocked.requested_fact_id,
                    "basis": blocked.basis.value,
                    "evidence_refs": list(blocked.evidence_refs),
                    "reviewed_read_ids": list(blocked.reviewed_read_ids),
                    "nearest_fields": [
                        {
                            "read_id": field.read_id,
                            "field_id": field.field_id,
                        }
                        for field in blocked.nearest_fields
                    ],
                    "explanation": blocked.explanation,
                }
                for blocked in outcome.blocked_facts
            ],
        }
    }
    return SourceBindingTurnResult(
        result=SourceBindingResult(outcome=outcome),
        usage={},
        duration_ms=0,
        artifact=ModelTurnArtifact(
            system_prompt=invocation.system_prompt,
            prompt_text=invocation.prompt_text,
            provider_schema=invocation.provider_schema,
            tool_specs=invocation.tool_specs,
            submitted_payload=submitted_payload,
            parsed_payload=submitted_payload,
            derived_payload={"backend_terminal": "no_answer_source_candidates"},
            selected_tool_name=SOURCE_BINDING_TOOL_NAME,
        ),
        subturns=(),
    )


@dataclass(frozen=True)
class _SourceBindingModelTurnOutput:
    raw_arguments: dict[str, Any]
    usage: dict[str, Any]
    duration_ms: int
    artifact: ModelTurnArtifact


def _run_source_binding_model_turn(
    *,
    request: SourceBindingRequest,
    prompt: Any,
    model_port: Any,
    provider: str,
    max_thinking_tokens: int,
    error_label: str,
) -> _SourceBindingModelTurnOutput:
    invocation = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
            memory_payload=request.memory_inputs,
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
        raise SourceBindingGenerationError(
            message=f"{error_label} prompt budget exceeded",
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
        raise SourceBindingGenerationError(
            message=f"{error_label} model turn failed",
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
    raw_arguments = output.arguments
    artifact = model_turn_artifact(
        system_prompt=system_prompt,
        prompt_text=prompt,
        provider_schema=schema,
        tool_specs=tool_specs,
        submitted_payload=raw_arguments,
        raw_output=output.raw_output,
        selected_tool_name=output.tool_spec.name,
    )
    return _SourceBindingModelTurnOutput(
        raw_arguments=raw_arguments,
        usage=dict(output.output.get("usage") or {}),
        duration_ms=duration_ms,
        artifact=artifact,
    )
