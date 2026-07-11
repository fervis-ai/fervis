"""Model turn for source binding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.model_io.turn_artifacts import (
    ModelTurnArtifact,
)
from fervis.lookup.fact_plan.fact_plan import PlanImpossible
from fervis.lookup.model_turn import (
    LookupModelTurnError,
    ModelTurnGenerationFailure,
    generation_error_kwargs,
    run_one_of_tool_model_turn,
)
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
from fervis.model_io.structured_output.provider_budget import (
    provider_budget_tool_specs,
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


class SourceBindingGenerationError(LookupModelTurnError):
    pass


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
    invocation = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            host=request.host,
            memory_payload=request.memory_inputs,
        )
    )
    try:
        output = run_one_of_tool_model_turn(
            invocation=invocation,
            model_port=model_port,
            provider=provider,
            max_thinking_tokens=max_thinking_tokens,
            prompt_budget_error_message="source binding prompt budget exceeded",
            model_error_message="source binding model turn failed",
            prompt_budget_tool_specs=provider_budget_tool_specs(
                provider=provider,
                tool_specs=invocation.tool_specs,
            ),
        )
    except ModelTurnGenerationFailure as exc:
        raise SourceBindingGenerationError(**generation_error_kwargs(exc)) from exc
    try:
        result = parse_source_binding(
            output.arguments,
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
        parsed_payload=output.arguments,
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
