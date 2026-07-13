"""Canonical storage codec for the closed question-run execution specs."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from .ports import (
    RerunProgramSpec,
    ResolveQuestionRunSpec,
    RunExecutionSpec,
    RunExecutionSpecKind,
    fold_run_execution_spec,
)
from fervis.lookup.clarification.response import (
    clarification_response_from_payload,
    clarification_response_payload,
)
from fervis.lookup.clarification.model import ClarificationOwnerResponse


def execution_spec_kind(spec: RunExecutionSpec) -> RunExecutionSpecKind:
    return fold_run_execution_spec(
        spec,
        resolve_question=lambda _spec: RunExecutionSpecKind.RESOLVE_QUESTION,
        rerun_program=lambda _spec: RunExecutionSpecKind.RERUN_PROGRAM,
    )


def execution_spec_to_storage_dict(spec: RunExecutionSpec) -> dict[str, Any]:
    return fold_run_execution_spec(
        spec,
        resolve_question=_resolve_question_storage_dict,
        rerun_program=_rerun_program_storage_dict,
    )


def _resolve_question_storage_dict(spec: ResolveQuestionRunSpec) -> dict[str, Any]:
    return {
        "question": spec.question,
        "provider": spec.provider,
        "model_key": spec.model_key,
        "context_run_id": spec.context_run_id,
        "conversation_context": dict(spec.conversation_context),
        "runtime_context": dict(spec.runtime_context),
        "max_budget_usd": (
            None if spec.max_budget_usd is None else str(spec.max_budget_usd)
        ),
        "max_thinking_tokens": spec.max_thinking_tokens,
        "clarification_responses": [
            clarification_response_payload(response)
            for response in spec.clarification_responses
        ],
    }


def _rerun_program_storage_dict(spec: RerunProgramSpec) -> dict[str, Any]:
    return {
        "invocation_id": spec.invocation_id,
        "runtime_context": dict(spec.runtime_context),
    }


def execution_spec_from_storage(
    kind: str | RunExecutionSpecKind,
    payload: Mapping[str, object],
) -> RunExecutionSpec:
    parsed_kind = (
        kind if isinstance(kind, RunExecutionSpecKind) else RunExecutionSpecKind(kind)
    )
    values = dict(payload)
    if parsed_kind is RunExecutionSpecKind.RESOLVE_QUESTION:
        _require_exact_fields(
            values,
            {
                "question",
                "provider",
                "model_key",
                "context_run_id",
                "conversation_context",
                "runtime_context",
                "max_budget_usd",
                "max_thinking_tokens",
                "clarification_responses",
            },
        )
        provider = values["provider"]
        context_run_id = values["context_run_id"]
        max_budget = values["max_budget_usd"]
        max_thinking = values["max_thinking_tokens"]
        clarification_responses = values["clarification_responses"]
        return ResolveQuestionRunSpec(
            question=_required_text(values["question"], "question"),
            provider=None if provider is None else _text(provider, "provider"),
            model_key=_text(values["model_key"], "model_key"),
            context_run_id=(
                None
                if context_run_id is None
                else _required_text(context_run_id, "context_run_id")
            ),
            conversation_context=_json_object(
                values["conversation_context"], "conversation_context"
            ),
            runtime_context=_json_object(values["runtime_context"], "runtime_context"),
            max_budget_usd=(
                None
                if max_budget is None
                else Decimal(_text(max_budget, "max_budget_usd"))
            ),
            max_thinking_tokens=(
                None
                if max_thinking is None
                else _integer(max_thinking, "max_thinking_tokens")
            ),
            clarification_responses=_clarification_responses_from_storage(
                clarification_responses
            ),
        )
    _require_exact_fields(values, {"invocation_id", "runtime_context"})
    return RerunProgramSpec(
        invocation_id=_required_text(values["invocation_id"], "invocation_id"),
        runtime_context=_json_object(values["runtime_context"], "runtime_context"),
    )


def _clarification_responses_from_storage(
    value: object,
) -> tuple[ClarificationOwnerResponse, ...]:
    if not isinstance(value, list):
        raise ValueError("execution spec clarification_responses must be a list")
    return tuple(
        clarification_response_from_payload(
            _json_object(item, "clarification_responses item")
        )
        for item in value
    )


def _require_exact_fields(values: Mapping[str, object], expected: set[str]) -> None:
    actual = set(values)
    if actual != expected:
        raise ValueError(
            "execution spec fields do not match contract: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _required_text(value: object, field: str) -> str:
    text = _text(value, field).strip()
    if not text:
        raise ValueError(f"execution spec {field} is required")
    return text


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"execution spec {field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"execution spec {field} must be an integer")
    return value


def _json_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"execution spec {field} must be an object")
    return dict(value)
