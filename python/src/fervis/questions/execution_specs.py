"""Canonical storage codec for the closed question-run execution specs."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from fervis.lineage.enums import QuestionRunKind

from .ports import (
    DeterministicRunSpec,
    ModelAssistedRunSpec,
    RunExecutionSpec,
    fold_run_execution_spec,
)


def execution_spec_kind(spec: RunExecutionSpec) -> QuestionRunKind:
    return fold_run_execution_spec(
        spec,
        model_assisted=lambda _spec: QuestionRunKind.MODEL_ASSISTED,
        deterministic=lambda _spec: QuestionRunKind.DETERMINISTIC,
    )


def execution_spec_to_storage_dict(spec: RunExecutionSpec) -> dict[str, Any]:
    return fold_run_execution_spec(
        spec,
        model_assisted=_model_assisted_storage_dict,
        deterministic=_deterministic_storage_dict,
    )


def _model_assisted_storage_dict(spec: ModelAssistedRunSpec) -> dict[str, Any]:
    return {
        "integrated_question": spec.integrated_question,
        "provider": spec.provider,
        "model_key": spec.model_key,
        "context_run_id": spec.context_run_id,
        "conversation_context": dict(spec.conversation_context),
        "runtime_context": dict(spec.runtime_context),
        "max_budget_usd": (
            None if spec.max_budget_usd is None else str(spec.max_budget_usd)
        ),
        "max_thinking_tokens": spec.max_thinking_tokens,
    }


def _deterministic_storage_dict(spec: DeterministicRunSpec) -> dict[str, Any]:
    return {
        "invocation_id": spec.invocation_id,
        "runtime_context": dict(spec.runtime_context),
    }


def execution_spec_from_storage(
    kind: str | QuestionRunKind,
    payload: Mapping[str, object],
) -> RunExecutionSpec:
    parsed_kind = kind if isinstance(kind, QuestionRunKind) else QuestionRunKind(kind)
    values = dict(payload)
    if parsed_kind is QuestionRunKind.MODEL_ASSISTED:
        _require_exact_fields(
            values,
            {
                "integrated_question",
                "provider",
                "model_key",
                "context_run_id",
                "conversation_context",
                "runtime_context",
                "max_budget_usd",
                "max_thinking_tokens",
            },
        )
        provider = values["provider"]
        context_run_id = values["context_run_id"]
        max_budget = values["max_budget_usd"]
        max_thinking = values["max_thinking_tokens"]
        return ModelAssistedRunSpec(
            integrated_question=_required_text(
                values["integrated_question"], "integrated_question"
            ),
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
            runtime_context=_json_object(
                values["runtime_context"], "runtime_context"
            ),
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
        )
    _require_exact_fields(values, {"invocation_id", "runtime_context"})
    return DeterministicRunSpec(
        invocation_id=_required_text(values["invocation_id"], "invocation_id"),
        runtime_context=_json_object(values["runtime_context"], "runtime_context"),
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
