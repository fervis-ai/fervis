"""Question lifecycle request and result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.lineage.enums import RunTriggerKind
from fervis.lookup.answer_program import BindingPatch, CapabilityApplication


DEFAULT_MAX_BUDGET_USD = Decimal("0.5")
DEFAULT_MAX_BUDGET_USD_LIMIT = Decimal("10.0")
DEFAULT_MAX_THINKING_TOKENS = 64
MAX_THINKING_TOKENS = 4096
DEFAULT_MODEL_KEY = "HAIKU"


class QuestionLifecycleError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ExecutionMode(str, Enum):
    INLINE = "inline"
    QUEUED = "queued"


@dataclass(frozen=True)
class QuestionPrincipal:
    principal_id: str
    tenant_id: str
    raw: Any = None
    read_context_ref: ReadContextRef = field(
        default_factory=lambda: ReadContextRef(scheme="anonymous")
    )
    delegated_credential: DelegatedReadCredential | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.read_context_ref, ReadContextRef):
            object.__setattr__(
                self,
                "read_context_ref",
                ReadContextRef.from_storage_dict(self.read_context_ref),
            )
        if self.delegated_credential is not None and not isinstance(
            self.delegated_credential,
            DelegatedReadCredential,
        ):
            object.__setattr__(
                self,
                "delegated_credential",
                DelegatedReadCredential.from_storage_dict(self.delegated_credential),
            )


@dataclass(frozen=True)
class AskRequestLimits:
    max_budget_usd: Any = DEFAULT_MAX_BUDGET_USD_LIMIT
    max_thinking_tokens: int = MAX_THINKING_TOKENS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_budget_usd",
            _normalize_positive_decimal(
                self.max_budget_usd,
                field_name="max_budget_usd_limit",
            ),
        )
        try:
            tokens = int(self.max_thinking_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_thinking_tokens_limit must be an integer") from exc
        if tokens < 1:
            raise ValueError("max_thinking_tokens_limit must be at least 1")
        object.__setattr__(self, "max_thinking_tokens", tokens)


@dataclass(frozen=True)
class AskRequest:
    question: str
    principal: QuestionPrincipal
    execution_mode: ExecutionMode = ExecutionMode.QUEUED
    conversation_id: str = ""
    context_run_id: str | None = None
    provider: str | None = None
    model_key: str = ""
    idempotency_key: str | None = None
    max_budget_usd: Any = None
    max_thinking_tokens: int | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)
    limits: AskRequestLimits = field(default_factory=AskRequestLimits)

    def __post_init__(self) -> None:
        if self.context_run_id is not None and (
            not isinstance(self.context_run_id, str) or not self.context_run_id.strip()
        ):
            raise ValueError("context_run_id must be a non-empty string")
        if not isinstance(self.execution_mode, ExecutionMode):
            object.__setattr__(
                self,
                "execution_mode",
                ExecutionMode(str(self.execution_mode)),
            )
        if not isinstance(self.limits, AskRequestLimits):
            raise ValueError("limits must be an AskRequestLimits instance")
        object.__setattr__(
            self,
            "model_key",
            _normalize_model_ref(self.model_key),
        )
        object.__setattr__(
            self,
            "max_budget_usd",
            _normalize_max_budget_usd(self.max_budget_usd, self.limits),
        )
        object.__setattr__(
            self,
            "max_thinking_tokens",
            _normalize_max_thinking_tokens(self.max_thinking_tokens, self.limits),
        )

    def accepted_trigger(self) -> dict[str, object] | None:
        return None


@dataclass(frozen=True)
class ClarificationResponseRequest:
    question_id: str
    run_id: str
    clarification_id: str
    response_text: str
    principal: QuestionPrincipal
    selected_option_id: str = ""
    execution_mode: ExecutionMode = ExecutionMode.QUEUED

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, ExecutionMode):
            object.__setattr__(
                self,
                "execution_mode",
                ExecutionMode(str(self.execution_mode)),
            )
        for field_name in (
            "question_id",
            "run_id",
            "clarification_id",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"clarification response requires {field_name}")
        if not self.response_text.strip() and not self.selected_option_id.strip():
            raise ValueError(
                "clarification response requires response_text or selected_option_id"
            )

    def accepted_trigger(self) -> dict[str, object]:
        return {
            "kind": "clarification_response",
            "run_id": self.run_id,
            "clarification_id": self.clarification_id,
        }


@dataclass(frozen=True)
class RetryQuestionRequest:
    question_id: str
    question: str
    principal: QuestionPrincipal
    base_run_id: str
    execution_mode: ExecutionMode = ExecutionMode.QUEUED
    provider: str | None = None
    model_key: str = ""
    idempotency_key: str | None = None
    max_budget_usd: Any = None
    max_thinking_tokens: int | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)
    limits: AskRequestLimits = field(default_factory=AskRequestLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, ExecutionMode):
            object.__setattr__(
                self,
                "execution_mode",
                ExecutionMode(str(self.execution_mode)),
            )
        if not self.question.strip():
            raise ValueError("retry requires question")
        if not self.base_run_id.strip():
            raise ValueError("retry requires base_run_id")
        if not isinstance(self.limits, AskRequestLimits):
            raise ValueError("limits must be an AskRequestLimits instance")
        object.__setattr__(self, "model_key", _normalize_model_ref(self.model_key))
        object.__setattr__(
            self,
            "max_budget_usd",
            _normalize_max_budget_usd(self.max_budget_usd, self.limits),
        )
        object.__setattr__(
            self,
            "max_thinking_tokens",
            _normalize_max_thinking_tokens(self.max_thinking_tokens, self.limits),
        )

    def accepted_trigger(self) -> dict[str, object]:
        return {
            "kind": RunTriggerKind.RETRY.value,
            "base_run_id": self.base_run_id,
        }


@dataclass(frozen=True)
class RerunQuestionRequest:
    question_id: str
    base_run_id: str
    principal: QuestionPrincipal
    patch: BindingPatch | None = None
    capability_application: CapabilityApplication | None = None
    execution_mode: ExecutionMode = ExecutionMode.QUEUED
    idempotency_key: str | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.patch is not None and self.capability_application is not None:
            raise ValueError(
                "rerun request accepts a patch or capability application, not both"
            )
        if not self.question_id.strip():
            raise ValueError("rerun request requires question_id")
        if not self.base_run_id.strip():
            raise ValueError("rerun request requires base_run_id")

    def accepted_trigger(self) -> dict[str, object]:
        return {
            "kind": RunTriggerKind.RERUN.value,
            "base_run_id": self.base_run_id,
        }


@dataclass(frozen=True)
class AskResult:
    status: str
    conversation_id: str
    question_id: str
    run_id: str
    answer: str | None = None
    result_data: dict[str, Any] | None = None
    error: str | None = None
    active_run_id: str | None = None
    duration_ms: int | None = None


def _normalize_max_budget_usd(value: Any, limits: AskRequestLimits) -> Decimal:
    if value is None or value == "":
        budget = DEFAULT_MAX_BUDGET_USD
    else:
        budget = _normalize_positive_decimal(
            value,
            field_name="max_budget_usd",
        )
    if budget > limits.max_budget_usd:
        raise ValueError(f"max_budget_usd must be at most {limits.max_budget_usd}")
    return budget


def _normalize_positive_decimal(value: Any, *, field_name: str) -> Decimal:
    try:
        budget = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive decimal value") from exc
    try:
        valid_budget = budget.is_finite() and budget > Decimal("0")
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} must be a positive decimal value") from exc
    if not valid_budget:
        raise ValueError(f"{field_name} must be greater than 0")
    return budget


def _normalize_model_ref(value: str) -> str:
    normalized = str(value or DEFAULT_MODEL_KEY).strip()
    if ":" in normalized:
        return normalized
    return normalized.upper()


def _normalize_max_thinking_tokens(
    value: int | None,
    limits: AskRequestLimits,
) -> int:
    if value is None:
        tokens = DEFAULT_MAX_THINKING_TOKENS
    else:
        try:
            tokens = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_thinking_tokens must be an integer") from exc
    if tokens < 1:
        raise ValueError("max_thinking_tokens must be at least 1")
    if tokens > limits.max_thinking_tokens:
        raise ValueError(
            f"max_thinking_tokens must be at most {limits.max_thinking_tokens}"
        )
    return tokens
