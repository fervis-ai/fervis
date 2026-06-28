"""Ports used by the framework-neutral question lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.lineage.enums import RunTriggerKind

from .contracts import ExecutionMode, QuestionPrincipal

_AUTHORIZED_QUESTION_ACCESS_SIGNATURE = object()


class QuestionRunSubmissionKind(str, Enum):
    CREATED = "created"
    EXISTING = "existing"
    ACTIVE_CONFLICT = "active_conflict"


@dataclass(frozen=True)
class QuestionStart:
    conversation_id: str
    tenant_id: str
    read_context_ref: ReadContextRef
    question_id: str
    question: str
    principal_id: str


@dataclass(frozen=True)
class QuestionRunStart:
    question_id: str
    run_id: str
    trigger_kind: RunTriggerKind
    integrated_question: str
    adapter_ref: str
    runtime_version: str
    previous_run_id: str | None = None
    trigger_clarification_response_run_id: str | None = None
    trigger_clarification_response_id: str | None = None


@dataclass(frozen=True)
class ClarificationResponseStart:
    response_id: str
    run_id: str
    clarification_id: str
    response_text: str


@dataclass(frozen=True)
class QuestionRunRecord:
    run: QuestionRunStart
    question: QuestionStart | None = None
    clarification_response: ClarificationResponseStart | None = None


@dataclass(frozen=True)
class RunSubmission:
    conversation_id: str
    tenant_id: str
    question_id: str
    run_id: str
    question: str
    principal: QuestionPrincipal
    provider: str | None
    model_key: str
    execution_mode: ExecutionMode = ExecutionMode.QUEUED
    conversation_context: dict[str, Any] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    max_budget_usd: Any = None
    max_thinking_tokens: int | None = None


@dataclass(frozen=True)
class AuthorizedQuestionAccess:
    question_id: str
    conversation_id: str
    tenant_id: str
    original_question: str
    read_context_ref: ReadContextRef
    _signature: object = field(repr=False, compare=False)

    @classmethod
    def _issue(
        cls,
        *,
        question_id: str,
        conversation_id: str,
        tenant_id: str,
        original_question: str,
        read_context_ref: ReadContextRef,
    ) -> "AuthorizedQuestionAccess":
        return cls(
            question_id=question_id,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            original_question=original_question,
            read_context_ref=read_context_ref,
            _signature=_AUTHORIZED_QUESTION_ACCESS_SIGNATURE,
        )

    def require_valid(self) -> None:
        if self._signature is not _AUTHORIZED_QUESTION_ACCESS_SIGNATURE:
            raise PermissionError("question access was not issued by Fervis")


@dataclass(frozen=True)
class QueuedRun:
    submission: RunSubmission
    status: str
    answer: str | None = None
    result_data: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class QuestionRunSubmissionResult:
    kind: QuestionRunSubmissionKind
    run: QueuedRun

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QuestionRunSubmissionKind):
            object.__setattr__(
                self,
                "kind",
                QuestionRunSubmissionKind(str(self.kind)),
            )


@dataclass(frozen=True)
class LookupExecutionRequest:
    run_id: str
    conversation_id: str
    tenant_id: str
    question: str
    read_context_ref: ReadContextRef
    principal: Any
    provider: str | None
    model_key: str
    conversation_context: dict[str, Any]
    runtime_context: dict[str, Any]
    max_budget_usd: Any
    max_thinking_tokens: int | None
    active_attempt: int | None = None
    delegated_credential: DelegatedReadCredential | None = None


@dataclass(frozen=True)
class LookupExecutionResult:
    status: str
    answer: str | None = None
    result_data: dict[str, Any] | None = None
    error: str | None = None
    terminal_lineage_recorded: bool = True
    usage: dict[str, Any] = field(default_factory=dict)


class QuestionIdPort(Protocol):
    def new_conversation_id(self) -> str: ...

    def new_question_id(self) -> str: ...

    def new_run_id(self) -> str: ...

    def new_clarification_response_id(self) -> str: ...


class QuestionLineagePort(Protocol):
    def conversation_memory_context(
        self,
        *,
        conversation_id: str,
        authority: ReadAuthority,
    ) -> dict[str, Any]: ...

    def record_failed_runtime_fallback(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
    ) -> None: ...


class QuestionLifecyclePort(Protocol):
    def get_question(
        self,
        *,
        question_id: str,
        authority: ReadAuthority,
    ) -> AuthorizedQuestionAccess | None: ...

    def authorize_conversation(
        self,
        *,
        conversation_id: str,
        authority: ReadAuthority,
    ) -> None: ...

    def find_idempotent_run(
        self,
        *,
        authority: ReadAuthority,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None: ...

    def submit_question_run_atomically(
        self,
        *,
        submission: RunSubmission,
        record: QuestionRunRecord,
    ) -> QuestionRunSubmissionResult: ...

    def load_executable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun: ...

    def load_failable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun: ...

    def terminalize(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
        worker_id: str = "",
        active_attempt: int | None = None,
    ) -> QueuedRun: ...


class QuestionStateReaderPort(Protocol):
    def list_conversations(
        self,
        *,
        authority: ReadAuthority,
    ) -> list[dict[str, Any]]: ...

    def get_question_state(
        self,
        *,
        access: AuthorizedQuestionAccess,
    ) -> dict[str, Any] | None: ...

    def list_question_runs(
        self,
        *,
        access: AuthorizedQuestionAccess,
    ) -> list[dict[str, Any]]: ...

    def get_question_run(
        self,
        run_id: str,
        *,
        access: AuthorizedQuestionAccess,
    ) -> dict[str, Any] | None: ...


class QuestionLookupPort(Protocol):
    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink: Any = None,
    ) -> LookupExecutionResult: ...
