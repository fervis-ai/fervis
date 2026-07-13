"""Ports used by the framework-neutral question lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from fervis.types.enums import StrEnum
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar
from typing_extensions import assert_never

from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.lineage.enums import QuestionRunKind, RunTriggerKind
from fervis.lineage.recorder import (
    ProgramInvocationBundleWrite,
    ProgramRevisionBundleWrite,
)
from fervis.lookup.clarification.model import ClarificationOwnerResponse

from .contracts import ExecutionMode, QuestionPrincipal

if TYPE_CHECKING:
    from fervis.lookup.answer_program.persistence import StoredProgramInvocation

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
    kind: QuestionRunKind
    trigger_kind: RunTriggerKind
    adapter_ref: str
    runtime_version: str
    base_run_id: str | None = None


@dataclass(frozen=True)
class ClarificationRunResume:
    question_id: str
    run_id: str
    clarification_id: str
    response_id: str
    response_text: str
    selected_option_id: str
    principal: QuestionPrincipal
    execution_mode: ExecutionMode


@dataclass(frozen=True)
class QuestionRunRecord:
    run: QuestionRunStart
    question: QuestionStart | None = None
    program_invocation: ProgramInvocationBundleWrite | None = None
    program_revision: ProgramRevisionBundleWrite | None = None


@dataclass(frozen=True)
class ResolveQuestionRunSpec:
    question: str
    provider: str | None
    model_key: str
    context_run_id: str | None = None
    conversation_context: dict[str, Any] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    max_budget_usd: Any = None
    max_thinking_tokens: int | None = None
    clarification_response: ClarificationOwnerResponse | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("resolve-question run requires question")
        if self.context_run_id is not None and not self.context_run_id.strip():
            raise ValueError("resolve-question context_run_id must be non-empty")


@dataclass(frozen=True)
class RerunProgramSpec:
    invocation_id: str
    runtime_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.invocation_id.strip():
            raise ValueError("rerun-program spec requires invocation_id")


RunExecutionSpec: TypeAlias = ResolveQuestionRunSpec | RerunProgramSpec


class RunExecutionSpecKind(StrEnum):
    RESOLVE_QUESTION = "resolve_question"
    RERUN_PROGRAM = "rerun_program"


_RunSpecResult = TypeVar("_RunSpecResult")


def fold_run_execution_spec(
    spec: RunExecutionSpec,
    *,
    resolve_question: Callable[[ResolveQuestionRunSpec], _RunSpecResult],
    rerun_program: Callable[[RerunProgramSpec], _RunSpecResult],
) -> _RunSpecResult:
    """Apply one exhaustive operation over the closed run-spec union."""

    match spec:
        case ResolveQuestionRunSpec():
            return resolve_question(spec)
        case RerunProgramSpec():
            return rerun_program(spec)
        case _:
            assert_never(spec)


@dataclass(frozen=True)
class RunSubmission:
    conversation_id: str
    tenant_id: str
    question_id: str
    run_id: str
    principal: QuestionPrincipal
    spec: RunExecutionSpec
    execution_mode: ExecutionMode = ExecutionMode.QUEUED
    idempotency_key: str | None = None
    idempotency_authority_ref: str = ""
    idempotency_scope: str = ""

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            return
        if not self.idempotency_authority_ref:
            object.__setattr__(
                self,
                "idempotency_authority_ref",
                _idempotency_authority_ref(self.principal),
            )
        if not self.idempotency_scope:
            object.__setattr__(
                self,
                "idempotency_scope",
                f"conversation:{self.conversation_id}",
            )


def _idempotency_authority_ref(principal: QuestionPrincipal) -> str:
    payload = json.dumps(
        {
            "tenant_id": principal.tenant_id,
            "principal_id": principal.principal_id,
            "read_context_ref": principal.read_context_ref.to_storage_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "idempotency-authority:sha256:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ParsedQuestionRunSubmission:
    submission: RunSubmission
    record: QuestionRunRecord

    def __post_init__(self) -> None:
        submission = self.submission
        record = self.record
        if record.run.run_id != submission.run_id:
            raise ValueError("submission and lineage run ids must match")
        if record.run.question_id != submission.question_id:
            raise ValueError("submission and lineage question ids must match")
        if submission.principal.tenant_id != submission.tenant_id:
            raise ValueError("submission principal and tenant must match")
        if record.question is not None:
            if (
                record.question.question_id != submission.question_id
                or record.question.conversation_id != submission.conversation_id
                or record.question.tenant_id != submission.tenant_id
            ):
                raise ValueError("submission and question lineage must match")
            if (
                record.question.read_context_ref
                != submission.principal.read_context_ref
            ):
                raise ValueError("submission and question authority must match")
        fold_run_execution_spec(
            submission.spec,
            resolve_question=lambda spec: _validate_resolve_question_record(
                spec,
                record=record,
            ),
            rerun_program=lambda spec: _validate_rerun_program_record(
                spec,
                submission=submission,
                record=record,
            ),
        )


def _validate_resolve_question_record(
    spec: ResolveQuestionRunSpec,
    *,
    record: QuestionRunRecord,
) -> None:
    del spec
    if record.run.kind is not QuestionRunKind.MODEL_ASSISTED:
        raise ValueError("model-assisted spec requires model-assisted run")
    if record.program_invocation is not None:
        raise ValueError("model-assisted admission cannot include an invocation")
    if record.program_revision is not None:
        raise ValueError("model-assisted admission cannot include a revision")


def _validate_rerun_program_record(
    spec: RerunProgramSpec,
    *,
    submission: RunSubmission,
    record: QuestionRunRecord,
) -> None:
    if record.run.kind is not QuestionRunKind.DETERMINISTIC:
        raise ValueError("deterministic spec requires deterministic run")
    if record.run.trigger_kind is not RunTriggerKind.RERUN:
        raise ValueError("deterministic run requires rerun trigger")
    bundle = record.program_invocation
    if bundle is None:
        raise ValueError("deterministic run requires program invocation")
    if (
        bundle.invocation.run_id != submission.run_id
        or bundle.invocation.invocation_id != spec.invocation_id
        or bundle.invocation.program_id != bundle.program.program_id
    ):
        raise ValueError("deterministic invocation must match submission")
    revision = record.program_revision
    if bundle.invocation.revision_id is None:
        if revision is not None:
            raise ValueError("program revision requires a revised invocation")
        return
    if revision is None:
        raise ValueError("revised invocation requires its program revision")
    if (
        bundle.invocation.revision_id != revision.revision.revision_id
        or bundle.invocation.program_id != revision.revision.revised_program_id
        or bundle.program.program_id != revision.program.program_id
    ):
        raise ValueError("program revision must match deterministic invocation")


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
    duration_ms: int | None = None
    active_attempt: int | None = None


@dataclass(frozen=True)
class QuestionRunSubmissionResult:
    kind: QuestionRunSubmissionKind
    run: QueuedRun


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
    clarification_response: ClarificationOwnerResponse | None = None


@dataclass(frozen=True)
class LookupExecutionResult:
    status: str
    answer: str | None = None
    result_data: dict[str, Any] | None = None
    error: str | None = None
    terminal_lineage_recorded: bool = True
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramExecutionRequest:
    run_id: str
    conversation_id: str
    tenant_id: str
    question: str
    read_context_ref: ReadContextRef
    principal: Any
    invocation: StoredProgramInvocation
    runtime_context: dict[str, Any]
    active_attempt: int | None = None
    delegated_credential: DelegatedReadCredential | None = None


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
        context_run_id: str | None = None,
        continuation_run_id: str | None = None,
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

    def load_answered_program_invocation(
        self,
        *,
        run_id: str,
        access: AuthorizedQuestionAccess,
    ) -> StoredProgramInvocation | None: ...

    def load_prior_answered_invocation(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None: ...

    def load_program_invocation_for_execution(
        self,
        *,
        invocation_id: str,
        run_id: str,
        question_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None: ...

    def find_idempotent_run(
        self,
        *,
        principal: QuestionPrincipal,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None: ...

    def submit_question_run_atomically(
        self,
        *,
        submission: RunSubmission,
        record: QuestionRunRecord,
    ) -> QuestionRunSubmissionResult: ...

    def resume_question_run_atomically(
        self,
        resume: ClarificationRunResume,
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

    def wait_for_clarification(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
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


class QuestionProgramPort(Protocol):
    def run_program(
        self,
        request: ProgramExecutionRequest,
        *,
        progress_sink: Any = None,
    ) -> LookupExecutionResult: ...
