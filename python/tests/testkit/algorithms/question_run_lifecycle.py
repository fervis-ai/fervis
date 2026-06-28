from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fervis.questions import (
    AskRequest,
    ExecutionMode,
    QuestionPrincipal,
)
from fervis.questions.service import QuestionService
from fervis.run_work import FailQueuedRunRequest, QueuedRunRequest
from fervis.questions.ports import (
    AuthorizedQuestionAccess,
    LookupExecutionRequest,
    LookupExecutionResult,
    QuestionRunRecord,
    QuestionRunSubmissionKind,
    QuestionRunSubmissionResult,
    QueuedRun,
    RunSubmission,
)
from tests.testkit.assertions import subset_mismatches


def run_question_run_lifecycle_case(payload: dict[str, Any]) -> list[str]:
    scenario = str(payload["input"]["scenario"])
    if scenario != "ask":
        return [f"unsupported question-run lifecycle scenario: {scenario}"]
    return _run_ask_case(payload)


def _run_ask_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    lineage = _InMemoryLineage(
        memory_by_conversation=_memory_by_conversation(input_payload),
        fail_memory_after_calls=input_payload.get("fail_memory_after_calls"),
    )
    runs = _InMemoryRuns(lineage=lineage)
    lookup = _FakeLookup(input_payload.get("lookup_result") or {})
    service = QuestionService(
        lineage=lineage,
        runs=runs,
        lookup=lookup,
        ids=_DeterministicIds(),
        adapter_ref="conformance",
        runtime_version="test-runtime",
    )
    model = dict(input_payload.get("model") or {})
    result = _ask(service, input_payload, model=model)
    repeated_result = None
    if int(input_payload.get("repeat_count") or 1) > 1:
        repeated_result = _ask(service, input_payload, model=model)
    queued_result = None
    second_queued_result = None
    failed_queued_result = None
    if input_payload.get("process_queued_run"):
        execute_count = int(input_payload.get("process_queued_run_count") or 1)
        queued_result = service.run_work.process_queued_run(
            QueuedRunRequest(
                run_id=result.run_id,
                worker_id=str(input_payload.get("worker_id") or "worker_1"),
                active_attempt=int(input_payload.get("active_attempt") or 1),
            )
        )
        if execute_count > 1:
            second_queued_result = service.run_work.process_queued_run(
                QueuedRunRequest(
                    run_id=result.run_id,
                    worker_id=str(input_payload.get("worker_id") or "worker_1"),
                    active_attempt=int(input_payload.get("active_attempt") or 1),
                )
            )
    if input_payload.get("fail_queued_run"):
        failed_queued_result = service.run_work.fail_queued_run(
            FailQueuedRunRequest(
                run_id=result.run_id,
                worker_id=str(input_payload.get("worker_id") or "worker_1"),
                active_attempt=int(input_payload.get("active_attempt") or 1),
                error=str(input_payload.get("worker_error") or "worker_failed"),
            )
        )
    actual = {
        "ask_result": _result_payload(result),
        "repeated_ask_result": _result_payload(repeated_result),
        "queued_result": _result_payload(queued_result),
        "second_queued_result": _result_payload(second_queued_result),
        "failed_queued_result": _result_payload(failed_queued_result),
        "lineage": lineage.summary(),
        "queue": runs.summary(),
        "lookup": lookup.summary(),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _ask(
    service: QuestionService,
    input_payload: dict[str, Any],
    *,
    model: dict[str, Any],
):
    return service.ask(
        AskRequest(
            question=str(input_payload["question"]),
            principal=QuestionPrincipal(
                principal_id=str(input_payload["principal_id"]),
                tenant_id=str(input_payload["tenant_id"]),
                raw=input_payload.get("principal_raw"),
            ),
            execution_mode=input_payload["execution_mode"],
            conversation_id=str(input_payload.get("conversation_id") or ""),
            provider=model.get("provider"),
            model_key=str(model.get("model_key") or ""),
            idempotency_key=input_payload.get("idempotency_key"),
            max_budget_usd=input_payload.get("max_budget_usd"),
            max_thinking_tokens=input_payload.get("max_thinking_tokens"),
            runtime_context=dict(input_payload.get("runtime_context") or {}),
        )
    )


def _result_payload(result: Any | None) -> dict[str, Any] | None:
    return asdict(result) if result is not None else None


def _memory_by_conversation(
    input_payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    memory_context = input_payload.get("conversation_memory_context")
    if not memory_context:
        return {}
    tenant_id = str(input_payload["tenant_id"])
    conversation_id = str(input_payload.get("conversation_id") or "conversation_1")
    return {(tenant_id, conversation_id): dict(memory_context)}


@dataclass
class _DeterministicIds:
    conversation_count: int = 0
    question_count: int = 0
    run_count: int = 0
    clarification_response_count: int = 0

    def new_conversation_id(self) -> str:
        self.conversation_count += 1
        return f"conversation_{self.conversation_count}"

    def new_question_id(self) -> str:
        self.question_count += 1
        return f"question_{self.question_count}"

    def new_run_id(self) -> str:
        self.run_count += 1
        return f"run_{self.run_count}"

    def new_clarification_response_id(self) -> str:
        self.clarification_response_count += 1
        return f"clarification_response_{self.clarification_response_count}"


@dataclass
class _InMemoryLineage:
    conversations: set[tuple[str, str]] = field(default_factory=set)
    question_counts_by_conversation: dict[str, int] = field(default_factory=dict)
    question_count: int = 0
    run_count: int = 0
    failed_runtime_fallback_count: int = 0
    first_question_sequence: int | None = None
    trigger_kind: str | None = None
    memory_by_conversation: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    fail_memory_after_calls: int | None = None
    memory_call_count: int = 0

    def ensure_conversation(self, *, conversation_id: str, tenant_id: str) -> None:
        self.conversations.add((tenant_id, conversation_id))

    def conversation_memory_context(
        self,
        *,
        conversation_id: str,
        authority,
    ) -> dict[str, Any]:
        self.memory_call_count += 1
        if (
            self.fail_memory_after_calls is not None
            and self.memory_call_count > self.fail_memory_after_calls
        ):
            raise RuntimeError("conversation memory should not be read")
        return dict(
            self.memory_by_conversation.get((authority.tenant_id, conversation_id))
            or {}
        )

    def record_question_run(self, record: QuestionRunRecord) -> None:
        if record.question is not None:
            sequence = (
                self.question_counts_by_conversation.get(
                    record.question.conversation_id,
                    0,
                )
                + 1
            )
            self.question_counts_by_conversation[
                record.question.conversation_id
            ] = sequence
            self.question_count += 1
            self.first_question_sequence = self.first_question_sequence or sequence
        self.run_count += 1
        self.trigger_kind = self.trigger_kind or record.run.trigger_kind.value

    def record_failed_runtime_fallback(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        del run_id, status, answer, result_data, error
        self.failed_runtime_fallback_count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "conversation_count": len(self.conversations),
            "question_count": self.question_count,
            "run_count": self.run_count,
            "failed_runtime_fallback_count": self.failed_runtime_fallback_count,
            "memory_call_count": self.memory_call_count,
            "first_question_sequence": self.first_question_sequence,
            "trigger_kind": self.trigger_kind,
        }


@dataclass
class _InMemoryRuns:
    lineage: _InMemoryLineage
    runs: dict[str, QueuedRun] = field(default_factory=dict)
    questions: dict[str, AuthorizedQuestionAccess] = field(default_factory=dict)

    def get_question(
        self,
        *,
        question_id: str,
        authority,
    ) -> AuthorizedQuestionAccess | None:
        question = self.questions.get(question_id)
        if question is None or question.tenant_id != authority.tenant_id:
            return None
        return question

    def authorize_conversation(self, *, conversation_id: str, authority) -> None:
        del conversation_id, authority

    def find_idempotent_run(
        self,
        *,
        authority,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None:
        return self._idempotent_run(
            tenant_id=authority.tenant_id,
            read_context_ref=authority.read_context_ref,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
        )

    def submit_question_run_atomically(
        self,
        *,
        submission: RunSubmission,
        record: QuestionRunRecord,
    ) -> QuestionRunSubmissionResult:
        existing = self._idempotent_run(
            tenant_id=submission.tenant_id,
            read_context_ref=submission.principal.read_context_ref,
            conversation_id=submission.conversation_id,
            idempotency_key=submission.idempotency_key,
        )
        if existing is not None:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.EXISTING,
                run=existing,
            )
        active = self._active_run(
            tenant_id=submission.tenant_id,
            conversation_id=submission.conversation_id,
            read_context_ref=submission.principal.read_context_ref,
        )
        if active is not None:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.ACTIVE_CONFLICT,
                run=active,
            )
        if record.question is not None:
            self.lineage.ensure_conversation(
                conversation_id=record.question.conversation_id,
                tenant_id=record.question.tenant_id,
            )
            self.questions[record.question.question_id] = AuthorizedQuestionAccess._issue(
                question_id=record.question.question_id,
                conversation_id=record.question.conversation_id,
                tenant_id=record.question.tenant_id,
                original_question=record.question.question,
                read_context_ref=submission.principal.read_context_ref,
            )
        self.lineage.record_question_run(record)
        run = QueuedRun(
            submission=submission,
            status="RUNNING"
            if submission.execution_mode is ExecutionMode.INLINE
            else "QUEUED",
        )
        self.runs[submission.run_id] = run
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=run,
        )

    def _idempotent_run(
        self,
        *,
        tenant_id: str,
        read_context_ref,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None:
        if not idempotency_key:
            return None
        for run in self.runs.values():
            submission = run.submission
            if (
                submission.tenant_id == tenant_id
                and submission.principal.read_context_ref == read_context_ref
                and (
                    conversation_id is None
                    or submission.conversation_id == conversation_id
                )
                and submission.idempotency_key == idempotency_key
            ):
                return run
        return None

    def _active_run(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        read_context_ref,
    ) -> QueuedRun | None:
        for run in self.runs.values():
            if run.status not in {"QUEUED", "RUNNING"}:
                continue
            submission = run.submission
            if (
                submission.tenant_id == tenant_id
                and submission.conversation_id == conversation_id
                and submission.principal.read_context_ref == read_context_ref
            ):
                return run
        return None

    def load_executable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        del worker_id, active_attempt
        run = self.runs[run_id]
        if run.status == "QUEUED":
            running = QueuedRun(submission=run.submission, status="RUNNING")
            self.runs[run_id] = running
            return running
        return run

    def load_failable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        del worker_id, active_attempt
        return self.runs[run_id]

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
    ) -> QueuedRun:
        del worker_id, active_attempt
        current = self.runs[run_id]
        terminal = QueuedRun(
            submission=current.submission,
            status=status,
            answer=answer,
            result_data=result_data,
            error=error,
        )
        self.runs[run_id] = terminal
        return terminal

    def summary(self) -> dict[str, Any]:
        return {
            "enqueued_count": len(self.runs),
            "terminal_count": sum(
                1
                for run in self.runs.values()
                if run.status not in {"QUEUED", "RUNNING"}
            ),
        }


@dataclass
class _FakeLookup:
    result_payload: dict[str, Any]
    calls: list[LookupExecutionRequest] = field(default_factory=list)

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del progress_sink
        self.calls.append(request)
        if self.result_payload.get("raise_error"):
            raise RuntimeError(str(self.result_payload.get("error") or "lookup_failed"))
        terminal_lineage_recorded = self.result_payload.get(
            "terminal_lineage_recorded",
            True,
        )
        return LookupExecutionResult(
            status=str(self.result_payload.get("status") or "COMPLETED"),
            answer=self.result_payload.get("answer"),
            result_data=self.result_payload.get("result_data"),
            error=self.result_payload.get("error"),
            terminal_lineage_recorded=bool(terminal_lineage_recorded),
        )

    def summary(self) -> dict[str, Any]:
        last = self.calls[-1] if self.calls else None
        return {
            "call_count": len(self.calls),
            "last_question": last.question if last is not None else None,
            "last_principal": last.principal if last is not None else None,
            "last_conversation_id": last.conversation_id if last is not None else None,
            "last_conversation_context": (
                last.conversation_context if last is not None else None
            ),
            "last_provider": last.provider if last is not None else None,
            "last_model_key": last.model_key if last is not None else None,
            "last_runtime_context": last.runtime_context if last is not None else None,
            "last_max_budget_usd": (
                str(last.max_budget_usd) if last is not None else None
            ),
            "last_max_thinking_tokens": (
                last.max_thinking_tokens if last is not None else None
            ),
            "last_active_attempt": last.active_attempt if last is not None else None,
        }
