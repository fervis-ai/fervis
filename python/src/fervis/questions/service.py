"""Framework-neutral question lifecycle service."""

from __future__ import annotations

import uuid

from .contracts import (
    AskRequest,
    AskResult,
    ContinueQuestionRequest,
    ExecutionMode,
)
from fervis.lineage.enums import RunTriggerKind
from fervis.host_api.contracts.authority import ReadAuthority
from fervis.run_work.contracts import QueuedRunRequest
from fervis.run_work.events import (
    NullQuestionRunEventSink,
    QuestionRunEventSink,
    run_accepted_event,
    run_active_conflict_event,
    run_terminal_event,
)
from fervis.run_work.service import RunWorkService
from .ports import (
    ClarificationResponseStart,
    QuestionRunRecord,
    QuestionRunStart,
    QuestionRunSubmissionKind,
    QueuedRun,
    RunSubmission,
    QuestionIdPort,
    QuestionStart,
    QuestionStateReaderPort,
    QuestionLineagePort,
    QuestionLookupPort,
    QuestionLifecyclePort,
)


class UuidQuestionIdPort:
    def new_conversation_id(self) -> str:
        return str(uuid.uuid4())

    def new_question_id(self) -> str:
        return str(uuid.uuid4())

    def new_run_id(self) -> str:
        return str(uuid.uuid4())

    def new_clarification_response_id(self) -> str:
        return str(uuid.uuid4())


class QuestionService:
    def __init__(
        self,
        *,
        lineage: QuestionLineagePort,
        runs: QuestionLifecyclePort,
        lookup: QuestionLookupPort,
        state_reader: QuestionStateReaderPort | None = None,
        ids: QuestionIdPort | None = None,
        adapter_ref: str = "questions",
        runtime_version: str = "development",
    ) -> None:
        self.lineage = lineage
        self.runs = runs
        self.lookup = lookup
        self.state_reader = state_reader
        self.ids = ids or UuidQuestionIdPort()
        self.adapter_ref = adapter_ref
        self.runtime_version = runtime_version
        self.run_work = RunWorkService(lineage=lineage, runs=runs, lookup=lookup)

    def ask(
        self,
        request: AskRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult:
        events = event_sink or NullQuestionRunEventSink()
        question = request.question.strip()
        if not question:
            raise ValueError("ask request question must not be empty")

        requested_conversation_id = request.conversation_id.strip()
        authority = ReadAuthority.from_principal(request.principal)
        existing = self.runs.find_idempotent_run(
            authority=authority,
            conversation_id=requested_conversation_id or None,
            idempotency_key=request.idempotency_key,
        )
        if existing is not None:
            result = self._ask_result_from_queued_run(existing)
            self._emit_result_events(result, events=events)
            return result

        conversation_id = requested_conversation_id or self.ids.new_conversation_id()
        if requested_conversation_id:
            self.runs.authorize_conversation(
                conversation_id=conversation_id,
                authority=authority,
            )
        submission = RunSubmission(
            conversation_id=conversation_id,
            tenant_id=authority.tenant_id,
            question_id=self.ids.new_question_id(),
            run_id=self.ids.new_run_id(),
            question=question,
            principal=request.principal,
            provider=request.provider,
            model_key=request.model_key,
            execution_mode=request.execution_mode,
            conversation_context=self.lineage.conversation_memory_context(
                conversation_id=conversation_id,
                authority=authority,
            ),
            runtime_context=dict(request.runtime_context),
            idempotency_key=request.idempotency_key,
            max_budget_usd=request.max_budget_usd,
            max_thinking_tokens=request.max_thinking_tokens,
        )
        record = QuestionRunRecord(
            question=QuestionStart(
                conversation_id=submission.conversation_id,
                tenant_id=submission.tenant_id,
                read_context_ref=authority.read_context_ref,
                question_id=submission.question_id,
                question=submission.question,
                principal_id=submission.principal.principal_id,
            ),
            run=QuestionRunStart(
                question_id=submission.question_id,
                run_id=submission.run_id,
                trigger_kind=RunTriggerKind.INITIAL,
                integrated_question=submission.question,
                adapter_ref=self.adapter_ref,
                runtime_version=self.runtime_version,
            ),
        )
        submitted = self.runs.submit_question_run_atomically(
            submission=submission,
            record=record,
        )
        return self._finish_submitted_question_run(
            request=request,
            submitted=submitted,
            events=events,
        )

    def continue_question(
        self,
        request: ContinueQuestionRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult:
        events = event_sink or NullQuestionRunEventSink()
        question_text = request.question.strip()
        if not question_text:
            raise ValueError("continue request question must not be empty")
        if request.trigger_kind is RunTriggerKind.INITIAL:
            raise ValueError("question continuation trigger must not be initial")
        authority = ReadAuthority.from_principal(request.principal)
        stored = self.runs.get_question(
            question_id=request.question_id,
            authority=authority,
        )
        if stored is None:
            raise ValueError(f"question not found: {request.question_id}")
        submission = RunSubmission(
            conversation_id=stored.conversation_id,
            tenant_id=request.principal.tenant_id,
            question_id=stored.question_id,
            run_id=self.ids.new_run_id(),
            question=question_text,
            principal=request.principal,
            provider=request.provider,
            model_key=request.model_key,
            execution_mode=request.execution_mode,
            conversation_context=self.lineage.conversation_memory_context(
                conversation_id=stored.conversation_id,
                authority=authority,
            ),
            runtime_context=dict(request.runtime_context),
            idempotency_key=request.idempotency_key,
            max_budget_usd=request.max_budget_usd,
            max_thinking_tokens=request.max_thinking_tokens,
        )
        clarification_response = _clarification_response_start(
            request,
            response_id=self.ids.new_clarification_response_id(),
        )
        record = QuestionRunRecord(
            run=QuestionRunStart(
                question_id=stored.question_id,
                run_id=submission.run_id,
                trigger_kind=request.trigger_kind,
                integrated_question=question_text,
                adapter_ref=self.adapter_ref,
                runtime_version=self.runtime_version,
                previous_run_id=request.previous_run_id,
                trigger_clarification_response_run_id=(
                    request.trigger_clarification_response_run_id
                ),
                trigger_clarification_response_id=(
                    clarification_response.response_id
                    if clarification_response is not None
                    else request.trigger_clarification_response_id
                ),
            ),
            clarification_response=clarification_response,
        )
        submitted = self.runs.submit_question_run_atomically(
            submission=submission,
            record=record,
        )
        return self._finish_submitted_question_run(
            request=request,
            submitted=submitted,
            events=events,
        )

    def _finish_submitted_question_run(
        self,
        *,
        request: AskRequest | ContinueQuestionRequest,
        submitted,
        events: QuestionRunEventSink,
    ) -> AskResult:
        queued = submitted.run
        if submitted.kind is QuestionRunSubmissionKind.EXISTING:
            result = self._ask_result_from_queued_run(queued)
            self._emit_result_events(result, events=events)
            return result
        if submitted.kind is QuestionRunSubmissionKind.ACTIVE_CONFLICT:
            result = AskResult(
                status="ACTIVE_RUN_CONFLICT",
                conversation_id=queued.submission.conversation_id,
                question_id=queued.submission.question_id,
                run_id=queued.submission.run_id,
                active_run_id=queued.submission.run_id,
                error="active_run_conflict",
            )
            events.emit(
                run_active_conflict_event(
                    conversation_id=result.conversation_id,
                    question_id=result.question_id,
                    run_id=result.run_id,
                    active_run_id=result.active_run_id or result.run_id,
                    error=result.error,
                )
            )
            return result
        if submitted.kind is not QuestionRunSubmissionKind.CREATED:
            raise ValueError(f"unsupported question run submission: {submitted.kind}")
        self._emit_accepted(
            queued,
            request=request,
            events=events,
        )
        if request.execution_mode is ExecutionMode.QUEUED:
            result = self._ask_result_from_queued_run(queued)
            events.emit(
                run_terminal_event(
                    status=result.status,
                    run_id=result.run_id,
                    question_id=result.question_id,
                    conversation_id=result.conversation_id,
                    answer=result.answer,
                    result_data=result.result_data,
                    error=result.error,
                )
            )
            return result

        executed = self.run_work.process_queued_run(
            QueuedRunRequest(
                run_id=queued.submission.run_id,
                worker_id="inline",
                active_attempt=1,
            ),
            event_sink=events,
        )
        return AskResult(
            status=executed.status,
            conversation_id=queued.submission.conversation_id,
            question_id=queued.submission.question_id,
            run_id=executed.run_id,
            answer=executed.answer,
            result_data=executed.result_data,
            error=executed.error,
        )

    def get_question_state(
        self,
        question_id: str,
        *,
        principal,
    ) -> dict[str, object] | None:
        access = self.runs.get_question(
            question_id=question_id,
            authority=ReadAuthority.from_principal(principal),
        )
        if access is None:
            return None
        return self._state_reader().get_question_state(
            access=access,
        )

    def list_conversations(
        self,
        *,
        principal,
    ) -> list[dict[str, object]]:
        return self._state_reader().list_conversations(
            authority=ReadAuthority.from_principal(principal),
        )

    def list_question_runs(
        self,
        question_id: str,
        *,
        principal,
    ) -> list[dict[str, object]]:
        access = self.runs.get_question(
            question_id=question_id,
            authority=ReadAuthority.from_principal(principal),
        )
        if access is None:
            return []
        return self._state_reader().list_question_runs(
            access=access,
        )

    def get_question_run(
        self,
        question_id: str,
        run_id: str,
        *,
        principal,
    ) -> dict[str, object] | None:
        access = self.runs.get_question(
            question_id=question_id,
            authority=ReadAuthority.from_principal(principal),
        )
        if access is None:
            return None
        return self._state_reader().get_question_run(
            run_id,
            access=access,
        )

    def _ask_result_from_queued_run(
        self,
        queued: QueuedRun,
    ) -> AskResult:
        return AskResult(
            status=queued.status,
            conversation_id=queued.submission.conversation_id,
            question_id=queued.submission.question_id,
            run_id=queued.submission.run_id,
            answer=queued.answer,
            result_data=queued.result_data,
            error=queued.error,
        )

    def _emit_accepted(
        self,
        queued: QueuedRun,
        *,
        request: AskRequest | ContinueQuestionRequest,
        events: QuestionRunEventSink,
    ) -> None:
        status = "QUEUED" if queued.status == "QUEUED" else "RUNNING"
        events.emit(
            run_accepted_event(
                conversation_id=queued.submission.conversation_id,
                question_id=queued.submission.question_id,
                run_id=queued.submission.run_id,
                status=status,
                trigger=_accepted_trigger(request),
            )
        )

    def _state_reader(self) -> QuestionStateReaderPort:
        if self.state_reader is None:
            raise RuntimeError("question state reader is not configured")
        return self.state_reader

    def _emit_result_events(
        self,
        result: AskResult,
        *,
        events: QuestionRunEventSink,
    ) -> None:
        if result.status == "ACTIVE_RUN_CONFLICT":
            events.emit(
                run_active_conflict_event(
                    conversation_id=result.conversation_id,
                    question_id=result.question_id,
                    run_id=result.run_id,
                    active_run_id=result.active_run_id or result.run_id,
                    error=result.error,
                )
            )
            return
        events.emit(
            run_terminal_event(
                status=result.status,
                run_id=result.run_id,
                question_id=result.question_id,
                conversation_id=result.conversation_id,
                answer=result.answer,
                result_data=result.result_data,
                error=result.error,
            )
        )


def _accepted_trigger(
    request: AskRequest | ContinueQuestionRequest,
) -> dict[str, object] | None:
    if not isinstance(request, ContinueQuestionRequest):
        return None
    trigger: dict[str, object] = {
        "kind": request.trigger_kind.value,
    }
    previous_run_id = (
        request.trigger_clarification_response_run_id
        if request.trigger_kind is RunTriggerKind.CLARIFICATION_RESPONSE
        else request.previous_run_id
    )
    if previous_run_id:
        trigger["previous_run_id"] = previous_run_id
    if request.trigger_clarification_response_id:
        trigger["clarification_id"] = request.trigger_clarification_response_id
    return trigger


def _clarification_response_start(
    request: ContinueQuestionRequest,
    *,
    response_id: str,
) -> ClarificationResponseStart | None:
    if request.trigger_kind is not RunTriggerKind.CLARIFICATION_RESPONSE:
        return None
    trigger_run_id = request.trigger_clarification_response_run_id
    clarification_id = request.trigger_clarification_response_id
    if not trigger_run_id or not clarification_id:
        return None
    return ClarificationResponseStart(
        response_id=response_id,
        run_id=trigger_run_id,
        clarification_id=clarification_id,
        response_text=request.question,
        selected_option_id=request.trigger_clarification_selected_option_id or "",
    )
