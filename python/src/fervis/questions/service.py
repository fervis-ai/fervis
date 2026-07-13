"""Framework-neutral question lifecycle service."""

from __future__ import annotations

import uuid
from dataclasses import replace

from .contracts import (
    AskRequest,
    AskResult,
    ClarificationResponseRequest,
    ExecutionMode,
    RerunQuestionRequest,
    RetryQuestionRequest,
    QuestionLifecycleError,
)
from fervis.lineage.enums import ProgramInvocationKind, QuestionRunKind, RunTriggerKind
from fervis.host_api.contracts.authority import ReadAuthority
from fervis.host_api.credentials import runtime_context_with_delegated_credential
from fervis.run_work.contracts import QueuedRunRequest
from fervis.run_work.events import (
    NullQuestionRunEventSink,
    QuestionRunEventSink,
    run_accepted_event,
    run_active_conflict_event,
    run_result_event,
    run_terminal_event,
)
from fervis.run_work.service import RunWorkService
from fervis.lookup.answer_program.codec import answer_program_id
from fervis.lookup.answer_program.inputs import apply_binding_patch
from fervis.lookup.answer_program.revisions import apply_capability
from fervis.lookup.answer_program.persistence import (
    program_invocation,
    program_invocation_bundle,
    program_revision_bundle,
)
from fervis.lookup.answer_program.rerun import (
    ProgramNotRerunnableError,
    RerunnableProgramInvocation,
)
from .ports import (
    ClarificationRunResume,
    QuestionRunRecord,
    QuestionRunStart,
    QuestionRunSubmissionKind,
    QueuedRun,
    ResolveQuestionRunSpec,
    RerunProgramSpec,
    RunSubmission,
    QuestionIdPort,
    QuestionStart,
    QuestionStateReaderPort,
    QuestionLineagePort,
    QuestionLookupPort,
    QuestionProgramPort,
    QuestionLifecyclePort,
)


def _require_matching_idempotent_ask(
    request: AskRequest,
    *,
    existing: QueuedRun,
) -> None:
    stored = existing.submission
    spec = stored.spec
    if not isinstance(spec, ResolveQuestionRunSpec) or (
        _ask_replay_payload(request)
        != _stored_ask_replay_payload(spec)
    ):
        raise QuestionLifecycleError(
            "idempotency_payload_conflict",
            "idempotency key was already used for a different request",
        )


def _ask_replay_payload(request: AskRequest) -> dict[str, object]:
    return {
        "question": request.question.strip(),
        "provider": request.provider,
        "model_key": request.model_key,
        "context_run_id": request.context_run_id,
        "runtime_context": runtime_context_with_delegated_credential(
            request.runtime_context,
            None,
        ),
        "max_budget_usd": _canonical_optional_number(request.max_budget_usd),
        "max_thinking_tokens": request.max_thinking_tokens,
    }


def _stored_ask_replay_payload(
    spec: ResolveQuestionRunSpec,
) -> dict[str, object]:
    return {
        "question": spec.question,
        "provider": spec.provider,
        "model_key": spec.model_key,
        "context_run_id": spec.context_run_id,
        "runtime_context": runtime_context_with_delegated_credential(
            spec.runtime_context,
            None,
        ),
        "max_budget_usd": _canonical_optional_number(spec.max_budget_usd),
        "max_thinking_tokens": spec.max_thinking_tokens,
    }


def _canonical_optional_number(value: object) -> str | None:
    return None if value is None else str(value)


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
        program: QuestionProgramPort,
        state_reader: QuestionStateReaderPort | None = None,
        ids: QuestionIdPort | None = None,
        adapter_ref: str = "questions",
        runtime_version: str = "development",
    ) -> None:
        self.lineage = lineage
        self.runs = runs
        self.lookup = lookup
        self.program = program
        self.state_reader = state_reader
        self.ids = ids or UuidQuestionIdPort()
        self.adapter_ref = adapter_ref
        self.runtime_version = runtime_version
        self.run_work = RunWorkService(
            lineage=lineage,
            runs=runs,
            lookup=lookup,
            program=program,
        )

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
            principal=request.principal,
            conversation_id=requested_conversation_id or None,
            idempotency_key=request.idempotency_key,
        )
        if existing is not None:
            _require_matching_idempotent_ask(request, existing=existing)
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
            principal=request.principal,
            spec=ResolveQuestionRunSpec(
                question=question,
                provider=request.provider,
                model_key=request.model_key,
                context_run_id=request.context_run_id,
                conversation_context=self.lineage.conversation_memory_context(
                    conversation_id=conversation_id,
                    context_run_id=request.context_run_id,
                    authority=authority,
                ),
                runtime_context=dict(request.runtime_context),
                max_budget_usd=request.max_budget_usd,
                max_thinking_tokens=request.max_thinking_tokens,
            ),
            execution_mode=request.execution_mode,
            idempotency_key=request.idempotency_key,
            idempotency_scope=(
                "" if requested_conversation_id else "new_conversation"
            ),
        )
        record = QuestionRunRecord(
            question=QuestionStart(
                conversation_id=submission.conversation_id,
                tenant_id=submission.tenant_id,
                read_context_ref=authority.read_context_ref,
                question_id=submission.question_id,
                question=question,
                principal_id=submission.principal.principal_id,
            ),
            run=QuestionRunStart(
                question_id=submission.question_id,
                run_id=submission.run_id,
                kind=QuestionRunKind.MODEL_ASSISTED,
                trigger_kind=RunTriggerKind.INITIAL,
                adapter_ref=self.adapter_ref,
                runtime_version=self.runtime_version,
            ),
        )
        submitted = self.runs.submit_question_run_atomically(
            submission=submission,
            record=record,
        )
        if submitted.kind is QuestionRunSubmissionKind.EXISTING:
            _require_matching_idempotent_ask(request, existing=submitted.run)
        return self._finish_submitted_question_run(
            request=request,
            submitted=submitted,
            events=events,
        )

    def respond_to_clarification(
        self,
        request: ClarificationResponseRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult:
        events = event_sink or NullQuestionRunEventSink()
        authority = ReadAuthority.from_principal(request.principal)
        stored = self.runs.get_question(
            question_id=request.question_id,
            authority=authority,
        )
        if stored is None:
            raise ValueError(f"question not found: {request.question_id}")
        submitted = self.runs.resume_question_run_atomically(
            ClarificationRunResume(
                question_id=stored.question_id,
                run_id=request.run_id,
                clarification_id=request.clarification_id,
                response_id=self.ids.new_clarification_response_id(),
                response_text=request.response_text,
                selected_option_id=request.selected_option_id,
                principal=request.principal,
                execution_mode=request.execution_mode,
            )
        )
        return self._finish_submitted_question_run(
            request=request,
            submitted=submitted,
            events=events,
        )

    def retry_question(
        self,
        request: RetryQuestionRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult:
        events = event_sink or NullQuestionRunEventSink()
        question_text = request.question.strip()
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
            principal=request.principal,
            spec=ResolveQuestionRunSpec(
                question=question_text,
                provider=request.provider,
                model_key=request.model_key,
                conversation_context=self.lineage.conversation_memory_context(
                    conversation_id=stored.conversation_id,
                    continuation_run_id=request.base_run_id,
                    authority=authority,
                ),
                runtime_context=dict(request.runtime_context),
                max_budget_usd=request.max_budget_usd,
                max_thinking_tokens=request.max_thinking_tokens,
            ),
            execution_mode=request.execution_mode,
            idempotency_key=request.idempotency_key,
        )
        record = QuestionRunRecord(
            run=QuestionRunStart(
                question_id=stored.question_id,
                run_id=submission.run_id,
                kind=QuestionRunKind.MODEL_ASSISTED,
                trigger_kind=RunTriggerKind.RETRY,
                adapter_ref=self.adapter_ref,
                runtime_version=self.runtime_version,
                base_run_id=request.base_run_id,
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

    def rerun_question(
        self,
        request: RerunQuestionRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult:
        events = event_sink or NullQuestionRunEventSink()
        authority = ReadAuthority.from_principal(request.principal)
        question = self.runs.get_question(
            question_id=request.question_id,
            authority=authority,
        )
        if question is None:
            raise ValueError(f"question not found: {request.question_id}")
        existing = self.runs.find_idempotent_run(
            principal=request.principal,
            conversation_id=question.conversation_id,
            idempotency_key=request.idempotency_key,
        )
        if existing is not None:
            result = self._ask_result_from_queued_run(existing)
            self._emit_result_events(result, events=events)
            return result
        stored_base = self.runs.load_answered_program_invocation(
            run_id=request.base_run_id,
            access=question,
        )
        if stored_base is None:
            raise QuestionLifecycleError(
                "rerun_base_not_reusable",
                "rerun base must be an answered reusable program run",
            )
        try:
            base = RerunnableProgramInvocation.parse(stored_base)
        except ProgramNotRerunnableError as exc:
            raise QuestionLifecycleError(
                "rerun_base_not_reusable",
                "rerun base must be executable without conversation memory",
            ) from exc
        patch = (
            replace(
                request.patch,
                provenance_refs=(
                    f"run:{request.base_run_id}",
                    f"invocation:{base.invocation.invocation_id}",
                ),
            )
            if request.patch is not None
            else None
        )
        revision = (
            apply_capability(
                program=base.program,
                bindings=base.bindings,
                application=request.capability_application,
            )
            if request.capability_application is not None
            else None
        )
        program = revision.program if revision is not None else base.program
        bindings = (
            revision.bindings
            if revision is not None
            else (
                apply_binding_patch(
                    program=program,
                    bindings=base.bindings,
                    patch=patch,
                )
                if patch is not None
                else base.bindings
            )
        )
        run_id = self.ids.new_run_id()
        invocation = program_invocation(
            run_id=run_id,
            program_id=answer_program_id(program),
            bindings=bindings,
            kind=ProgramInvocationKind.RERUN_PROGRAM,
            base_invocation_id=base.invocation.invocation_id,
            patch=patch,
            revision_id=(revision.revision_id if revision is not None else None),
        )
        submission = RunSubmission(
            conversation_id=question.conversation_id,
            tenant_id=question.tenant_id,
            question_id=question.question_id,
            run_id=run_id,
            principal=request.principal,
            spec=RerunProgramSpec(
                invocation_id=invocation.invocation_id,
                runtime_context=dict(request.runtime_context),
            ),
            execution_mode=request.execution_mode,
            idempotency_key=request.idempotency_key,
        )
        record = QuestionRunRecord(
            run=QuestionRunStart(
                question_id=question.question_id,
                run_id=run_id,
                kind=QuestionRunKind.DETERMINISTIC,
                trigger_kind=RunTriggerKind.RERUN,
                adapter_ref=self.adapter_ref,
                runtime_version=self.runtime_version,
                base_run_id=request.base_run_id,
            ),
            program_invocation=program_invocation_bundle(
                program=program,
                invocation=invocation,
            ),
            program_revision=(
                program_revision_bundle(revision=revision)
                if revision is not None
                else None
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

    def _finish_submitted_question_run(
        self,
        *,
        request: (
            AskRequest
            | ClarificationResponseRequest
            | RetryQuestionRequest
            | RerunQuestionRequest
        ),
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
                active_attempt=queued.active_attempt or 1,
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
            duration_ms=executed.duration_ms,
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
            duration_ms=queued.duration_ms,
        )

    def _emit_accepted(
        self,
        queued: QueuedRun,
        *,
        request: (
            AskRequest
            | ClarificationResponseRequest
            | RetryQuestionRequest
            | RerunQuestionRequest
        ),
        events: QuestionRunEventSink,
    ) -> None:
        status = "QUEUED" if queued.status == "QUEUED" else "RUNNING"
        events.emit(
            run_accepted_event(
                conversation_id=queued.submission.conversation_id,
                question_id=queued.submission.question_id,
                run_id=queued.submission.run_id,
                status=status,
                trigger=request.accepted_trigger(),
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
            run_result_event(
                status=result.status,
                run_id=result.run_id,
                question_id=result.question_id,
                conversation_id=result.conversation_id,
                answer=result.answer,
                result_data=result.result_data,
                error=result.error,
            )
        )
