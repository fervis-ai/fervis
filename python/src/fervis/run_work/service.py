"""Framework-neutral queued run work service."""

from __future__ import annotations

from fervis.questions.ports import (
    RerunProgramSpec,
    LookupExecutionRequest,
    LookupExecutionResult,
    ResolveQuestionRunSpec,
    ProgramExecutionRequest,
    QueuedRun,
    RunSubmission,
    RunExecutionSpec,
    fold_run_execution_spec,
    QuestionLifecyclePort,
    QuestionLineagePort,
    QuestionLookupPort,
    QuestionProgramPort,
)
from fervis.host_api.contracts.authority import ReadAuthority

from .contracts import FailQueuedRunRequest, QueuedRunRequest, QueuedRunResult
from .events import (
    NullQuestionRunEventSink,
    QuestionRunEventSink,
    run_progress_event,
    run_result_event,
)


class RunWorkService:
    def __init__(
        self,
        *,
        lineage: QuestionLineagePort,
        runs: QuestionLifecyclePort,
        lookup: QuestionLookupPort,
        program: QuestionProgramPort,
    ) -> None:
        self.lineage = lineage
        self.runs = runs
        self.lookup = lookup
        self.program = program

    def process_queued_run(
        self,
        request: QueuedRunRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> QueuedRunResult:
        events = event_sink or NullQuestionRunEventSink()
        queued = self.runs.load_executable_run(
            run_id=request.run_id,
            worker_id=request.worker_id,
            active_attempt=request.active_attempt,
        )
        if queued.status not in {"QUEUED", "RUNNING"}:
            result = _queued_result(queued)
            _emit_queued_result_events(
                result,
                conversation_id=queued.submission.conversation_id,
                question_id=queued.submission.question_id,
                events=events,
            )
            return result
        active_attempt = request.active_attempt
        if active_attempt is None:
            raise ValueError("active_attempt is required for queued run work")
        try:
            stage, message = _run_start_progress(queued.submission.spec)
            events.emit(
                run_progress_event(
                    run_id=queued.submission.run_id,
                    stage=stage,
                    message=message,
                )
            )
            lookup_result = self._execute_submission(
                queued.submission,
                active_attempt=active_attempt,
                event_sink=events,
            )
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            self.lineage.record_failed_runtime_fallback(
                run_id=queued.submission.run_id,
                status="FAILED",
                answer=None,
                result_data=None,
                error=error,
            )
            terminal = self.runs.terminalize(
                run_id=queued.submission.run_id,
                status="FAILED",
                answer=None,
                result_data=None,
                error=error,
                worker_id=request.worker_id,
                active_attempt=active_attempt,
            )
            result = _queued_result(terminal)
            _emit_queued_result_events(
                result,
                conversation_id=queued.submission.conversation_id,
                question_id=queued.submission.question_id,
                events=events,
            )
            return result
        if lookup_result.status == "NEEDS_CLARIFICATION":
            waiting = self.runs.wait_for_clarification(
                run_id=queued.submission.run_id,
                worker_id=request.worker_id,
                active_attempt=active_attempt,
            )
            result = _queued_result(waiting)
            events.emit(
                run_result_event(
                    status=result.status,
                    run_id=result.run_id,
                    conversation_id=queued.submission.conversation_id,
                    question_id=queued.submission.question_id,
                    result_data=dict(result.result_data or {}),
                )
            )
            return result
        lookup_result = self._ensure_terminal_lineage_or_failure(
            queued.submission.run_id,
            lookup_result,
        )
        terminal = self.runs.terminalize(
            run_id=queued.submission.run_id,
            status=lookup_result.status,
            answer=lookup_result.answer,
            result_data=lookup_result.result_data,
            error=lookup_result.error,
            worker_id=request.worker_id,
                active_attempt=active_attempt,
        )
        result = _queued_result(terminal)
        _emit_queued_result_events(
            result,
            conversation_id=queued.submission.conversation_id,
            question_id=queued.submission.question_id,
            events=events,
        )
        return result

    def fail_queued_run(self, request: FailQueuedRunRequest) -> QueuedRunResult:
        queued = self.runs.load_failable_run(
            run_id=request.run_id,
            worker_id=request.worker_id,
            active_attempt=request.active_attempt,
        )
        if queued.status not in {"QUEUED", "RUNNING"}:
            return _queued_result(queued)
        error = request.error or "run_worker_failed"
        self.lineage.record_failed_runtime_fallback(
            run_id=queued.submission.run_id,
            status="FAILED",
            answer=None,
            result_data=None,
            error=error,
        )
        terminal = self.runs.terminalize(
            run_id=queued.submission.run_id,
            status="FAILED",
            answer=None,
            result_data=None,
            error=error,
            worker_id=request.worker_id,
            active_attempt=request.active_attempt,
        )
        return _queued_result(terminal)

    def _run_lookup(
        self,
        submission: RunSubmission,
        *,
        spec: ResolveQuestionRunSpec,
        active_attempt: int | None,
        event_sink: QuestionRunEventSink | None = None,
    ) -> LookupExecutionResult:
        return self.lookup.run_lookup(
            LookupExecutionRequest(
                run_id=submission.run_id,
                conversation_id=submission.conversation_id,
                tenant_id=submission.tenant_id,
                question=spec.question,
                read_context_ref=submission.principal.read_context_ref,
                delegated_credential=submission.principal.delegated_credential,
                principal=(
                    submission.principal.raw
                    if submission.principal.raw is not None
                    else submission.principal.principal_id
                ),
                provider=spec.provider,
                model_key=spec.model_key,
                conversation_context=dict(spec.conversation_context),
                runtime_context=dict(spec.runtime_context),
                max_budget_usd=spec.max_budget_usd,
                max_thinking_tokens=spec.max_thinking_tokens,
                active_attempt=active_attempt,
                clarification_responses=spec.clarification_responses,
            ),
            progress_sink=event_sink,
        )

    def _execute_submission(
        self,
        submission: RunSubmission,
        *,
        active_attempt: int | None,
        event_sink: QuestionRunEventSink | None = None,
    ) -> LookupExecutionResult:
        return fold_run_execution_spec(
            submission.spec,
            resolve_question=lambda spec: self._run_lookup(
                submission,
                spec=spec,
                active_attempt=active_attempt,
                event_sink=event_sink,
            ),
            rerun_program=lambda spec: self._run_program(
                submission,
                spec=spec,
                active_attempt=active_attempt,
                event_sink=event_sink,
            ),
        )

    def _run_program(
        self,
        submission: RunSubmission,
        *,
        spec: RerunProgramSpec,
        active_attempt: int | None,
        event_sink: QuestionRunEventSink | None = None,
    ) -> LookupExecutionResult:
        authority = ReadAuthority.from_principal(submission.principal)
        question = self.runs.get_question(
            question_id=submission.question_id,
            authority=authority,
        )
        if question is None:
            raise PermissionError("deterministic run question is not authorized")
        invocation = self.runs.load_program_invocation_for_execution(
            invocation_id=spec.invocation_id,
            run_id=submission.run_id,
            question_id=submission.question_id,
            tenant_id=submission.tenant_id,
        )
        if invocation is None:
            raise RuntimeError("deterministic run invocation is not executable")
        return self.program.run_program(
            ProgramExecutionRequest(
                run_id=submission.run_id,
                conversation_id=submission.conversation_id,
                tenant_id=submission.tenant_id,
                question=question.original_question,
                read_context_ref=submission.principal.read_context_ref,
                delegated_credential=submission.principal.delegated_credential,
                principal=(
                    submission.principal.raw
                    if submission.principal.raw is not None
                    else submission.principal.principal_id
                ),
                invocation=invocation,
                runtime_context=dict(spec.runtime_context),
                active_attempt=active_attempt,
            ),
            progress_sink=event_sink,
        )

    def _record_failed_runtime_fallback(
        self,
        run_id: str,
        lookup_result: LookupExecutionResult,
    ) -> None:
        if lookup_result.terminal_lineage_recorded:
            return
        if lookup_result.status != "FAILED":
            raise RuntimeError(
                f"lookup completed without terminal lineage for {run_id}: "
                f"{lookup_result.status}"
            )
        self.lineage.record_failed_runtime_fallback(
            run_id=run_id,
            status=lookup_result.status,
            answer=lookup_result.answer,
            result_data=lookup_result.result_data,
            error=lookup_result.error,
        )

    def _ensure_terminal_lineage_or_failure(
        self,
        run_id: str,
        lookup_result: LookupExecutionResult,
    ) -> LookupExecutionResult:
        try:
            self._record_failed_runtime_fallback(run_id, lookup_result)
            return lookup_result
        except RuntimeError as exc:
            error = str(exc) or exc.__class__.__name__
            self.lineage.record_failed_runtime_fallback(
                run_id=run_id,
                status="FAILED",
                answer=None,
                result_data=None,
                error=error,
            )
            return LookupExecutionResult(
                status="FAILED",
                answer=None,
                result_data=None,
                error=error,
                terminal_lineage_recorded=True,
            )


def _queued_result(queued: QueuedRun) -> QueuedRunResult:
    return QueuedRunResult(
        status=queued.status,
        run_id=queued.submission.run_id,
        answer=queued.answer,
        result_data=queued.result_data,
        error=queued.error,
        duration_ms=queued.duration_ms,
    )


def _run_start_progress(spec: RunExecutionSpec) -> tuple[str, str]:
    return fold_run_execution_spec(
        spec,
        resolve_question=lambda _spec: ("lookup", "starting lookup"),
        rerun_program=lambda _spec: (
            "program",
            "starting deterministic program",
        ),
    )


def _emit_queued_result_events(
    result: QueuedRunResult,
    *,
    conversation_id: str,
    question_id: str,
    events: QuestionRunEventSink,
) -> None:
    events.emit(
        run_result_event(
            status=result.status,
            run_id=result.run_id,
            question_id=question_id,
            conversation_id=conversation_id,
            answer=result.answer,
            result_data=result.result_data,
            error=result.error,
        )
    )
