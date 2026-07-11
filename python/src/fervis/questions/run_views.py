"""Canonical public projection of persisted question runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from fervis.lineage.enums import RunResultKind
from fervis.lineage.views.explain import ExplainViewService
from fervis.lineage.views.explanation import answer_explanation_view
from fervis.lineage.views.json_payload import view_json
from fervis.lineage.views.model import QuestionView, RunView
from fervis.lineage.views.query import LineageQueryPort
from fervis.observability.query import ObservabilityQueryPort
from fervis.observability.usage import RuntimeUsageService, usage_payload_from_report

from .projection import QuestionRunStatus


@dataclass(frozen=True)
class RunWorkSnapshot:
    run_id: str
    conversation_id: str
    tenant_id: str
    status: QuestionRunStatus
    question_override: str | None
    model_key: str
    attempt_count: int
    active_attempt: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    last_error: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class RunWorkQueryPort(Protocol):
    def for_run(self, run_id: str) -> RunWorkSnapshot | None: ...


@dataclass(frozen=True)
class ProjectedQuestionRun:
    run_id: str
    question_id: str
    conversation_id: str
    tenant_id: str
    run_number: int
    kind: str
    trigger_kind: str
    base_run_id: str | None
    program_id: str | None
    invocation_id: str | None
    execution_kind: str | None
    base_invocation_id: str | None
    patch_id: str | None
    revision_id: str | None
    status: QuestionRunStatus
    question: str
    answer: str | None
    result_data: dict[str, object] | None
    explanation: dict[str, object]
    model_key: str
    duration_ms: int | None
    steps: tuple[dict[str, object], ...]
    error: str | None
    worker: dict[str, object]
    usage: dict[str, object]

    def to_payload(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "questionId": self.question_id,
            "conversationId": self.conversation_id,
            "tenantId": self.tenant_id,
            "runNumber": self.run_number,
            "kind": self.kind,
            "triggerKind": self.trigger_kind,
            "baseRunId": self.base_run_id,
            "programId": self.program_id,
            "invocationId": self.invocation_id,
            "executionKind": self.execution_kind,
            "baseInvocationId": self.base_invocation_id,
            "patchId": self.patch_id,
            "revisionId": self.revision_id,
            "status": self.status.value,
            "question": self.question,
            "answer": self.answer,
            "resultData": self.result_data,
            "explanation": self.explanation,
            "modelKey": self.model_key,
            "durationMs": self.duration_ms,
            "steps": list(self.steps),
            "error": self.error,
            "guardrail": None,
            "worker": self.worker,
            "usage": self.usage,
        }


class QuestionRunViewService:
    def __init__(
        self,
        *,
        lineage_query: LineageQueryPort,
        work_query: RunWorkQueryPort,
        observability_query: ObservabilityQueryPort,
    ) -> None:
        self._work_query = work_query
        self._explain = ExplainViewService(
            lineage_query=lineage_query,
            observability_query=observability_query,
        )
        self._usage = RuntimeUsageService(observability_query)

    def for_run(self, run_id: str) -> ProjectedQuestionRun | None:
        work = self._work_query.for_run(run_id)
        if work is None:
            return None
        explain = self._explain.for_run(run_id)
        located = _question_run(explain.lineage.questions, run_id)
        if located is None:
            return None
        question, run = located
        return project_question_run(
            question=question,
            run=run,
            work=work,
            explanation=view_json(answer_explanation_view(explain)),
            usage=dict(usage_payload_from_report(self._usage.for_run(run_id))),
        )


def project_question_run(
    *,
    question: QuestionView,
    run: RunView,
    work: RunWorkSnapshot,
    explanation: dict[str, object],
    usage: dict[str, object],
) -> ProjectedQuestionRun:
    derivation = run.program_derivation
    revision = derivation.revision if derivation is not None else None
    return ProjectedQuestionRun(
        run_id=run.run_id,
        question_id=question.question_id,
        conversation_id=question.conversation_id,
        tenant_id=work.tenant_id,
        run_number=run.run_number,
        kind=run.kind,
        trigger_kind=run.trigger_kind,
        base_run_id=run.base_run_id,
        program_id=derivation.program.program_id if derivation is not None else None,
        invocation_id=derivation.invocation_id if derivation is not None else None,
        execution_kind=derivation.kind if derivation is not None else None,
        base_invocation_id=(
            derivation.base_invocation_id if derivation is not None else None
        ),
        patch_id=(
            derivation.patch.patch_id
            if derivation is not None and derivation.patch is not None
            else None
        ),
        revision_id=revision.revision_id if revision is not None else None,
        status=_status(run, work),
        question=work.question_override or question.text,
        answer=_answer_text(run),
        result_data=_result_data(run),
        explanation=explanation,
        model_key=work.model_key,
        duration_ms=_duration_ms(work),
        steps=tuple(view_json(step) for step in run.steps),
        error=_run_error(run, work),
        worker=_worker_payload(work),
        usage=usage,
    )


def _question_run(
    questions: tuple[QuestionView, ...],
    run_id: str,
) -> tuple[QuestionView, RunView] | None:
    for question in questions:
        for run in question.runs:
            if run.run_id == run_id:
                return question, run
    return None


def _status(run: RunView, work: RunWorkSnapshot) -> QuestionRunStatus:
    if run.result_kind == RunResultKind.ANSWERED.value:
        return QuestionRunStatus.COMPLETED
    if run.result_kind == RunResultKind.RUNTIME_ERROR.value:
        return QuestionRunStatus.FAILED
    if run.result_kind == RunResultKind.FACTUAL_TERMINAL.value:
        if run.clarification_requests:
            return QuestionRunStatus.NEEDS_CLARIFICATION
        return QuestionRunStatus.COMPLETED
    return work.status


def _answer_text(run: RunView) -> str | None:
    for answer in run.answers:
        for presentation in answer.presentations:
            if presentation.value:
                return presentation.value
    values = tuple(
        output.value
        for answer in run.answers
        for output in answer.outputs
        if output.value
    )
    return "\n".join(values) if values else None


def _result_data(run: RunView) -> dict[str, object] | None:
    if run.clarification_requests:
        return {
            "kind": "needs_clarification",
            "details": {
                "clarifications": [
                    dict(request.payload_json)
                    for request in run.clarification_requests
                ]
            },
        }
    outputs = [
        {
            "key": output.output_key,
            "valueKind": output.value_kind,
            "value": output.value,
        }
        for answer in run.answers
        for output in answer.outputs
    ]
    return {"kind": "answer", "outputs": outputs} if outputs else None


def _run_error(run: RunView, work: RunWorkSnapshot) -> str | None:
    if run.runtime_errors:
        error = run.runtime_errors[0]
        return error.message or error.error_kind
    return work.last_error or None


def _duration_ms(work: RunWorkSnapshot) -> int | None:
    if work.started_at is None or work.completed_at is None:
        return None
    return max(0, int((work.completed_at - work.started_at).total_seconds() * 1000))


def _worker_payload(work: RunWorkSnapshot) -> dict[str, object]:
    return {
        "status": work.status.value,
        "attemptCount": work.attempt_count,
        "activeAttempt": work.active_attempt,
        "leaseOwner": work.lease_owner,
        "leaseExpiresAt": _iso_datetime(work.lease_expires_at),
        "lastError": work.last_error or None,
        "createdAt": work.created_at.isoformat(),
        "startedAt": _iso_datetime(work.started_at),
        "completedAt": _iso_datetime(work.completed_at),
    }


def _iso_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
