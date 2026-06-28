"""Public run projection helpers for the Django interface."""

from __future__ import annotations

from typing import Any

from fervis.run_work.queue.django.models import (
    RunWorkItem,
    RunWorkStatus,
)
from fervis.lineage import models as lineage_models
from fervis.lineage.enums import FactResultKind, RunResultKind
from fervis.lineage.django.terminal_results import (
    terminal_status_from_result_kind,
)
from fervis.lineage.views.django import DjangoLineageQuery
from fervis.lineage.views.explain import ExplainViewService
from fervis.lineage.views.explanation import answer_explanation_view
from fervis.lineage.views.json_payload import view_json
from fervis.lineage.views.model import (
    AnswerOutputView,
    ClarificationRequestView,
    LineageView,
    QuestionView,
    RunView,
    StepView,
)
from fervis.lineage.views.service import LineageRootNotFound
from fervis.observability.django import DjangoObservabilityQuery
from fervis.observability.usage import (
    ObservabilityRootNotFound,
    RuntimeUsageService,
    usage_payload_from_report,
)


TERMINAL_INTERFACE_STATUSES = frozenset({"COMPLETED", "FAILED", "NEEDS_CLARIFICATION"})


def get_run_view(
    run_id: str,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any] | None:
    run_record = _run_record(run_id, tenant_id=tenant_id)
    if run_record is None:
        return None
    try:
        explain = ExplainViewService(
            lineage_query=DjangoLineageQuery(),
            observability_query=DjangoObservabilityQuery(),
        ).for_run(run_id)
    except LineageRootNotFound:
        return None
    question_view, run_view = _lineage_run(explain.lineage, run_id)
    if question_view is None or run_view is None:
        return None
    return _run_view(
        run_record=run_record,
        question=question_view,
        run=run_view,
        explanation=view_json(answer_explanation_view(explain)),
    )


def with_worker_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    snapshot = _work_item_snapshot_for_run(str(run.get("runId") or ""))
    if snapshot is None:
        return dict(run)
    return {**run, "worker": snapshot}


def with_lineage_usage(run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run.get("runId") or "")
    if not run_id:
        return run
    try:
        usage = _run_usage_payload(run_id)
    except ObservabilityRootNotFound:
        return run
    return {**run, "usage": usage}


def _run_usage_payload(run_id: str) -> dict[str, Any]:
    report = RuntimeUsageService(DjangoObservabilityQuery()).for_run(run_id)
    return dict(usage_payload_from_report(report))


def _run_record(
    run_id: str,
    *,
    tenant_id: str | None,
) -> lineage_models.QuestionRun | None:
    rows = lineage_models.QuestionRun.objects.select_related(
        "question__conversation"
    ).filter(run_id=run_id)
    if tenant_id is not None:
        rows = rows.filter(question__conversation__tenant_id=tenant_id)
    return rows.first()


def _lineage_run(
    lineage: LineageView,
    run_id: str,
) -> tuple[QuestionView | None, RunView | None]:
    for question in lineage.questions:
        for run in question.runs:
            if run.run_id == run_id:
                return question, run
    return None, None


def _run_view(
    *,
    run_record: lineage_models.QuestionRun,
    question: QuestionView,
    run: RunView,
    explanation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "runId": run.run_id,
        "runNumber": run.run_number,
        "triggerKind": run_record.trigger_kind,
        "questionId": run_record.question_id,
        "conversationId": run_record.question.conversation_id,
        "tenantId": run_record.question.conversation.tenant_id,
        "status": _interface_status(run_record, run),
        "question": question.text,
        "answer": _answer_text(run),
        "resultData": _result_data(run),
        "explanation": explanation,
        "modelKey": _work_item_model_key(run.run_id),
        "steps": [_step_view(step) for step in run.steps],
        "error": _run_error(run),
        "guardrail": None,
    }


def _interface_status(
    run_record: lineage_models.QuestionRun,
    run: RunView,
) -> str:
    if run.result_kind != "unknown":
        if _needs_clarification(run):
            return "NEEDS_CLARIFICATION"
        return terminal_status_from_result_kind(run.result_kind)
    work_item = _work_item(run_record.run_id)
    if work_item is None:
        return "RUNNING"
    if work_item.status == RunWorkStatus.FAILED:
        return "FAILED"
    if work_item.status == RunWorkStatus.COMPLETED:
        return "COMPLETED"
    return "RUNNING"


def _needs_clarification(run: RunView) -> bool:
    return any(
        fact_result.result_kind == FactResultKind.NEEDS_CLARIFICATION.value
        for requested_fact in run.requested_facts
        for fact_result in requested_fact.fact_results
    )


def _work_item(run_id: str) -> RunWorkItem | None:
    return RunWorkItem.objects.filter(run_id=run_id).first()


def _work_item_model_key(run_id: str) -> str:
    work_item = _work_item(run_id)
    return work_item.model_key if work_item is not None else ""


def _work_item_snapshot_for_run(run_id: str) -> dict[str, Any] | None:
    item = _work_item(run_id)
    if item is None:
        return None
    return {
        "status": item.status,
        "attemptCount": item.attempt_count,
        "activeAttempt": item.active_attempt,
        "leaseOwner": item.lease_owner,
        "leaseExpiresAt": item.lease_expires_at.isoformat()
        if item.lease_expires_at
        else None,
        "lastError": item.last_error,
        "createdAt": item.created_at.isoformat() if item.created_at else None,
        "startedAt": item.started_at.isoformat() if item.started_at else None,
        "completedAt": item.completed_at.isoformat() if item.completed_at else None,
    }


def _answer_text(run: RunView) -> str | None:
    for answer in run.answers:
        for presentation in answer.presentations:
            if presentation.value:
                return presentation.value
    values = [
        output.value
        for answer in run.answers
        for output in answer.outputs
        if output.value
    ]
    if values:
        return "\n".join(values)
    return None


def _result_data(run: RunView) -> dict[str, Any] | None:
    if run.result_kind == RunResultKind.RUNTIME_ERROR.value:
        return None
    clarification = _clarification_result(run)
    if clarification is not None:
        return clarification
    outputs = [
        _answer_output_data(output)
        for answer in run.answers
        for output in answer.outputs
    ]
    if outputs:
        return {"kind": "answer", "outputs": outputs}
    return None


def _clarification_result(run: RunView) -> dict[str, Any] | None:
    if not run.clarification_requests:
        return None
    return {
        "kind": "needs_clarification",
        "details": {
            "clarifications": [
                _clarification_data(request) for request in run.clarification_requests
            ],
        },
    }


def _clarification_data(request: ClarificationRequestView) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": request.clarification_id,
        "basis": request.basis,
        "question": request.question_text,
        "factResultId": request.fact_result_id,
        "stepId": request.step_id,
    }
    if request.options:
        payload["availableOptions"] = list(request.options)
    if request.evidence_refs:
        payload["evidenceRefs"] = list(request.evidence_refs)
    return payload


def _answer_output_data(output: AnswerOutputView) -> dict[str, Any]:
    return {
        "key": output.output_key,
        "valueKind": output.value_kind,
        "value": output.value,
    }


def _run_error(run: RunView) -> str | None:
    if not run.runtime_errors:
        return None
    error = run.runtime_errors[0]
    return error.message or error.error_kind


def _step_view(step: StepView) -> dict[str, Any]:
    error = _step_error(step)
    return {
        "stepId": step.step_id,
        "stepKey": step.step_key,
        "semantic": view_json(step.semantic),
        "stepType": step.step_key,
        "toolName": step.step_key,
        "statusCode": 500 if error else 200,
        "requestBody": {},
        "responseBody": {"errorCode": error} if error else _step_response(step),
    }


def _step_error(step: StepView) -> str:
    if step.error:
        return str(
            step.error.get("errorCode")
            or step.error.get("error")
            or step.error.get("message")
            or ""
        )
    for decision in step.decisions:
        for line in decision.lines:
            if "failed" in line.lower() or "error" in line.lower():
                return line
    return ""


def _step_response(step: StepView) -> dict[str, Any]:
    if not step.decisions:
        return {}
    return {
        "decisions": [line for decision in step.decisions for line in decision.lines]
    }
