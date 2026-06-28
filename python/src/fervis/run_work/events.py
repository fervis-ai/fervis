"""Question-run lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from fervis.questions.result_data import result_data_clarifications


QuestionRunEventPayload = Mapping[str, object]


class QuestionRunEventSink(Protocol):
    def emit(self, event: QuestionRunEventPayload) -> None: ...


class NullQuestionRunEventSink:
    def emit(self, event: QuestionRunEventPayload) -> None:
        del event


@dataclass
class CollectingQuestionRunEventSink:
    events: list[dict[str, object]] = field(default_factory=list)

    def emit(self, event: QuestionRunEventPayload) -> None:
        self.events.append(dict(event))


class ForwardingQuestionRunEventSink:
    def __init__(self, *sinks: QuestionRunEventSink) -> None:
        self._sinks = tuple(sinks)

    def emit(self, event: QuestionRunEventPayload) -> None:
        for sink in self._sinks:
            sink.emit(event)


def run_accepted_event(
    *,
    conversation_id: str,
    question_id: str,
    run_id: str,
    status: str,
    trigger: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "event": "run.accepted",
        "conversation_id": conversation_id,
        "question_id": question_id,
        "run_id": run_id,
        "status": status,
    }
    if trigger:
        event["trigger"] = dict(trigger)
    return event


def run_progress_event(
    *,
    run_id: str,
    stage: str,
    message: str,
) -> dict[str, object]:
    return {
        "event": "run.progress",
        "run_id": run_id,
        "stage": stage,
        "message": message,
    }


def run_terminal_event(
    *,
    status: str,
    run_id: str,
    question_id: str | None = None,
    conversation_id: str | None = None,
    answer: str | None = None,
    result_data: dict[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    if status == "COMPLETED":
        return {
            "event": "run.completed",
            "run_id": run_id,
            "question_id": question_id or "",
            "conversation_id": conversation_id or "",
            "status": status,
            "answer": answer,
            "result_data": result_data or {},
        }
    if status == "NEEDS_CLARIFICATION":
        clarifications = _actionable_clarifications(result_data)
        return {
            "event": "run.needs_clarification",
            "conversation_id": conversation_id or "",
            "question_id": question_id or "",
            "run_id": run_id,
            "status": status,
            "clarifications": clarifications,
        }
    if status == "QUEUED":
        return {
            "event": "run.queued",
            "run_id": run_id,
            "question_id": question_id or "",
            "conversation_id": conversation_id or "",
            "status": status,
        }
    return {
        "event": "run.failed",
        "run_id": run_id,
        "question_id": question_id or "",
        "conversation_id": conversation_id or "",
        "status": status,
        "error": {
            "code": error or "runtime_ask_failed",
            "message": error or "runtime ask failed",
            "retryable": False,
        },
    }


def _actionable_clarifications(
    result_data: dict[str, object] | None,
) -> list[dict[str, object]]:
    raw = result_data_clarifications(result_data)
    if not raw:
        raise ValueError("NEEDS_CLARIFICATION terminal event requires clarifications")
    clarifications: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(
                "NEEDS_CLARIFICATION terminal event requires clarification objects"
            )
        clarification = dict(item)
        clarification_id = str(
            clarification.get("id") or clarification.get("clarification_id") or ""
        ).strip()
        if not clarification_id:
            raise ValueError(
                "NEEDS_CLARIFICATION terminal event requires clarification id"
            )
        if not str(clarification.get("question") or "").strip():
            raise ValueError(
                "NEEDS_CLARIFICATION terminal event requires clarification question"
            )
        clarifications.append(clarification)
    return clarifications


def run_active_conflict_event(
    *,
    conversation_id: str,
    question_id: str,
    run_id: str,
    active_run_id: str,
    error: str | None,
) -> dict[str, object]:
    return {
        "event": "run.active_conflict",
        "conversation_id": conversation_id,
        "question_id": question_id,
        "run_id": run_id,
        "active_run_id": active_run_id,
        "status": "ACTIVE_RUN_CONFLICT",
        "error": {
            "code": error or "active_run_conflict",
            "message": error or "active_run_conflict",
            "retryable": True,
        },
    }
