"""Local queued run following."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from fervis.questions import AskResult

from .contracts import QueuedRunRequest, QueuedRunResult
from .events import NullQuestionRunEventSink, QuestionRunEventSink


class ClaimedRunWorkItem(Protocol):
    run_id: str
    active_attempt: int


class QueuedRunWorkQueue(Protocol):
    def claim_run_work_item(
        self,
        *,
        run_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> ClaimedRunWorkItem | None: ...


class QueuedRunWorker(Protocol):
    def process_queued_run(
        self,
        request: QueuedRunRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> QueuedRunResult: ...


@dataclass(frozen=True)
class LocalQueuedRunFollower:
    run_work: QueuedRunWorker
    work_queue: QueuedRunWorkQueue
    worker_id: str = "fervis-cli"
    lease_seconds: int = 60
    poll_interval_seconds: float = 0.1

    def follow(
        self,
        result: AskResult,
        *,
        event_sink: QuestionRunEventSink | None = None,
        wait_seconds: float = 0.0,
    ) -> AskResult:
        if result.status not in {"QUEUED", "RUNNING"}:
            return result
        events = event_sink or NullQuestionRunEventSink()
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        while True:
            claimed = self.work_queue.claim_run_work_item(
                run_id=result.run_id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            if claimed is not None:
                executed = self.run_work.process_queued_run(
                    QueuedRunRequest(
                        run_id=claimed.run_id,
                        worker_id=self.worker_id,
                        active_attempt=claimed.active_attempt,
                    ),
                    event_sink=events,
                )
                return _ask_result_from_queued_result(result, executed)
            if time.monotonic() >= deadline:
                return result
            remaining = max(0.0, deadline - time.monotonic())
            time.sleep(min(self.poll_interval_seconds, remaining))


def _ask_result_from_queued_result(
    original: AskResult,
    executed: QueuedRunResult,
) -> AskResult:
    return AskResult(
        status=executed.status,
        conversation_id=original.conversation_id,
        question_id=original.question_id,
        run_id=executed.run_id,
        answer=executed.answer,
        result_data=executed.result_data,
        error=executed.error,
        duration_ms=executed.duration_ms,
    )
