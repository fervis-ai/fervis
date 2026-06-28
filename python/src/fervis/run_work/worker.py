"""Shared queued-run worker batch processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import FailQueuedRunRequest, QueuedRunRequest, QueuedRunResult


class ClaimedRunWorkItem(Protocol):
    run_id: str
    active_attempt: int


class RunWorkQueue(Protocol):
    def claim_run_work_items(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ) -> list[ClaimedRunWorkItem]: ...

    def queue_counts(self) -> dict[str, int]: ...


class RunWorker(Protocol):
    def process_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> "WorkerRunResult": ...

    def fail_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
        error: str,
    ) -> "WorkerRunResult": ...


class QueuedRunProcessor(Protocol):
    def process_queued_run(self, request: QueuedRunRequest) -> QueuedRunResult: ...

    def fail_queued_run(self, request: FailQueuedRunRequest) -> QueuedRunResult: ...


@dataclass(frozen=True)
class WorkerRunResult:
    run_id: str
    active_attempt: int
    status: str
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "active_attempt": self.active_attempt,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkerCycle:
    claimed_count: int
    completed_count: int
    failed_count: int
    queue_counts: dict[str, int]
    runs: tuple[WorkerRunResult, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "claimed_count": self.claimed_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "queue_counts": dict(self.queue_counts),
            "runs": [run.to_payload() for run in self.runs],
        }


@dataclass(frozen=True)
class RunWorkBatchProcessor:
    work_queue: RunWorkQueue
    worker: RunWorker

    def process_once(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ) -> WorkerCycle:
        return process_run_work_batch(
            worker_id=worker_id,
            batch_size=batch_size,
            lease_seconds=lease_seconds,
            work_queue=self.work_queue,
            worker=self.worker,
        )


@dataclass(frozen=True)
class RunWorkServiceWorker:
    run_work: QueuedRunProcessor

    def process_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> WorkerRunResult:
        result = self.run_work.process_queued_run(
            QueuedRunRequest(
                run_id=run_id,
                worker_id=worker_id,
                active_attempt=active_attempt,
            )
        )
        return WorkerRunResult(
            run_id=result.run_id,
            active_attempt=active_attempt,
            status=result.status,
            error=result.error,
        )

    def fail_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
        error: str,
    ) -> WorkerRunResult:
        result = self.run_work.fail_queued_run(
            FailQueuedRunRequest(
                run_id=run_id,
                worker_id=worker_id,
                active_attempt=active_attempt,
                error=error,
            )
        )
        return WorkerRunResult(
            run_id=result.run_id,
            active_attempt=active_attempt,
            status=result.status,
            error=result.error,
        )


def process_run_work_batch(
    *,
    worker_id: str,
    batch_size: int,
    lease_seconds: int,
    work_queue: RunWorkQueue,
    worker: RunWorker,
    stale_exceptions: tuple[type[Exception], ...] = (),
) -> WorkerCycle:
    claimed = work_queue.claim_run_work_items(
        worker_id=worker_id,
        batch_size=batch_size,
        lease_seconds=lease_seconds,
    )
    completed_count = 0
    failed_count = 0
    results: list[WorkerRunResult] = []
    for item in claimed:
        try:
            result = worker.process_run_work(
                run_id=item.run_id,
                worker_id=worker_id,
                active_attempt=item.active_attempt,
            )
        except Exception as exc:
            if isinstance(exc, stale_exceptions):
                continue
            error = str(exc) or exc.__class__.__name__
            try:
                result = worker.fail_run_work(
                    run_id=item.run_id,
                    worker_id=worker_id,
                    active_attempt=item.active_attempt,
                    error=error,
                )
            except Exception as fail_exc:
                if isinstance(fail_exc, stale_exceptions):
                    continue
                raise
        if result.status == "FAILED":
            failed_count += 1
        else:
            completed_count += 1
        results.append(result)
    return WorkerCycle(
        claimed_count=len(claimed),
        completed_count=completed_count,
        failed_count=failed_count,
        queue_counts=work_queue.queue_counts(),
        runs=tuple(results),
    )
