"""Django interface adapter for processing queued Fervis runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fervis.run_work.worker import WorkerRunResult
from fervis.run_work.queue.django import (
    process_run_batch as process_queued_run_batch,
)

from .runs import process_run_work, fail_run_work

if TYPE_CHECKING:
    from fervis.run_work.worker import WorkerCycle


class DjangoInterfaceRunWorker:
    def process_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> WorkerRunResult:
        run = process_run_work(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        return _worker_result(run, run_id=run_id, active_attempt=active_attempt)

    def fail_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
        error: str,
    ) -> WorkerRunResult:
        run = fail_run_work(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
            error=error,
        )
        return _worker_result(
            run,
            run_id=run_id,
            active_attempt=active_attempt,
            error=error,
        )


def process_run_batch(
    *,
    worker_id: str,
    batch_size: int,
    lease_seconds: int,
) -> WorkerCycle:
    return process_queued_run_batch(
        worker_id=worker_id,
        batch_size=batch_size,
        lease_seconds=lease_seconds,
        worker=DjangoInterfaceRunWorker(),
    )


def _worker_result(
    run: dict[str, object],
    *,
    run_id: str,
    active_attempt: int,
    error: str | None = None,
) -> WorkerRunResult:
    return WorkerRunResult(
        run_id=str(run.get("run_id") or run.get("runId") or run_id),
        active_attempt=active_attempt,
        status=str(run.get("status") or "FAILED"),
        error=str(run.get("error") or error or "") or None,
    )
