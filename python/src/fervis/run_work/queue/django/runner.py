from __future__ import annotations

from typing import Protocol

from fervis.run_work.worker import (
    WorkerCycle,
    WorkerRunResult,
    process_run_work_batch,
)

from .queue import claim_run_work_items, queue_counts, StaleRunLease


class RunWorker(Protocol):
    def process_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> WorkerRunResult: ...

    def fail_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
        error: str,
    ) -> WorkerRunResult: ...


def process_run_batch(
    *,
    worker_id: str,
    batch_size: int,
    lease_seconds: int,
    worker: RunWorker,
) -> WorkerCycle:
    return process_run_work_batch(
        worker_id=worker_id,
        batch_size=batch_size,
        lease_seconds=lease_seconds,
        work_queue=_DjangoRunWorkQueue(),
        worker=worker,
        stale_exceptions=(StaleRunLease,),
    )


class _DjangoRunWorkQueue:
    def claim_run_work_items(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ):
        return claim_run_work_items(
            worker_id=worker_id,
            batch_size=batch_size,
            lease_seconds=lease_seconds,
        )

    def queue_counts(self) -> dict[str, int]:
        return queue_counts()
