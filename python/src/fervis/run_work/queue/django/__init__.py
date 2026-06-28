"""Durable Django queue for Fervis run work items."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fervis.run_work.worker import WorkerCycle

    from .runner import RunWorker


def process_run_batch(
    *,
    worker_id: str,
    batch_size: int,
    lease_seconds: int,
    worker: RunWorker,
) -> WorkerCycle:
    from .runner import process_run_batch as process_batch

    return process_batch(
        worker_id=worker_id,
        batch_size=batch_size,
        lease_seconds=lease_seconds,
        worker=worker,
    )
