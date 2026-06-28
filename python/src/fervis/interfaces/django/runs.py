"""Fervis run work entrypoints for the Django interface."""

from __future__ import annotations

from typing import Any

from fervis.run_work import FailQueuedRunRequest, QueuedRunRequest

from .question_run_ports import django_run_work_service
from .run_views import (
    get_run_view,
    with_lineage_usage,
    with_worker_snapshot,
)


def process_run_work(
    *,
    run_id: str,
    worker_id: str,
    active_attempt: int,
) -> dict[str, Any]:
    django_run_work_service().process_queued_run(
        QueuedRunRequest(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
    )
    run = get_run_view(run_id)
    if run is None:
        raise RuntimeError("fervis interface view was not finalized")
    return with_lineage_usage(with_worker_snapshot(run))


def fail_run_work(
    *,
    run_id: str,
    worker_id: str,
    active_attempt: int,
    error: str,
) -> dict[str, Any]:
    django_run_work_service().fail_queued_run(
        FailQueuedRunRequest(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
            error=error,
        )
    )
    run = get_run_view(run_id)
    if run is None:
        raise RuntimeError("fervis interface view was not failed")
    return with_lineage_usage(with_worker_snapshot(run))


def get_run(run_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
    run = get_run_view(run_id, tenant_id=tenant_id)
    if run is None:
        return None
    return with_lineage_usage(with_worker_snapshot(run))
