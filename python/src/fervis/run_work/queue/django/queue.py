from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

from django.db import IntegrityError, models, transaction
from django.db.models import Count, Q
from django.utils import timezone

from fervis.lineage.django.runtime_failures import (
    record_worker_runtime_error,
)
from fervis.lineage.django.terminal_results import (
    terminal_status_for_run,
)
from .models import RunWorkItem, RunWorkStatus

INLINE_RUN_LEASE_SECONDS = 300


class ActiveRunConflict(Exception):
    def __init__(self, *, run_id: str, run: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.run = run
        super().__init__(run_id)


class StaleRunLease(RuntimeError):
    def __init__(self, *, run_id: str, worker_id: str, active_attempt: int) -> None:
        super().__init__(
            f"stale Fervis run lease for {run_id}: "
            f"{worker_id} attempt {active_attempt}"
        )
        self.run_id = run_id
        self.worker_id = worker_id
        self.active_attempt = active_attempt


@dataclass(frozen=True)
class EnqueuedRunWorkItem:
    item: RunWorkItem
    created: bool


def reset_question_run_queue_for_tests() -> None:
    RunWorkItem.objects.all().delete()


def work_item_snapshot_for_run(run_id: str) -> dict[str, Any] | None:
    item = RunWorkItem.objects.filter(run_id=str(run_id)).first()
    if item is None:
        return None
    return _work_item_snapshot(item)


def get_work_item_for_run(run_id: str) -> RunWorkItem:
    return RunWorkItem.objects.get(run_id=str(run_id))


def find_idempotent_work_item(
    *,
    tenant_id: str,
    conversation_id: str | None,
    idempotency_key: str | None,
) -> RunWorkItem | None:
    if not idempotency_key:
        return None
    rows = RunWorkItem.objects.filter(
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
    )
    if conversation_id is not None:
        rows = rows.filter(conversation_id=conversation_id)
    return rows.order_by("created_at").first()


def queue_counts() -> dict[str, int]:
    counts = {status.value: 0 for status in RunWorkStatus}
    rows = (
        RunWorkItem.objects.values("status")
        .order_by()
        .annotate(count=Count("id"))
    )
    for row in rows:
        counts[str(row["status"])] = int(row["count"])
    now = timezone.now()
    counts["EXPIRED_RUNNING"] = RunWorkItem.objects.filter(
        status=RunWorkStatus.RUNNING,
        lease_expires_at__lt=now,
    ).count()
    return counts


def enqueue_run_work_item(
    *,
    run_id: str,
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    question: str,
    provider: str | None,
    model_key: str,
    execution_mode: str,
    conversation_context: dict[str, Any],
    runtime_context: dict[str, Any],
    read_context_ref: dict[str, Any],
    idempotency_key: str | None,
    max_budget_usd: float | Decimal,
    max_thinking_tokens: int,
) -> EnqueuedRunWorkItem:
    existing_run = RunWorkItem.objects.filter(run_id=run_id).first()
    if existing_run is not None:
        if (
            idempotency_key
            and existing_run.idempotency_key == idempotency_key
            and existing_run.tenant_id == tenant_id
            and existing_run.conversation_id == conversation_id
        ):
            return EnqueuedRunWorkItem(item=existing_run, created=False)
        raise IntegrityError(f"Fervis run id already exists: {run_id}")

    if idempotency_key:
        existing = find_idempotent_work_item(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            return EnqueuedRunWorkItem(item=existing, created=False)

    reconcile_active_work_items(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
    )
    active = _active_work_item(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
    )
    if active is not None:
        raise ActiveRunConflict(
            run_id=active.run_id,
            run=None,
        )

    try:
        with transaction.atomic():
            inline = execution_mode == "inline"
            now = timezone.now()
            lease_until = now + timedelta(seconds=INLINE_RUN_LEASE_SECONDS)
            return EnqueuedRunWorkItem(
                item=RunWorkItem.objects.create(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    provider=provider,
                    model_key=model_key,
                    question=question,
                    status=RunWorkStatus.RUNNING
                    if inline
                    else RunWorkStatus.QUEUED,
                    conversation_context=conversation_context,
                    runtime_context=dict(runtime_context or {}),
                    read_context_ref=dict(read_context_ref or {}),
                    idempotency_key=idempotency_key or None,
                    session_mode="continue",
                    session_id=None,
                    approval_mode="auto_allow",
                    approval_decision=None,
                    max_budget_usd=Decimal(str(max_budget_usd)),
                    max_thinking_tokens=max_thinking_tokens,
                    attempt_count=1 if inline else 0,
                    active_attempt=1 if inline else 0,
                    lease_owner="inline" if inline else None,
                    lease_expires_at=lease_until if inline else None,
                    started_at=now if inline else None,
                ),
                created=True,
            )
    except IntegrityError:
        if idempotency_key:
            existing = find_idempotent_work_item(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                return EnqueuedRunWorkItem(item=existing, created=False)
        if RunWorkItem.objects.filter(run_id=run_id).exists():
            raise
        active = _active_work_item(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
        if active is not None:
            raise ActiveRunConflict(
                run_id=active.run_id,
                run=None,
            )
        raise


def _active_work_item(
    *,
    tenant_id: str,
    conversation_id: str,
) -> RunWorkItem | None:
    rows = RunWorkItem.objects.filter(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        status__in=[
            RunWorkStatus.QUEUED,
            RunWorkStatus.RUNNING,
        ],
    )
    return rows.order_by("created_at").first()


def reconcile_active_work_items(*, tenant_id: str, conversation_id: str) -> None:
    active_items = RunWorkItem.objects.filter(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        status__in=[
            RunWorkStatus.QUEUED,
            RunWorkStatus.RUNNING,
        ],
    )
    for item in active_items:
        reconcile_work_item_from_terminal_lineage(item)


def reconcile_work_item_from_terminal_lineage(item: RunWorkItem) -> bool:
    status = terminal_status_for_run(item.run_id, tenant_id=item.tenant_id)
    if status is None:
        return False
    terminal_status = (
        RunWorkStatus.COMPLETED
        if status in {"COMPLETED", "NEEDS_CLARIFICATION"}
        else RunWorkStatus.FAILED
    )
    item.status = terminal_status
    item.completed_at = timezone.now()
    item.lease_owner = None
    item.lease_expires_at = None
    item.save(
        update_fields=[
            "status",
            "completed_at",
            "lease_owner",
            "lease_expires_at",
            "updated_at",
        ]
    )
    return True


def claim_run_work_items(
    *,
    worker_id: str,
    batch_size: int,
    lease_seconds: int,
) -> list[RunWorkItem]:
    now = timezone.now()
    batch_size = max(1, int(batch_size))
    lease_until = now + timedelta(seconds=max(1, int(lease_seconds)))

    with transaction.atomic():
        _fail_expired_items_at_max_attempts(now)
        candidates = (
            RunWorkItem.objects.select_for_update(skip_locked=True)
            .filter(_claimable_filter(now))
            .order_by("created_at")[:batch_size]
        )
        claimed = list(candidates)
        for item in claimed:
            if reconcile_work_item_from_terminal_lineage(item):
                continue
            next_attempt = item.attempt_count + 1
            item.status = RunWorkStatus.RUNNING
            item.attempt_count = next_attempt
            item.active_attempt = next_attempt
            item.lease_owner = worker_id
            item.lease_expires_at = lease_until
            item.started_at = item.started_at or now
            item.save(
                update_fields=[
                    "status",
                    "attempt_count",
                    "active_attempt",
                    "lease_owner",
                    "lease_expires_at",
                    "started_at",
                    "updated_at",
                ]
            )
    return [
        item
        for item in claimed
        if item.status == RunWorkStatus.RUNNING
        and item.lease_owner == worker_id
    ]


def _fail_expired_items_at_max_attempts(now) -> None:
    items = RunWorkItem.objects.select_for_update(skip_locked=True).filter(
        status=RunWorkStatus.RUNNING,
        lease_expires_at__lt=now,
        attempt_count__gte=models.F("max_attempts"),
    )
    for item in items:
        error = "run_max_attempts_exceeded"
        record_worker_runtime_error(
            run_id=item.run_id,
            error_code=error,
            message=error,
        )
        item.status = RunWorkStatus.FAILED
        item.completed_at = now
        item.lease_owner = None
        item.lease_expires_at = None
        item.last_error = error
        item.save(
            update_fields=[
                "status",
                "completed_at",
                "lease_owner",
                "lease_expires_at",
                "last_error",
                "updated_at",
            ]
        )


def mark_work_item_terminal(
    *,
    run_id: str,
    status: str,
    error: str = "",
) -> None:
    terminal_status = (
        RunWorkStatus.COMPLETED
        if status in {"COMPLETED", "NEEDS_CLARIFICATION"}
        else RunWorkStatus.FAILED
    )
    RunWorkItem.objects.filter(run_id=run_id).update(
        status=terminal_status,
        completed_at=timezone.now(),
        lease_owner=None,
        lease_expires_at=None,
        last_error=error or "",
    )


def require_current_run_lease(
    *,
    run_id: str,
    worker_id: str,
    active_attempt: int,
    lock: bool = False,
) -> RunWorkItem:
    rows = RunWorkItem.objects
    if lock:
        rows = rows.select_for_update()
    item = rows.get(run_id=run_id)
    if (
        item.status != RunWorkStatus.RUNNING
        or item.lease_owner != worker_id
        or item.active_attempt != active_attempt
    ):
        raise StaleRunLease(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
    return item


def mark_work_item_terminal_for_current_lease(
    *,
    run_id: str,
    worker_id: str,
    active_attempt: int,
    status: str,
    error: str = "",
) -> None:
    terminal_status = (
        RunWorkStatus.COMPLETED
        if status in {"COMPLETED", "NEEDS_CLARIFICATION"}
        else RunWorkStatus.FAILED
    )
    updated = RunWorkItem.objects.filter(
        run_id=run_id,
        status=RunWorkStatus.RUNNING,
        lease_owner=worker_id,
        active_attempt=active_attempt,
    ).update(
        status=terminal_status,
        completed_at=timezone.now(),
        lease_owner=None,
        lease_expires_at=None,
        last_error=error or "",
    )
    if updated != 1:
        raise StaleRunLease(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )


def _claimable_filter(now):
    ready = Q(next_attempt_at__isnull=True) | Q(next_attempt_at__lte=now)
    queued_ready = Q(status=RunWorkStatus.QUEUED) & ready
    expired_running = Q(
        status=RunWorkStatus.RUNNING,
        lease_expires_at__lt=now,
    )
    return queued_ready | expired_running


def _work_item_snapshot(item: RunWorkItem) -> dict[str, Any]:
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
