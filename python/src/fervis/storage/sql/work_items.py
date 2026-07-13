"""SQL work-item queue operations for question runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.host_api.credentials import (
    runtime_context_with_delegated_credential,
)
from fervis.project.persistence.schema import metadata
from fervis.questions.execution_specs import (
    execution_spec_from_storage,
    execution_spec_kind,
    execution_spec_to_storage_dict,
)
from fervis.questions.ports import RunExecutionSpec, RunSubmission

from .rows import normalize_json_value, now_utc, row_mapping
from .terminal import record_runtime_error_result, terminal_result_for_run
from .transaction import sql_connection, sql_transaction

INLINE_RUN_LEASE_SECONDS = 300
TERMINAL_SUCCEEDED_STATUSES = frozenset({"COMPLETED"})
ACTIVE_STATUSES = ("QUEUED", "RUNNING", "WAITING_FOR_CLARIFICATION")


class ActiveRunConflict(Exception):
    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(run_id)


class StaleRunLease(RuntimeError):
    def __init__(self, *, run_id: str, worker_id: str, active_attempt: int) -> None:
        super().__init__(
            f"stale Fervis run lease for {run_id}: {worker_id} attempt {active_attempt}"
        )
        self.run_id = run_id
        self.worker_id = worker_id
        self.active_attempt = active_attempt


@dataclass(frozen=True)
class SQLRunWorkItem:
    run_id: str
    conversation_id: str
    tenant_id: str
    user_id: str
    status: str
    spec: RunExecutionSpec
    read_context_ref: ReadContextRef
    idempotency_key: str | None
    idempotency_authority_ref: str
    idempotency_scope: str
    attempt_count: int
    active_attempt: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    last_error: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class EnqueuedRunWorkItem:
    item: SQLRunWorkItem
    created: bool


class SQLWorkItemQueue:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.table = metadata.tables["fervis_run_work_item"]

    def find_idempotent_work_item(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        read_context_ref: ReadContextRef,
        conversation_id: str | None,
        idempotency_key: str | None,
        idempotency_scope: str,
    ) -> SQLRunWorkItem | None:
        if not idempotency_key:
            return None
        statement = sa.select(self.table).where(
            self.table.c.tenant_id == tenant_id,
            self.table.c.user_id == principal_id,
            self.table.c.idempotency_key == idempotency_key,
            self.table.c.idempotency_scope == idempotency_scope,
        )
        if conversation_id is not None:
            statement = statement.where(self.table.c.conversation_id == conversation_id)
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                statement.order_by(self.table.c.created_at)
            ).all()
        return next(
            (
                item
                for row in rows
                for item in (_work_item(row),)
                if item is not None and item.read_context_ref == read_context_ref
            ),
            None,
        )

    def get_work_item_for_run(self, run_id: str) -> SQLRunWorkItem:
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                sa.select(self.table).where(self.table.c.run_id == run_id)
            ).first()
        item = _work_item(row)
        if item is None:
            raise LookupError(f"Fervis run work item not found: {run_id}")
        return item

    def enqueue_run_work_item(
        self,
        *,
        submission: RunSubmission,
    ) -> EnqueuedRunWorkItem:
        with sql_transaction(self.engine) as connection:
            existing = self._find_by_run(connection, submission.run_id)
            if existing is not None:
                if (
                    submission.idempotency_key
                    and existing.idempotency_key == submission.idempotency_key
                    and existing.tenant_id == submission.tenant_id
                    and existing.conversation_id == submission.conversation_id
                ):
                    return EnqueuedRunWorkItem(item=existing, created=False)
                raise IntegrityError(
                    statement=None,
                    params=None,
                    orig=RuntimeError(
                        f"Fervis run id already exists: {submission.run_id}"
                    ),
                )
            if submission.idempotency_key:
                idempotent = self._find_idempotent(connection, submission)
                if idempotent is not None:
                    return EnqueuedRunWorkItem(item=idempotent, created=False)
            active = self._find_active(connection, submission)
            if active is not None:
                raise ActiveRunConflict(run_id=active.run_id)

            now = now_utc()
            inline = submission.execution_mode.value == "inline"
            execution_spec = execution_spec_to_storage_dict(submission.spec)
            execution_spec["runtime_context"] = normalize_json_value(
                runtime_context_with_delegated_credential(
                    submission.spec.runtime_context,
                    submission.principal.delegated_credential,
                )
            )
            values = {
                "run_id": submission.run_id,
                "conversation_id": submission.conversation_id,
                "tenant_id": submission.tenant_id,
                "user_id": str(submission.principal.principal_id),
                "status": "RUNNING" if inline else "QUEUED",
                "spec_kind": execution_spec_kind(submission.spec).value,
                "execution_spec": normalize_json_value(execution_spec),
                "read_context_ref": normalize_json_value(
                    submission.principal.read_context_ref.to_storage_dict()
                ),
                "idempotency_key": submission.idempotency_key or None,
                "idempotency_authority_ref": submission.idempotency_authority_ref,
                "idempotency_scope": submission.idempotency_scope,
                "attempt_count": 1 if inline else 0,
                "active_attempt": 1 if inline else 0,
                "max_attempts": 2,
                "lease_owner": "inline" if inline else None,
                "lease_expires_at": now + timedelta(seconds=INLINE_RUN_LEASE_SECONDS)
                if inline
                else None,
                "next_attempt_at": None,
                "last_error": "",
                "started_at": now if inline else None,
                "completed_at": None,
                "created_at": now,
                "updated_at": now,
            }
            try:
                connection.execute(sa.insert(self.table).values(**values))
            except IntegrityError:
                if submission.idempotency_key:
                    idempotent = self._find_idempotent(connection, submission)
                    if idempotent is not None:
                        return EnqueuedRunWorkItem(item=idempotent, created=False)
                active = self._find_active(connection, submission)
                if active is not None:
                    raise ActiveRunConflict(run_id=active.run_id) from None
                raise
            created = self._find_by_run(connection, submission.run_id)
            if created is None:
                raise RuntimeError(f"failed to enqueue Fervis run {submission.run_id}")
            return EnqueuedRunWorkItem(item=created, created=True)

    def claim_run_work_items(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ) -> list[SQLRunWorkItem]:
        now = now_utc()
        lease_until = now + timedelta(seconds=max(1, int(lease_seconds)))
        claimed: list[SQLRunWorkItem] = []
        with sql_transaction(self.engine) as connection:
            self._fail_expired_items_at_max_attempts(connection, now)
            candidates = connection.execute(
                sa.select(self.table)
                .where(
                    sa.or_(
                        sa.and_(
                            self.table.c.status == "QUEUED",
                            sa.or_(
                                self.table.c.next_attempt_at.is_(None),
                                self.table.c.next_attempt_at <= now,
                            ),
                        ),
                        sa.and_(
                            self.table.c.status == "RUNNING",
                            self.table.c.lease_expires_at < now,
                            self.table.c.attempt_count < self.table.c.max_attempts,
                        ),
                    )
                )
                .order_by(self.table.c.created_at)
                .limit(max(1, int(batch_size)))
            ).all()
            for row in candidates:
                item = _work_item(row)
                if item is None:
                    continue
                terminal = terminal_result_for_run(self.engine, item.run_id)
                if terminal is not None:
                    self._mark_terminal_from_lineage(
                        connection,
                        item=item,
                        terminal=terminal,
                    )
                    continue
                claimed_item = self._claim_item(
                    connection,
                    item=item,
                    worker_id=worker_id,
                    lease_until=lease_until,
                    now=now,
                )
                if claimed_item is not None:
                    claimed.append(claimed_item)
        return claimed

    def claim_run_work_item(
        self,
        *,
        run_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> SQLRunWorkItem | None:
        now = now_utc()
        lease_until = now + timedelta(seconds=max(1, int(lease_seconds)))
        with sql_transaction(self.engine) as connection:
            self._fail_expired_items_at_max_attempts(connection, now)
            item = self._find_by_run(connection, run_id)
            if item is None:
                return None
            terminal = terminal_result_for_run(self.engine, item.run_id)
            if terminal is not None:
                self._mark_terminal_from_lineage(
                    connection,
                    item=item,
                    terminal=terminal,
                )
                return self._find_by_run(connection, item.run_id)
            return self._claim_item(
                connection,
                item=item,
                worker_id=worker_id,
                lease_until=lease_until,
                now=now,
            )

    def require_current_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> SQLRunWorkItem:
        item = self.get_work_item_for_run(run_id)
        if (
            item.status != "RUNNING"
            or item.lease_owner != worker_id
            or item.active_attempt != active_attempt
        ):
            raise StaleRunLease(
                run_id=run_id,
                worker_id=worker_id,
                active_attempt=active_attempt,
            )
        return item

    def mark_work_item_terminal(
        self,
        *,
        run_id: str,
        status: str,
        error: str = "",
        worker_id: str = "",
        active_attempt: int | None = None,
    ) -> None:
        terminal_status = _terminal_work_status(status)
        now = now_utc()
        with sql_transaction(self.engine) as connection:
            statement = (
                sa.update(self.table)
                .where(self.table.c.run_id == run_id)
                .values(
                    status=terminal_status,
                    completed_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error=error or "",
                    updated_at=now,
                )
            )
            if worker_id and active_attempt is not None:
                statement = statement.where(
                    self.table.c.status == "RUNNING",
                    self.table.c.lease_owner == worker_id,
                    self.table.c.active_attempt == active_attempt,
                )
            result = connection.execute(statement)
        if worker_id and active_attempt is not None and result.rowcount != 1:
            raise StaleRunLease(
                run_id=run_id,
                worker_id=worker_id,
                active_attempt=active_attempt,
            )

    def wait_for_clarification(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> SQLRunWorkItem:
        now = now_utc()
        with sql_transaction(self.engine) as connection:
            result = connection.execute(
                sa.update(self.table)
                .where(
                    self.table.c.run_id == run_id,
                    self.table.c.status == "RUNNING",
                    self.table.c.lease_owner == worker_id,
                    self.table.c.active_attempt == active_attempt,
                )
                .values(
                    status="WAITING_FOR_CLARIFICATION",
                    completed_at=None,
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error="",
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise StaleRunLease(
                    run_id=run_id,
                    worker_id=worker_id,
                    active_attempt=active_attempt,
                )
            item = self._find_by_run(connection, run_id)
        if item is None:
            raise LookupError(f"Fervis run work item not found: {run_id}")
        return item

    def resume_from_clarification(
        self,
        *,
        run_id: str,
        spec: RunExecutionSpec,
        execution_mode: str,
    ) -> SQLRunWorkItem:
        inline = execution_mode == "inline"
        now = now_utc()
        with sql_transaction(self.engine) as connection:
            current = self._find_by_run(connection, run_id)
            if current is None or current.status != "WAITING_FOR_CLARIFICATION":
                raise ValueError("clarification does not belong to a resumable run")
            next_attempt = current.attempt_count + 1
            result = connection.execute(
                sa.update(self.table)
                .where(
                    self.table.c.run_id == run_id,
                    self.table.c.status == "WAITING_FOR_CLARIFICATION",
                )
                .values(
                    status="RUNNING" if inline else "QUEUED",
                    spec_kind=execution_spec_kind(spec).value,
                    execution_spec=normalize_json_value(
                        execution_spec_to_storage_dict(spec)
                    ),
                    attempt_count=next_attempt if inline else current.attempt_count,
                    active_attempt=next_attempt if inline else 0,
                    max_attempts=max(current.max_attempts, next_attempt + 1),
                    lease_owner="inline" if inline else None,
                    lease_expires_at=(
                        now + timedelta(seconds=INLINE_RUN_LEASE_SECONDS)
                        if inline
                        else None
                    ),
                    next_attempt_at=None,
                    completed_at=None,
                    last_error="",
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise ValueError("clarification run changed while resuming")
            item = self._find_by_run(connection, run_id)
        if item is None:
            raise LookupError(f"Fervis run work item not found: {run_id}")
        return item

    def queue_counts(self) -> dict[str, int]:
        counts = {
            "QUEUED": 0,
            "RUNNING": 0,
            "WAITING_FOR_CLARIFICATION": 0,
            "COMPLETED": 0,
            "FAILED": 0,
        }
        now = now_utc()
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(self.table.c.status, sa.func.count()).group_by(
                    self.table.c.status
                )
            ).all()
            expired = connection.execute(
                sa.select(sa.func.count()).where(
                    self.table.c.status == "RUNNING",
                    self.table.c.lease_expires_at < now,
                )
            ).scalar_one()
        for status, count in rows:
            counts[str(status)] = int(count)
        counts["EXPIRED_RUNNING"] = int(expired)
        return counts

    def reconcile_work_item_from_terminal_lineage(
        self,
        *,
        run_id: str,
    ) -> SQLRunWorkItem | None:
        with sql_transaction(self.engine) as connection:
            item = self._find_by_run(connection, run_id)
            if item is None:
                return None
            terminal = terminal_result_for_run(self.engine, run_id)
            if terminal is None:
                return item
            self._mark_terminal_from_lineage(
                connection,
                item=item,
                terminal=terminal,
            )
            return self._find_by_run(connection, run_id)

    def _claim_filter_for_candidate(
        self,
        *,
        run_id: str,
        attempt_count: int,
        now,
    ):
        return sa.and_(
            self.table.c.run_id == run_id,
            sa.or_(
                sa.and_(
                    self.table.c.status == "QUEUED",
                    self.table.c.attempt_count == attempt_count,
                    sa.or_(
                        self.table.c.next_attempt_at.is_(None),
                        self.table.c.next_attempt_at <= now,
                    ),
                ),
                sa.and_(
                    self.table.c.status == "RUNNING",
                    self.table.c.lease_expires_at < now,
                    self.table.c.attempt_count == attempt_count,
                    self.table.c.attempt_count < self.table.c.max_attempts,
                ),
            ),
        )

    def _claim_item(
        self,
        connection,
        *,
        item: SQLRunWorkItem,
        worker_id: str,
        lease_until,
        now,
    ) -> SQLRunWorkItem | None:
        active_attempt = item.attempt_count + 1
        result = connection.execute(
            sa.update(self.table)
            .where(
                self._claim_filter_for_candidate(
                    run_id=item.run_id,
                    attempt_count=item.attempt_count,
                    now=now,
                )
            )
            .values(
                status="RUNNING",
                attempt_count=active_attempt,
                active_attempt=active_attempt,
                lease_owner=worker_id,
                lease_expires_at=lease_until,
                started_at=sa.func.coalesce(self.table.c.started_at, now),
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            return None
        return self._find_by_run(connection, item.run_id)

    def _fail_expired_items_at_max_attempts(self, connection, now) -> None:
        error = "run_max_attempts_exceeded"
        rows = connection.execute(
            sa.select(self.table.c.run_id).where(
                self.table.c.status == "RUNNING",
                self.table.c.lease_expires_at < now,
                self.table.c.attempt_count >= self.table.c.max_attempts,
            )
        ).all()
        for row in rows:
            terminal = terminal_result_for_run(self.engine, str(row.run_id))
            if terminal is not None:
                status = _terminal_work_status(terminal.status)
                last_error = terminal.error or ""
            else:
                record_runtime_error_result(
                    engine=self.engine,
                    run_id=str(row.run_id),
                    error_code=error,
                )
                status = "FAILED"
                last_error = error
            connection.execute(
                sa.update(self.table)
                .where(
                    self.table.c.run_id == row.run_id,
                    self.table.c.status == "RUNNING",
                    self.table.c.lease_expires_at < now,
                    self.table.c.attempt_count >= self.table.c.max_attempts,
                )
                .values(
                    status=status,
                    completed_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error=last_error,
                    updated_at=now,
                )
            )

    def _find_by_run(self, connection, run_id: str) -> SQLRunWorkItem | None:
        row = connection.execute(
            sa.select(self.table).where(self.table.c.run_id == run_id)
        ).first()
        return _work_item(row)

    def _find_idempotent(
        self,
        connection,
        submission: RunSubmission,
    ) -> SQLRunWorkItem | None:
        if not submission.idempotency_key:
            return None
        row = connection.execute(
            sa.select(self.table).where(
                self.table.c.tenant_id == submission.tenant_id,
                self.table.c.idempotency_authority_ref
                == submission.idempotency_authority_ref,
                self.table.c.idempotency_scope == submission.idempotency_scope,
                self.table.c.idempotency_key == submission.idempotency_key,
            )
        ).first()
        return _work_item(row)

    def _find_active(
        self, connection, submission: RunSubmission
    ) -> SQLRunWorkItem | None:
        row = connection.execute(
            sa.select(self.table)
            .where(
                self.table.c.tenant_id == submission.tenant_id,
                self.table.c.conversation_id == submission.conversation_id,
                self.table.c.status.in_(ACTIVE_STATUSES),
            )
            .order_by(self.table.c.created_at)
        ).first()
        item = _work_item(row)
        if item is None:
            return None
        terminal = terminal_result_for_run(self.engine, item.run_id)
        if terminal is None:
            return item
        self._mark_terminal_from_lineage(connection, item=item, terminal=terminal)
        return None

    def _mark_terminal_from_lineage(
        self, connection, *, item: SQLRunWorkItem, terminal
    ):
        now = now_utc()
        connection.execute(
            sa.update(self.table)
            .where(
                self.table.c.run_id == item.run_id,
                self.table.c.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status=_terminal_work_status(terminal.status),
                completed_at=now,
                lease_owner=None,
                lease_expires_at=None,
                last_error=terminal.error or "",
                updated_at=now,
            )
        )


def _work_item(row) -> SQLRunWorkItem | None:
    if row is None:
        return None
    values = row_mapping(row)
    return SQLRunWorkItem(
        run_id=str(values["run_id"]),
        conversation_id=str(values["conversation_id"]),
        tenant_id=str(values["tenant_id"]),
        user_id=str(values["user_id"]),
        status=str(values["status"]),
        spec=execution_spec_from_storage(
            str(values["spec_kind"]),
            values["execution_spec"] or {},
        ),
        read_context_ref=ReadContextRef.from_storage_dict(
            values["read_context_ref"] or {}
        ),
        idempotency_key=values["idempotency_key"],
        idempotency_authority_ref=str(values["idempotency_authority_ref"] or ""),
        idempotency_scope=str(values["idempotency_scope"] or ""),
        attempt_count=int(values["attempt_count"]),
        active_attempt=int(values["active_attempt"]),
        max_attempts=int(values["max_attempts"]),
        lease_owner=values["lease_owner"],
        lease_expires_at=values["lease_expires_at"],
        last_error=str(values["last_error"] or ""),
        created_at=values["created_at"],
        started_at=values["started_at"],
        completed_at=values["completed_at"],
    )


def _terminal_work_status(status: str) -> str:
    return "COMPLETED" if status in TERMINAL_SUCCEEDED_STATUSES else "FAILED"
