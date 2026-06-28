"""SQL-backed question lifecycle ports."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.lineage.recorder_core import LineageRecorder
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.credentials import (
    delegated_credential_from_runtime_context,
)
from fervis.lineage.run_spine import (
    ClarificationResponseStart as SpineClarificationResponseStart,
    QuestionRunSequenceStore,
    QuestionRunStart as SpineQuestionRunStart,
    QuestionRunStartRequest,
    QuestionStart as SpineQuestionStart,
    record_question_run_start,
)
from fervis.project.persistence.schema import metadata
from fervis.questions.contracts import ExecutionMode, QuestionPrincipal
from fervis.questions.ports import (
    AuthorizedQuestionAccess,
    QueuedRun,
    QuestionRunRecord,
    QuestionRunSubmissionKind,
    QuestionRunSubmissionResult,
    RunSubmission,
)
from fervis.questions.service import QuestionService
from fervis.run_work.service import RunWorkService

from .lineage_query import SQLLineageQuery
from .lineage_store import SQLLineageRecorderStore
from .authority_scope import (
    conversation_is_authorized,
    question_is_authorized,
)
from .terminal import (
    record_runtime_error_result,
    run_has_terminal_result,
    terminal_result_for_run,
)
from .transaction import sql_connection, sql_transaction
from .work_items import (
    ActiveRunConflict,
    SQLRunWorkItem,
    SQLWorkItemQueue,
)

DEFAULT_ADAPTER_REF = "fervis_sql"
DEFAULT_RUNTIME_VERSION = "development"


class SQLQuestionLineagePort:
    def __init__(self, *, engine: Engine) -> None:
        self.engine = engine
        self.lineage_query = SQLLineageQuery(engine)

    def conversation_memory_context(
        self,
        *,
        conversation_id: str,
        authority: ReadAuthority,
    ) -> dict[str, Any]:
        if not conversation_is_authorized(
            self.engine,
            conversation_id=conversation_id,
            authority=authority,
        ):
            return {}
        from fervis.memory.lineage import LineageMemoryArtifactService

        artifacts = LineageMemoryArtifactService(self.lineage_query).for_conversation(
            conversation_id
        )
        if not artifacts:
            return {}
        return {"factArtifacts": [artifact.to_dict() for artifact in artifacts]}

    def record_failed_runtime_fallback(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        del status, answer, result_data
        record_runtime_error_result(
            engine=self.engine,
            run_id=run_id,
            error_code=error or "fervis_failed",
        )


class SQLQuestionLifecyclePort:
    def __init__(self, *, engine: Engine) -> None:
        self.engine = engine
        self.queue = SQLWorkItemQueue(engine)
        self.recorder = LineageRecorder(SQLLineageRecorderStore(engine))

    def get_question(
        self,
        *,
        question_id: str,
        authority: ReadAuthority,
    ) -> AuthorizedQuestionAccess | None:
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                sa.select(
                    question.c.question_id,
                    question.c.conversation_id,
                    conversation.c.tenant_id,
                    conversation.c.read_context_ref,
                    question.c.original_question,
                )
                .select_from(
                    question.join(
                        conversation,
                        question.c.conversation_id == conversation.c.conversation_id,
                    )
                )
                .where(
                    question.c.question_id == question_id,
                    conversation.c.tenant_id == authority.tenant_id,
                )
            ).first()
        if row is None:
            return None
        if not question_is_authorized(
            self.engine,
            question_id=question_id,
            authority=authority,
        ):
            return None
        return AuthorizedQuestionAccess._issue(
            question_id=str(row.question_id),
            conversation_id=str(row.conversation_id),
            tenant_id=str(row.tenant_id),
            original_question=str(row.original_question),
            read_context_ref=ReadContextRef.from_storage_dict(
                row.read_context_ref or {}
            ),
        )

    def authorize_conversation(
        self,
        *,
        conversation_id: str,
        authority: ReadAuthority,
    ) -> None:
        if not conversation_is_authorized(
            self.engine,
            conversation_id=conversation_id,
            authority=authority,
        ):
            raise PermissionError("conversation is not owned by read authority")

    def find_idempotent_run(
        self,
        *,
        authority: ReadAuthority,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None:
        item = self.queue.find_idempotent_work_item(
            tenant_id=authority.tenant_id,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
        )
        if item is not None and not conversation_is_authorized(
            self.engine,
            conversation_id=item.conversation_id,
            authority=authority,
        ):
            return None
        return (
            _queued_run_from_work_item(self.engine, item) if item is not None else None
        )

    def submit_question_run_atomically(
        self,
        *,
        submission: RunSubmission,
        record: QuestionRunRecord,
    ) -> QuestionRunSubmissionResult:
        try:
            with sql_transaction(self.engine):
                enqueued = self.queue.enqueue_run_work_item(submission=submission)
                if not enqueued.created:
                    return QuestionRunSubmissionResult(
                        kind=QuestionRunSubmissionKind.EXISTING,
                        run=_queued_run_from_work_item(self.engine, enqueued.item),
                    )
                record_question_run_start(
                    _spine_question_run_start(record),
                    sequence_store=SQLQuestionRunSequenceStore(self.engine),
                    recorder=self.recorder,
                )
        except ActiveRunConflict as exc:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.ACTIVE_CONFLICT,
                run=_queued_run_from_work_item(
                    self.engine,
                    self.queue.get_work_item_for_run(exc.run_id),
                ),
            )
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=_queued_run_from_work_item(self.engine, enqueued.item),
        )

    def load_executable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        item = self.queue.reconcile_work_item_from_terminal_lineage(run_id=run_id)
        if item is None:
            item = self.queue.get_work_item_for_run(run_id)
        terminal = _terminal_run_from_lineage(self.engine, item)
        if terminal is not None:
            return terminal
        if active_attempt is None:
            raise ValueError("active_attempt is required for queued run work")
        item = self.queue.require_current_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        return _queued_run_from_work_item(self.engine, item)

    def load_failable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        if active_attempt is None:
            raise ValueError("active_attempt is required for queued run failure")
        item = self.queue.reconcile_work_item_from_terminal_lineage(run_id=run_id)
        if item is not None and item.status not in {"QUEUED", "RUNNING"}:
            return _queued_run_from_work_item(self.engine, item)
        item = self.queue.require_current_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        terminal = _terminal_run_from_lineage(self.engine, item)
        return terminal or _queued_run_from_work_item(self.engine, item)

    def terminalize(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
        worker_id: str = "",
        active_attempt: int | None = None,
    ) -> QueuedRun:
        self.queue.mark_work_item_terminal(
            run_id=run_id,
            status=status,
            error=error or "",
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        terminal = _queued_run_from_work_item(
            self.engine,
            self.queue.get_work_item_for_run(run_id),
        )
        return replace(
            terminal,
            answer=answer if answer is not None else terminal.answer,
            result_data=result_data
            if result_data is not None
            else terminal.result_data,
            error=error or terminal.error,
        )


class SQLQuestionStateReaderPort:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.queue = SQLWorkItemQueue(engine)

    def list_conversations(
        self,
        *,
        authority: ReadAuthority,
    ) -> list[dict[str, Any]]:
        conversation = metadata.tables["fervis_conversation"]
        statement = (
            sa.select(
                conversation.c.conversation_id,
                conversation.c.tenant_id,
                conversation.c.read_context_ref,
                conversation.c.created_at,
            )
            .where(conversation.c.tenant_id == authority.tenant_id)
            .order_by(conversation.c.created_at.desc())
        )
        with sql_connection(self.engine) as connection:
            rows = list(connection.execute(statement))

        items: list[dict[str, Any]] = []
        for row in rows:
            if not authority.read_context_ref.matches_storage_dict(
                row.read_context_ref or {}
            ):
                continue
            latest_question = self._latest_question_for_conversation(
                str(row.conversation_id)
            )
            if latest_question is None:
                continue
            first_question = self._first_question_for_conversation(
                str(row.conversation_id)
            )
            access = AuthorizedQuestionAccess._issue(
                question_id=str(latest_question.question_id),
                conversation_id=str(row.conversation_id),
                tenant_id=str(row.tenant_id),
                original_question=str(latest_question.original_question),
                read_context_ref=ReadContextRef.from_storage_dict(
                    row.read_context_ref or {}
                ),
            )
            current_run_id = self._latest_run_id(access.question_id)
            current = (
                self.get_question_run(current_run_id, access=access)
                if current_run_id
                else None
            )
            items.append(
                {
                    "conversationId": str(row.conversation_id),
                    "firstQuestion": (
                        str(first_question.original_question)
                        if first_question is not None
                        else access.original_question
                    ),
                    "latestQuestionId": access.question_id,
                    "currentRunId": (current or {}).get("runId"),
                    "status": str((current or {}).get("status") or "RUNNING"),
                    "runCount": self._run_count_for_question(access.question_id),
                    "updatedAt": _iso_datetime(
                        self._latest_activity_at(access.question_id) or row.created_at
                    ),
                }
            )
        return sorted(
            items,
            key=lambda item: str(item["updatedAt"]),
            reverse=True,
        )

    def get_question_state(
        self,
        *,
        access: AuthorizedQuestionAccess,
    ) -> dict[str, Any] | None:
        access.require_valid()
        question = self._question_row(access)
        if question is None:
            return None
        latest_run_id = self._latest_run_id(access.question_id)
        current = (
            self.get_question_run(
                latest_run_id,
                access=access,
            )
            if latest_run_id
            else None
        )
        return {
            "questionId": str(question.question_id),
            "conversationId": str(question.conversation_id),
            "tenantId": str(question.tenant_id),
            "status": str((current or {}).get("status") or "RUNNING"),
            "currentRunId": (current or {}).get("runId"),
            "question": str(question.original_question),
            "answer": (current or {}).get("answer"),
            "resultData": (current or {}).get("resultData"),
            "error": (current or {}).get("error"),
        }

    def list_question_runs(
        self,
        *,
        access: AuthorizedQuestionAccess,
    ) -> list[dict[str, Any]]:
        access.require_valid()
        runs: list[dict[str, Any]] = []
        for run_id in self._run_ids_for_question(
            access,
        ):
            run = self.get_question_run(
                run_id,
                access=access,
            )
            if run is not None:
                runs.append(run)
        return runs

    def get_question_run(
        self,
        run_id: str,
        *,
        access: AuthorizedQuestionAccess,
    ) -> dict[str, Any] | None:
        access.require_valid()
        run_question_id = self._question_id_for_run(
            run_id,
            access=access,
        )
        if run_question_id != access.question_id:
            return None
        try:
            item = self.queue.reconcile_work_item_from_terminal_lineage(run_id=run_id)
            if item is None:
                item = self.queue.get_work_item_for_run(run_id)
        except LookupError:
            return None
        if item.tenant_id != access.tenant_id:
            return None
        terminal = terminal_result_for_run(self.engine, run_id)
        return {
            "runId": item.run_id,
            "runNumber": self._run_number_for_run(item.run_id, access=access),
            "triggerKind": self._trigger_kind_for_run(item.run_id, access=access),
            "questionId": access.question_id,
            "conversationId": item.conversation_id,
            "tenantId": item.tenant_id,
            "status": terminal.status if terminal is not None else item.status,
            "question": item.question,
            "answer": terminal.answer if terminal is not None else None,
            "resultData": terminal.result_data if terminal is not None else None,
            "error": (terminal.error if terminal is not None else item.last_error)
            or None,
            "modelKey": item.model_key,
        }

    def _question_row(
        self,
        access: AuthorizedQuestionAccess,
    ):
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        statement = (
            sa.select(
                question.c.question_id,
                question.c.conversation_id,
                question.c.original_question,
                conversation.c.tenant_id,
            )
            .select_from(
                question.join(
                    conversation,
                    question.c.conversation_id == conversation.c.conversation_id,
                )
            )
            .where(question.c.question_id == access.question_id)
            .where(conversation.c.tenant_id == access.tenant_id)
        )
        with sql_connection(self.engine) as connection:
            row = connection.execute(statement).first()
        return row

    def _latest_run_id(self, question_id: str) -> str:
        run = metadata.tables["fervis_question_run"]
        with sql_connection(self.engine) as connection:
            value = connection.execute(
                sa.select(run.c.run_id)
                .where(run.c.question_id == question_id)
                .order_by(run.c.run_number.desc(), run.c.created_at.desc())
                .limit(1)
            ).scalar()
        return str(value or "")

    def _latest_question_for_conversation(self, conversation_id: str):
        question = metadata.tables["fervis_question"]
        statement = (
            sa.select(
                question.c.question_id,
                question.c.original_question,
                question.c.created_at,
            )
            .where(question.c.conversation_id == conversation_id)
            .order_by(
                question.c.conversation_sequence.desc(),
                question.c.created_at.desc(),
            )
            .limit(1)
        )
        with sql_connection(self.engine) as connection:
            return connection.execute(statement).first()

    def _first_question_for_conversation(self, conversation_id: str):
        question = metadata.tables["fervis_question"]
        statement = (
            sa.select(
                question.c.question_id,
                question.c.original_question,
                question.c.created_at,
            )
            .where(question.c.conversation_id == conversation_id)
            .order_by(
                question.c.conversation_sequence.asc(),
                question.c.created_at.asc(),
            )
            .limit(1)
        )
        with sql_connection(self.engine) as connection:
            return connection.execute(statement).first()

    def _run_count_for_question(self, question_id: str) -> int:
        run = metadata.tables["fervis_question_run"]
        with sql_connection(self.engine) as connection:
            value = connection.execute(
                sa.select(sa.func.count()).where(run.c.question_id == question_id)
            ).scalar()
        return int(value or 0)

    def _latest_activity_at(self, question_id: str):
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        with sql_connection(self.engine) as connection:
            value = connection.execute(
                sa.select(run.c.created_at)
                .where(run.c.question_id == question_id)
                .order_by(run.c.run_number.desc(), run.c.created_at.desc())
                .limit(1)
            ).scalar()
            if value is not None:
                return value
            return connection.execute(
                sa.select(question.c.created_at)
                .where(question.c.question_id == question_id)
                .limit(1)
            ).scalar()

    def _run_ids_for_question(
        self,
        access: AuthorizedQuestionAccess,
    ) -> list[str]:
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        statement = (
            sa.select(run.c.run_id)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id).join(
                    conversation,
                    question.c.conversation_id == conversation.c.conversation_id,
                )
            )
            .where(run.c.question_id == access.question_id)
            .where(conversation.c.tenant_id == access.tenant_id)
            .order_by(run.c.run_number.asc(), run.c.created_at.asc())
        )
        with sql_connection(self.engine) as connection:
            values = [str(value) for value in connection.execute(statement).scalars()]
        return values

    def _question_id_for_run(
        self,
        run_id: str,
        *,
        access: AuthorizedQuestionAccess,
    ) -> str:
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        statement = (
            sa.select(run.c.question_id)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id).join(
                    conversation,
                    question.c.conversation_id == conversation.c.conversation_id,
                )
            )
            .where(run.c.run_id == run_id)
            .where(conversation.c.tenant_id == access.tenant_id)
        )
        with sql_connection(self.engine) as connection:
            value = connection.execute(statement).scalar()
        question_id = str(value or "")
        if question_id and question_id != access.question_id:
            return ""
        return question_id

    def _run_number_for_run(
        self,
        run_id: str,
        *,
        access: AuthorizedQuestionAccess,
    ) -> int:
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        statement = (
            sa.select(run.c.run_number)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id).join(
                    conversation,
                    question.c.conversation_id == conversation.c.conversation_id,
                )
            )
            .where(run.c.run_id == run_id)
            .where(run.c.question_id == access.question_id)
            .where(conversation.c.tenant_id == access.tenant_id)
        )
        with sql_connection(self.engine) as connection:
            value = connection.execute(statement).scalar()
        return int(value or 0)

    def _trigger_kind_for_run(
        self,
        run_id: str,
        *,
        access: AuthorizedQuestionAccess,
    ) -> str:
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        statement = (
            sa.select(run.c.trigger_kind)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id).join(
                    conversation,
                    question.c.conversation_id == conversation.c.conversation_id,
                )
            )
            .where(run.c.run_id == run_id)
            .where(run.c.question_id == access.question_id)
            .where(conversation.c.tenant_id == access.tenant_id)
        )
        with sql_connection(self.engine) as connection:
            value = connection.execute(statement).scalar()
        return str(value or "")


def _iso_datetime(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


class SQLQuestionRunSequenceStore(QuestionRunSequenceStore):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def transaction(self):
        return sql_transaction(self.engine)

    def next_conversation_sequence(self, conversation_id: str) -> int:
        question = metadata.tables["fervis_question"]
        with sql_transaction(self.engine) as connection:
            current = connection.execute(
                sa.select(sa.func.max(question.c.conversation_sequence)).where(
                    question.c.conversation_id == conversation_id
                )
            ).scalar()
        return int(current or 0) + 1

    def next_question_run_number(self, question_id: str) -> int:
        run = metadata.tables["fervis_question_run"]
        with sql_transaction(self.engine) as connection:
            current = connection.execute(
                sa.select(sa.func.max(run.c.run_number)).where(
                    run.c.question_id == question_id
                )
            ).scalar()
        return int(current or 0) + 1


def sql_question_service(
    *,
    engine: Engine,
    lookup,
    adapter_ref: str = DEFAULT_ADAPTER_REF,
    runtime_version: str = DEFAULT_RUNTIME_VERSION,
) -> QuestionService:
    return QuestionService(
        lineage=SQLQuestionLineagePort(engine=engine),
        runs=SQLQuestionLifecyclePort(engine=engine),
        lookup=lookup,
        state_reader=SQLQuestionStateReaderPort(engine),
        adapter_ref=adapter_ref,
        runtime_version=runtime_version,
    )


def sql_run_work_service(
    *,
    engine: Engine,
    lookup,
) -> RunWorkService:
    return RunWorkService(
        lineage=SQLQuestionLineagePort(engine=engine),
        runs=SQLQuestionLifecyclePort(engine=engine),
        lookup=lookup,
    )


def _spine_question_run_start(record: QuestionRunRecord) -> QuestionRunStartRequest:
    question = None
    if record.question is not None:
        question = SpineQuestionStart(
            conversation_id=record.question.conversation_id,
            tenant_id=record.question.tenant_id,
            read_context_ref=record.question.read_context_ref,
            question_id=record.question.question_id,
            origin_message_ref=record.question.question_id,
            question=record.question.question,
        )
    return QuestionRunStartRequest(
        question=question,
        clarification_response=_spine_clarification_response(
            record.clarification_response
        ),
        run=SpineQuestionRunStart(
            question_id=record.run.question_id,
            run_id=record.run.run_id,
            trigger_kind=record.run.trigger_kind,
            integrated_question=record.run.integrated_question,
            adapter_ref=record.run.adapter_ref,
            runtime_version=record.run.runtime_version,
            previous_run_id=record.run.previous_run_id,
            trigger_clarification_response_run_id=(
                record.run.trigger_clarification_response_run_id
            ),
            trigger_clarification_response_id=(
                record.run.trigger_clarification_response_id
            ),
        ),
    )


def _spine_clarification_response(response):
    if response is None:
        return None
    return SpineClarificationResponseStart(
        response_id=response.response_id,
        run_id=response.run_id,
        clarification_id=response.clarification_id,
        response_text=response.response_text,
    )


def _queued_run_from_work_item(engine: Engine, item: SQLRunWorkItem) -> QueuedRun:
    terminal = terminal_result_for_run(engine, item.run_id)
    return QueuedRun(
        submission=RunSubmission(
            conversation_id=item.conversation_id,
            tenant_id=item.tenant_id,
            question_id=_question_id_for_run(engine, item.run_id),
            run_id=item.run_id,
            question=item.question,
            principal=QuestionPrincipal(
                principal_id=item.user_id,
                tenant_id=item.tenant_id,
                read_context_ref=item.read_context_ref,
                delegated_credential=delegated_credential_from_runtime_context(
                    item.runtime_context
                ),
            ),
            provider=item.provider,
            model_key=item.model_key,
            execution_mode=ExecutionMode.INLINE
            if item.lease_owner == "inline"
            else ExecutionMode.QUEUED,
            conversation_context=dict(item.conversation_context or {}),
            runtime_context=dict(item.runtime_context or {}),
            idempotency_key=item.idempotency_key,
            max_budget_usd=item.max_budget_usd,
            max_thinking_tokens=item.max_thinking_tokens,
        ),
        status=terminal.status if terminal is not None else item.status,
        answer=terminal.answer if terminal is not None else None,
        result_data=terminal.result_data if terminal is not None else None,
        error=terminal.error if terminal is not None else item.last_error or None,
    )


def _terminal_run_from_lineage(
    engine: Engine,
    item: SQLRunWorkItem,
) -> QueuedRun | None:
    if not run_has_terminal_result(engine, item.run_id):
        return None
    return _queued_run_from_work_item(engine, item)


def _question_id_for_run(engine: Engine, run_id: str) -> str:
    run = metadata.tables["fervis_question_run"]
    with sql_connection(engine) as connection:
        value = connection.execute(
            sa.select(run.c.question_id).where(run.c.run_id == run_id)
        ).scalar()
    if value is None:
        raise RuntimeError(f"Fervis run has no lineage question: {run_id}")
    return str(value)
