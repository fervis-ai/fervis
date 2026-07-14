"""SQL-backed question lifecycle ports."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.lineage.recorder_core import LineageRecorder
from fervis.lineage.recorder import ClarificationResponseWrite
from fervis.lineage.enums import QuestionRunKind, RunResultKind
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.credentials import (
    delegated_credential_from_runtime_context,
)
from fervis.lineage.run_spine import (
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
    ClarificationRunResponse,
    QueuedRun,
    QuestionRunRecord,
    ParsedQuestionRunSubmission,
    QuestionRunSubmissionKind,
    QuestionRunSubmissionResult,
    RunSubmission,
    ResolveQuestionRunSpec,
)
from fervis.lookup.clarification.response import parse_clarification_response
from fervis.lookup.clarification import clarification_response_ref
from fervis.lookup.clarification.payload import clarification_from_payload
from fervis.lookup.clarification.model import ConversationResolutionResponse
from fervis.questions.service import QuestionService, clarification_successor_run
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    parse_stored_program_invocation,
)
from fervis.questions.projection import (
    QuestionMemoryRunSelection,
    QuestionRunProjection,
    QuestionRunStatus,
    QuestionRunSummary,
    project_question_runs,
    select_conversation_memory_runs,
)
from fervis.questions.clarification_state import pending_clarification_ids
from fervis.run_work.service import RunWorkService
from fervis.run_work.contracts import run_wall_clock_duration_ms

from .lineage_query import SQLLineageQuery
from .lineage_store import SQLLineageRecorderStore
from .run_views import get_sql_run_view
from .authority_scope import (
    conversation_is_authorized,
    question_is_authorized,
)
from .terminal import (
    record_runtime_error_result,
    run_has_terminal_result,
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
        context_run_id: str | None = None,
        continuation_run_id: str | None = None,
    ) -> dict[str, Any]:
        if context_run_id is not None and continuation_run_id is not None:
            raise ValueError("memory context accepts one selected run")
        if not conversation_is_authorized(
            self.engine,
            conversation_id=conversation_id,
            authority=authority,
        ):
            return {}
        if context_run_id is not None and not _is_answered_context_run(
            self.engine,
            conversation_id=conversation_id,
            run_id=context_run_id,
            tenant_id=authority.tenant_id,
        ):
            raise PermissionError("context run is not an authorized answered run")
        if continuation_run_id is not None and not _is_terminal_context_run(
            self.engine,
            conversation_id=conversation_id,
            run_id=continuation_run_id,
            tenant_id=authority.tenant_id,
        ):
            raise PermissionError("continuation run is not an authorized terminal run")
        selected_run_id = context_run_id or continuation_run_id
        from fervis.memory.lineage import LineageMemoryArtifactService

        artifacts = LineageMemoryArtifactService(self.lineage_query).for_runs(
            _primary_run_ids_for_conversation(
                self.engine,
                conversation_id,
                context_run_id=selected_run_id,
            )
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

    def load_answered_program_invocation(
        self,
        *,
        run_id: str,
        access: AuthorizedQuestionAccess,
    ) -> StoredProgramInvocation | None:
        access.require_valid()
        question = metadata.tables["fervis_question"]
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                _answered_program_invocation_statement(
                    run_id=run_id,
                    tenant_id=access.tenant_id,
                ).where(question.c.question_id == access.question_id)
            ).first()
        if row is None:
            return None
        return _stored_program_invocation(row)

    def load_prior_answered_invocation(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None:
        question = metadata.tables["fervis_question"]
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                _answered_program_invocation_statement(
                    run_id=run_id,
                    tenant_id=tenant_id,
                ).where(question.c.conversation_id == conversation_id)
            ).first()
        return _stored_program_invocation(row) if row is not None else None

    def load_program_invocation_for_execution(
        self,
        *,
        invocation_id: str,
        run_id: str,
        question_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None:
        invocation = metadata.tables["fervis_program_invocation"]
        program = metadata.tables["fervis_answer_program"]
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        conversation = metadata.tables["fervis_conversation"]
        result = metadata.tables["fervis_run_result"]
        base_run = run.alias("base_run")
        base_result = result.alias("base_result")
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                sa.select(
                    invocation.c.invocation_id,
                    invocation.c.run_id,
                    invocation.c.program_id,
                    invocation.c.bindings_json,
                    invocation.c.kind,
                    invocation.c.base_invocation_id,
                    invocation.c.patch_id,
                    invocation.c.binding_patch_json,
                    invocation.c.revision_id,
                    program.c.canonical_json,
                )
                .select_from(
                    invocation.join(
                        program,
                        invocation.c.program_id == program.c.program_id,
                    )
                    .join(run, invocation.c.run_id == run.c.run_id)
                    .join(question, run.c.question_id == question.c.question_id)
                    .join(
                        conversation,
                        question.c.conversation_id == conversation.c.conversation_id,
                    )
                    .join(base_run, base_run.c.run_id == run.c.base_run_id)
                    .join(base_result, base_result.c.run_id == base_run.c.run_id)
                )
                .where(
                    invocation.c.invocation_id == invocation_id,
                    invocation.c.run_id == run_id,
                    run.c.question_id == question_id,
                    base_run.c.question_id == question_id,
                    conversation.c.tenant_id == tenant_id,
                    base_result.c.result_kind == RunResultKind.ANSWERED.value,
                )
            ).first()
        return _stored_program_invocation(row) if row is not None else None

    def find_idempotent_run(
        self,
        *,
        principal: QuestionPrincipal,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None:
        authority = ReadAuthority.from_principal(principal)
        item = self.queue.find_idempotent_work_item(
            tenant_id=authority.tenant_id,
            principal_id=principal.principal_id,
            read_context_ref=authority.read_context_ref,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            idempotency_scope=(
                f"conversation:{conversation_id}"
                if conversation_id is not None
                else "new_conversation"
            ),
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
        parsed = ParsedQuestionRunSubmission(submission=submission, record=record)
        submission = parsed.submission
        record = parsed.record
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
                if record.program_revision is not None:
                    self.recorder.record_program_revision(record.program_revision)
                if record.program_invocation is not None:
                    self.recorder.record_program_invocation(record.program_invocation)
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

    def respond_to_clarification_atomically(
        self,
        resume: ClarificationRunResponse,
    ) -> QuestionRunSubmissionResult:
        authority = ReadAuthority.from_principal(resume.principal)
        item = self.queue.get_work_item_for_run(resume.run_id)
        if (
            item.tenant_id != authority.tenant_id
            or item.read_context_ref != authority.read_context_ref
            or not question_is_authorized(
                self.engine,
                question_id=resume.question_id,
                authority=authority,
            )
            or _question_id_for_run(self.engine, resume.run_id) != resume.question_id
        ):
            raise PermissionError("clarification does not belong to a resumable run")
        clarification_payload = _clarification_payload(
            self.engine,
            run_id=resume.run_id,
            clarification_id=resume.clarification_id,
        )
        if not isinstance(item.spec, ResolveQuestionRunSpec):
            raise ValueError("clarification can resume only a question lookup")
        response = parse_clarification_response(
            clarification_from_payload(clarification_payload),
            response_id=resume.response_id,
            response_text=resume.response_text,
            selected_option_id=resume.selected_option_id,
            suspended_question_text=item.spec.question,
        )
        with sql_transaction(self.engine):
            self.recorder.record_clarification_response(
                ClarificationResponseWrite(
                    response_id=resume.response_id,
                    run_id=resume.run_id,
                    clarification_id=resume.clarification_id,
                    evidence_ref=clarification_response_ref(resume.response_id),
                    response_text=resume.response_text,
                    selected_option_id=resume.selected_option_id,
                )
            )
            if not (
                isinstance(response, ConversationResolutionResponse)
                and response.annotation is not None
            ):
                resumed = self.queue.resume_from_clarification(
                    run_id=resume.run_id,
                    spec=replace(
                        item.spec,
                        clarification_responses=(
                            *item.spec.clarification_responses,
                            response,
                        ),
                    ),
                    execution_mode=resume.execution_mode.value,
                )
            else:
                current = _queued_run_from_work_item(self.engine, item).submission
                submission, record = clarification_successor_run(
                    current,
                    response=resume,
                    annotation=response,
                )
                self.queue.supersede_waiting_run(resume.run_id)
                enqueued = self.queue.enqueue_run_work_item(submission=submission)
                record_question_run_start(
                    _spine_question_run_start(record),
                    sequence_store=SQLQuestionRunSequenceStore(self.engine),
                    recorder=self.recorder,
                )
                resumed = enqueued.item
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=_queued_run_from_work_item(self.engine, resumed),
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
        del answer, result_data
        self.queue.mark_work_item_terminal(
            run_id=run_id,
            status=status,
            error=error or "",
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        return _queued_run_from_work_item(
            self.engine,
            self.queue.get_work_item_for_run(run_id),
        )

    def wait_for_clarification(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> QueuedRun:
        item = self.queue.wait_for_clarification(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        return _queued_run_from_work_item(self.engine, item)


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
            projection = self._question_run_projection(access.question_id)
            primary = (
                self.get_question_run(projection.primary_run_id, access=access)
                if projection.primary_run_id is not None
                else None
            )
            active = (
                self.get_question_run(projection.active_run_id, access=access)
                if projection.active_run_id is not None
                else None
            )
            projected_state = _required_projected_run_state(
                projection,
                primary=primary,
                active=active,
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
                    "primaryRunId": projection.primary_run_id,
                    "latestRunId": projection.latest_run_id,
                    "activeRunId": projection.active_run_id,
                    "status": str(projected_state["status"]),
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
        projection = self._question_run_projection(access.question_id)
        primary = (
            self.get_question_run(
                projection.primary_run_id,
                access=access,
            )
            if projection.primary_run_id is not None
            else None
        )
        if primary is None:
            raise RuntimeError("question projection is missing its primary run state")
        return {
            "questionId": str(question.question_id),
            "conversationId": str(question.conversation_id),
            "tenantId": str(question.tenant_id),
            "status": str(primary["status"]),
            "primaryRunId": projection.primary_run_id,
            "latestRunId": projection.latest_run_id,
            "activeRunId": projection.active_run_id,
            "question": str(question.original_question),
            "answer": (primary or {}).get("answer"),
            "resultData": (primary or {}).get("resultData"),
            "error": (primary or {}).get("error"),
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
        return get_sql_run_view(
            self.engine,
            run_id,
            tenant_id=access.tenant_id,
        )

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

    def _question_run_projection(self, question_id: str) -> QuestionRunProjection:
        run = metadata.tables["fervis_question_run"]
        result = metadata.tables["fervis_run_result"]
        work = metadata.tables["fervis_run_work_item"]
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(
                    run.c.run_id,
                    run.c.run_number,
                    run.c.kind,
                    work.c.status,
                    result.c.result_kind,
                )
                .select_from(
                    run.outerjoin(work, work.c.run_id == run.c.run_id).outerjoin(
                        result,
                        result.c.run_id == run.c.run_id,
                    )
                )
                .where(run.c.question_id == question_id)
                .order_by(run.c.run_number, run.c.run_id)
            ).all()
        if any(row.status is None for row in rows):
            raise RuntimeError("question run is missing its persisted work state")
        return project_question_runs(
            tuple(
                QuestionRunSummary(
                    run_id=str(row.run_id),
                    run_number=int(row.run_number),
                    kind=QuestionRunKind(str(row.kind)),
                    status=QuestionRunStatus(str(row.status)),
                    answered=(row.result_kind == RunResultKind.ANSWERED.value),
                    terminal=(row.result_kind is not None),
                )
                for row in rows
            )
        )

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


def _required_projected_run_state(
    projection: QuestionRunProjection,
    *,
    primary: dict[str, Any] | None,
    active: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if projection.primary_run_id is None or primary is None:
        raise RuntimeError("question projection is missing its primary run state")
    if projection.active_run_id is not None:
        if active is None:
            raise RuntimeError("question projection is missing its active run state")
        return active
    return primary


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
    program,
    adapter_ref: str = DEFAULT_ADAPTER_REF,
    runtime_version: str = DEFAULT_RUNTIME_VERSION,
) -> QuestionService:
    return QuestionService(
        lineage=SQLQuestionLineagePort(engine=engine),
        runs=SQLQuestionLifecyclePort(engine=engine),
        lookup=lookup,
        program=program,
        state_reader=SQLQuestionStateReaderPort(engine),
        adapter_ref=adapter_ref,
        runtime_version=runtime_version,
    )


def sql_run_work_service(
    *,
    engine: Engine,
    lookup,
    program,
) -> RunWorkService:
    return RunWorkService(
        lineage=SQLQuestionLineagePort(engine=engine),
        runs=SQLQuestionLifecyclePort(engine=engine),
        lookup=lookup,
        program=program,
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
        run=SpineQuestionRunStart(
            question_id=record.run.question_id,
            run_id=record.run.run_id,
            kind=record.run.kind,
            trigger_kind=record.run.trigger_kind,
            adapter_ref=record.run.adapter_ref,
            runtime_version=record.run.runtime_version,
            base_run_id=record.run.base_run_id,
            trigger_clarification_response_id=(
                record.run.trigger_clarification_response_id
            ),
        ),
    )


def _answered_program_invocation_statement(
    *,
    run_id: str,
    tenant_id: str,
):
    invocation = metadata.tables["fervis_program_invocation"]
    program = metadata.tables["fervis_answer_program"]
    run = metadata.tables["fervis_question_run"]
    question = metadata.tables["fervis_question"]
    conversation = metadata.tables["fervis_conversation"]
    result = metadata.tables["fervis_run_result"]
    return (
        sa.select(
            invocation.c.invocation_id,
            invocation.c.run_id,
            invocation.c.program_id,
            invocation.c.bindings_json,
            invocation.c.kind,
            invocation.c.base_invocation_id,
            invocation.c.patch_id,
            invocation.c.binding_patch_json,
            invocation.c.revision_id,
            program.c.canonical_json,
        )
        .select_from(
            invocation.join(
                program,
                invocation.c.program_id == program.c.program_id,
            )
            .join(run, invocation.c.run_id == run.c.run_id)
            .join(question, run.c.question_id == question.c.question_id)
            .join(
                conversation,
                question.c.conversation_id == conversation.c.conversation_id,
            )
            .join(result, result.c.run_id == run.c.run_id)
        )
        .where(
            invocation.c.run_id == run_id,
            conversation.c.tenant_id == tenant_id,
            result.c.result_kind == RunResultKind.ANSWERED.value,
        )
    )


def _stored_program_invocation(row: Any) -> StoredProgramInvocation:
    return parse_stored_program_invocation(
        invocation_id=str(row.invocation_id),
        run_id=str(row.run_id),
        program_id=str(row.program_id),
        canonical_json=str(row.canonical_json),
        bindings_json=str(row.bindings_json),
        kind=str(row.kind),
        base_invocation_id=(
            str(row.base_invocation_id) if row.base_invocation_id is not None else None
        ),
        patch_id=str(row.patch_id) if row.patch_id is not None else None,
        binding_patch_json=(
            str(row.binding_patch_json) if row.binding_patch_json is not None else None
        ),
        revision_id=(str(row.revision_id) if row.revision_id is not None else None),
    )


def _queued_run_from_work_item(engine: Engine, item: SQLRunWorkItem) -> QueuedRun:
    run_view = (
        get_sql_run_view(engine, item.run_id, tenant_id=item.tenant_id)
        if run_has_terminal_result(engine, item.run_id)
        else None
    )
    spec = item.spec
    result_data = (run_view or {}).get("resultData")
    if result_data is None and item.status == "WAITING_FOR_CLARIFICATION":
        result_data = _pending_clarification_result_data(engine, item.run_id)
    return QueuedRun(
        submission=RunSubmission(
            conversation_id=item.conversation_id,
            tenant_id=item.tenant_id,
            question_id=_question_id_for_run(engine, item.run_id),
            run_id=item.run_id,
            principal=QuestionPrincipal(
                principal_id=item.user_id,
                tenant_id=item.tenant_id,
                read_context_ref=item.read_context_ref,
                delegated_credential=delegated_credential_from_runtime_context(
                    spec.runtime_context
                ),
            ),
            spec=spec,
            execution_mode=ExecutionMode.INLINE
            if item.lease_owner == "inline"
            else ExecutionMode.QUEUED,
            idempotency_key=item.idempotency_key,
            idempotency_authority_ref=item.idempotency_authority_ref,
            idempotency_scope=item.idempotency_scope,
        ),
        status=str((run_view or {}).get("status") or item.status),
        answer=(run_view or {}).get("answer"),
        result_data=result_data,
        error=(run_view or {}).get("error") or item.last_error or None,
        duration_ms=(run_view or {}).get("durationMs") or _duration_ms(item),
        active_attempt=item.active_attempt if item.active_attempt > 0 else None,
    )


def _terminal_run_from_lineage(
    engine: Engine,
    item: SQLRunWorkItem,
) -> QueuedRun | None:
    if not run_has_terminal_result(engine, item.run_id):
        return None
    return _queued_run_from_work_item(engine, item)


def _duration_ms(item: SQLRunWorkItem) -> int | None:
    return run_wall_clock_duration_ms(
        created_at=item.created_at,
        completed_at=item.completed_at,
    )


def _question_id_for_run(engine: Engine, run_id: str) -> str:
    run = metadata.tables["fervis_question_run"]
    with sql_connection(engine) as connection:
        value = connection.execute(
            sa.select(run.c.question_id).where(run.c.run_id == run_id)
        ).scalar()
    if value is None:
        raise RuntimeError(f"Fervis run has no lineage question: {run_id}")
    return str(value)


def _clarification_payload(
    engine: Engine,
    *,
    run_id: str,
    clarification_id: str,
) -> dict[str, Any]:
    request = metadata.tables["fervis_clarification_request"]
    response = metadata.tables["fervis_clarification_response"]
    with sql_connection(engine) as connection:
        value = connection.execute(
            sa.select(request.c.payload_json).where(
                request.c.run_id == run_id,
                request.c.clarification_id == clarification_id,
                ~sa.exists(
                    sa.select(response.c.response_id).where(
                        response.c.run_id == run_id,
                        response.c.clarification_id == clarification_id,
                    )
                ),
            )
        ).scalar()
    if not isinstance(value, dict):
        raise ValueError("clarification does not belong to the waiting run")
    return value


def _pending_clarification_result_data(
    engine: Engine,
    run_id: str,
) -> dict[str, Any]:
    request = metadata.tables["fervis_clarification_request"]
    response = metadata.tables["fervis_clarification_response"]
    with sql_connection(engine) as connection:
        request_rows = tuple(
            connection.execute(
                sa.select(request.c.clarification_id, request.c.payload_json)
                .where(request.c.run_id == run_id)
                .order_by(request.c.created_at, request.c.clarification_id)
            )
        )
        response_ids = tuple(
            str(value)
            for value in connection.execute(
                sa.select(response.c.clarification_id).where(
                    response.c.run_id == run_id
                )
            ).scalars()
        )
    pending_ids = frozenset(
        pending_clarification_ids(
            tuple(str(row.clarification_id) for row in request_rows),
            response_ids,
        )
    )
    payloads = [
        row.payload_json
        for row in request_rows
        if str(row.clarification_id) in pending_ids
    ]
    return {
        "kind": "needs_clarification",
        "details": {"clarifications": payloads},
    }


def _primary_run_ids_for_conversation(
    engine: Engine,
    conversation_id: str,
    *,
    context_run_id: str | None = None,
) -> tuple[str, ...]:
    from fervis.memory.lineage import DEFAULT_RECENT_MEMORY_RUN_LIMIT

    question = metadata.tables["fervis_question"]
    with sql_connection(engine) as connection:
        question_ids = tuple(
            str(value)
            for value in connection.execute(
                sa.select(question.c.question_id)
                .where(question.c.conversation_id == conversation_id)
                .order_by(question.c.conversation_sequence.desc())
                .limit(DEFAULT_RECENT_MEMORY_RUN_LIMIT)
            ).scalars()
        )
    reader = SQLQuestionStateReaderPort(engine)
    context_question_id = (
        _question_id_for_run(engine, context_run_id)
        if context_run_id is not None
        else None
    )
    return select_conversation_memory_runs(
        tuple(
            QuestionMemoryRunSelection(
                question_id=question_id,
                primary_run_id=(
                    reader._question_run_projection(question_id).primary_run_id
                ),
            )
            for question_id in reversed(question_ids)
        ),
        selected_run_id=context_run_id,
        selected_question_id=context_question_id,
    )


def _is_answered_context_run(
    engine: Engine,
    *,
    conversation_id: str,
    run_id: str,
    tenant_id: str,
) -> bool:
    return _is_terminal_context_run(
        engine,
        conversation_id=conversation_id,
        run_id=run_id,
        tenant_id=tenant_id,
        result_kind=RunResultKind.ANSWERED,
    )


def _is_terminal_context_run(
    engine: Engine,
    *,
    conversation_id: str,
    run_id: str,
    tenant_id: str,
    result_kind: RunResultKind | None = None,
) -> bool:
    run = metadata.tables["fervis_question_run"]
    question = metadata.tables["fervis_question"]
    conversation = metadata.tables["fervis_conversation"]
    result = metadata.tables["fervis_run_result"]
    with sql_connection(engine) as connection:
        value = connection.execute(
            sa.select(run.c.run_id)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id)
                .join(
                    conversation,
                    question.c.conversation_id == conversation.c.conversation_id,
                )
                .join(result, result.c.run_id == run.c.run_id)
            )
            .where(
                run.c.run_id == run_id,
                question.c.conversation_id == conversation_id,
                conversation.c.tenant_id == tenant_id,
                *(
                    (result.c.result_kind == result_kind.value,)
                    if result_kind is not None
                    else ()
                ),
            )
        ).scalar()
    return value is not None
