"""SQL adapter for the canonical public question-run projection."""

from __future__ import annotations

from typing import Any, assert_never

from sqlalchemy.engine import Engine

from fervis.lineage.views.service import LineageRootNotFound
from fervis.questions.ports import RerunProgramSpec, ResolveQuestionRunSpec
from fervis.questions.projection import QuestionRunStatus
from fervis.questions.run_views import QuestionRunViewService, RunWorkSnapshot

from .lineage_query import SQLLineageQuery
from .observability_query import SQLObservabilityQuery
from .work_items import SQLRunWorkItem, SQLWorkItemQueue


def get_sql_run_view(
    engine: Engine,
    run_id: str,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        projected = QuestionRunViewService(
            lineage_query=SQLLineageQuery(engine),
            work_query=SQLRunWorkQuery(engine, tenant_id=tenant_id),
            observability_query=SQLObservabilityQuery(engine),
        ).for_run(run_id)
    except LineageRootNotFound:
        return None
    return projected.to_payload() if projected is not None else None


class SQLRunWorkQuery:
    def __init__(self, engine: Engine, *, tenant_id: str | None = None) -> None:
        self._queue = SQLWorkItemQueue(engine)
        self._tenant_id = tenant_id

    def for_run(self, run_id: str) -> RunWorkSnapshot | None:
        try:
            item = self._queue.get_work_item_for_run(run_id)
        except LookupError:
            return None
        if self._tenant_id is not None and item.tenant_id != self._tenant_id:
            return None
        question_override, model_key = _display_fields(item)
        return RunWorkSnapshot(
            run_id=item.run_id,
            conversation_id=item.conversation_id,
            tenant_id=item.tenant_id,
            status=QuestionRunStatus(item.status),
            question_override=question_override,
            model_key=model_key,
            attempt_count=item.attempt_count,
            active_attempt=item.active_attempt,
            lease_owner=item.lease_owner,
            lease_expires_at=item.lease_expires_at,
            last_error=item.last_error,
            created_at=item.created_at,
            started_at=item.started_at,
            completed_at=item.completed_at,
        )


def _display_fields(item: SQLRunWorkItem) -> tuple[str | None, str]:
    match item.spec:
        case ResolveQuestionRunSpec(question=question, model_key=model_key):
            return question, model_key
        case RerunProgramSpec():
            return None, ""
        case _ as unreachable:
            assert_never(unreachable)
