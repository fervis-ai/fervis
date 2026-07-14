"""Django adapter for the canonical public question-run projection."""

from __future__ import annotations

from typing import Any
from typing_extensions import assert_never

from fervis.lineage.views.django import DjangoLineageQuery
from fervis.lineage.views.service import LineageRootNotFound
from fervis.observability.django import DjangoObservabilityQuery
from fervis.questions.execution_specs import execution_spec_from_storage
from fervis.questions.ports import RerunProgramSpec, ResolveQuestionRunSpec
from fervis.questions.projection import QuestionRunStatus
from fervis.questions.run_views import (
    QuestionRunViewService,
    RunWorkSnapshot,
)
from fervis.run_work.queue.django.models import RunWorkItem


def get_run_view(
    run_id: str,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        projected = QuestionRunViewService(
            lineage_query=DjangoLineageQuery(),
            work_query=DjangoRunWorkQuery(tenant_id=tenant_id),
            observability_query=DjangoObservabilityQuery(),
        ).for_run(run_id)
    except LineageRootNotFound:
        return None
    return projected.to_payload() if projected is not None else None


class DjangoRunWorkQuery:
    def __init__(self, *, tenant_id: str | None = None) -> None:
        self._tenant_id = tenant_id

    def for_run(self, run_id: str) -> RunWorkSnapshot | None:
        rows = RunWorkItem.objects.filter(run_id=run_id)
        if self._tenant_id is not None:
            rows = rows.filter(tenant_id=self._tenant_id)
        item = rows.order_by("created_at").first()
        if item is None:
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


def _display_fields(item: RunWorkItem) -> tuple[str | None, str]:
    spec = execution_spec_from_storage(item.spec_kind, item.execution_spec or {})
    match spec:
        case ResolveQuestionRunSpec(question=question, model_key=model_key):
            return question, model_key
        case RerunProgramSpec():
            return None, ""
        case _ as unreachable:
            assert_never(unreachable)
