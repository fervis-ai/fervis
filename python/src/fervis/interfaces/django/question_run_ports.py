"""Django adapters for framework-neutral question lifecycle ports."""

from __future__ import annotations

from typing import Any

from django.db import transaction

from fervis.run_work.queue.django.queue import (
    ActiveRunConflict,
    enqueue_run_work_item,
    find_idempotent_work_item,
    get_work_item_for_run,
    mark_work_item_terminal,
    mark_work_item_terminal_for_current_lease,
    reconcile_work_item_from_terminal_lineage,
    require_current_run_lease,
)
from fervis.run_work.queue.django.models import RunWorkItem
from fervis.lineage.django.runtime_spine import (
    DJANGO_DRF_ADAPTER_REF,
    record_question_run_start,
    runtime_version_from_settings,
)
from fervis.lineage.models import Conversation, Question, QuestionRun
from fervis.lineage.run_spine import (
    ClarificationResponseStart as SpineClarificationResponseStart,
    QuestionRunStart as SpineQuestionRunStart,
    QuestionRunStartRequest,
    QuestionStart as SpineQuestionStart,
)
from fervis.lineage.django.runtime_failures import record_worker_runtime_error
from fervis.lineage.django.terminal_results import run_has_terminal_result
from fervis.lineage.views.django import DjangoLineageQuery
from fervis.host_api.contracts.authority import (
    ReadAuthority,
    ReadContextRef,
    read_context_ref_matches,
)
from fervis.host_api.credentials import (
    delegated_credential_from_runtime_context,
    runtime_context_with_delegated_credential,
)
from fervis.lookup.orchestration.question_lookup_port import (
    LookupServiceQuestionLookupPort,
)
from fervis.memory.lineage import LineageMemoryArtifactService
from fervis.questions.contracts import QuestionPrincipal
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

from .composition import (
    RUN_CONTEXT_KEY,
    get_runtime,
    lookup_conversation_context,
    runtime_context_from_conversation,
)
from .run_views import get_run_view, with_lineage_usage, with_worker_snapshot


class DjangoQuestionLineagePort:
    def conversation_memory_context(
        self,
        *,
        conversation_id: str,
        authority: ReadAuthority,
    ) -> dict[str, Any]:
        if not Conversation.objects.filter(
            conversation_id=conversation_id,
            tenant_id=authority.tenant_id,
        ).exists():
            return {}
        if not _conversation_owned_by_authority(
            conversation_id=conversation_id,
            authority=authority,
        ):
            return {}
        artifacts = LineageMemoryArtifactService(DjangoLineageQuery()).for_conversation(
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
        del answer, result_data
        record_worker_runtime_error(
            run_id=run_id,
            error_code=error or status or "fervis_failed",
            message=error or status or "fervis_failed",
        )


class DjangoQuestionLifecyclePort:
    def get_question(
        self,
        *,
        question_id: str,
        authority: ReadAuthority,
    ) -> AuthorizedQuestionAccess | None:
        question = (
            Question.objects.select_related("conversation")
            .filter(
                question_id=question_id,
                conversation__tenant_id=authority.tenant_id,
            )
            .first()
        )
        if question is None:
            return None
        if not _conversation_read_context_matches(question.conversation, authority):
            return None
        return AuthorizedQuestionAccess._issue(
            question_id=str(question.question_id),
            conversation_id=str(question.conversation_id),
            tenant_id=str(question.conversation.tenant_id),
            original_question=question.original_question,
            read_context_ref=ReadContextRef.from_storage_dict(
                question.conversation.read_context_ref or {}
            ),
        )

    def authorize_conversation(
        self,
        *,
        conversation_id: str,
        authority: ReadAuthority,
    ) -> None:
        if not Conversation.objects.filter(
            conversation_id=conversation_id,
            tenant_id=authority.tenant_id,
        ).exists():
            return
        if not _conversation_owned_by_authority(
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
        item = find_idempotent_work_item(
            tenant_id=authority.tenant_id,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
        )
        if item is None:
            return None
        if not _conversation_owned_by_authority(
            conversation_id=item.conversation_id,
            authority=authority,
        ):
            return None
        return _queued_run_from_work_item(item)

    def submit_question_run_atomically(
        self,
        *,
        submission: RunSubmission,
        record: QuestionRunRecord,
    ) -> QuestionRunSubmissionResult:
        try:
            with transaction.atomic():
                enqueued = enqueue_run_work_item(
                    run_id=submission.run_id,
                    conversation_id=submission.conversation_id,
                    tenant_id=submission.tenant_id,
                    user_id=_principal_user_id(submission.principal),
                    question=submission.question,
                    provider=submission.provider,
                    model_key=submission.model_key,
                    execution_mode=submission.execution_mode.value,
                    conversation_context=_conversation_context(submission),
                    runtime_context=runtime_context_with_delegated_credential(
                        submission.runtime_context,
                        submission.principal.delegated_credential,
                    ),
                    read_context_ref=submission.principal.read_context_ref.to_storage_dict(),
                    idempotency_key=submission.idempotency_key,
                    max_budget_usd=submission.max_budget_usd,
                    max_thinking_tokens=submission.max_thinking_tokens,
                )
                if not enqueued.created:
                    return QuestionRunSubmissionResult(
                        kind=QuestionRunSubmissionKind.EXISTING,
                        run=_queued_run_from_work_item(enqueued.item),
                    )
                record_question_run_start(_spine_question_run_start(record))
        except ActiveRunConflict as exc:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.ACTIVE_CONFLICT,
                run=_queued_run_from_work_item(get_work_item_for_run(exc.run_id)),
            )
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=_queued_run_from_work_item(enqueued.item),
        )

    def load_executable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        item = get_work_item_for_run(run_id)
        terminal = _terminal_run_from_lineage(item)
        if terminal is not None:
            return terminal
        if active_attempt is None:
            raise ValueError("active_attempt is required for queued run work")
        item = require_current_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
        )
        return _queued_run_from_work_item(item)

    def load_failable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        if active_attempt is None:
            raise ValueError("active_attempt is required for queued run failure")
        item = require_current_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            active_attempt=active_attempt,
            lock=True,
        )
        terminal = _terminal_run_from_lineage(item)
        if terminal is not None:
            return terminal
        return _queued_run_from_work_item(item)

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
        if worker_id and active_attempt is not None:
            mark_work_item_terminal_for_current_lease(
                run_id=run_id,
                worker_id=worker_id,
                active_attempt=active_attempt,
                status=status,
                error=error or "",
            )
        else:
            mark_work_item_terminal(run_id=run_id, status=status, error=error or "")
        item = get_work_item_for_run(run_id)
        run = get_run_view(run_id, tenant_id=item.tenant_id)
        return _queued_run_from_work_item(item, run_view=run)


class DjangoQuestionStateReaderPort:
    def list_conversations(
        self,
        *,
        authority: ReadAuthority,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        conversations = Conversation.objects.filter(
            tenant_id=authority.tenant_id,
        ).order_by("-created_at")
        for conversation in conversations:
            if not _conversation_read_context_matches(conversation, authority):
                continue
            latest_question = (
                Question.objects.filter(conversation=conversation)
                .order_by("-conversation_sequence", "-created_at")
                .first()
            )
            if latest_question is None:
                continue
            first_question = (
                Question.objects.filter(conversation=conversation)
                .order_by("conversation_sequence", "created_at")
                .first()
            )
            access = AuthorizedQuestionAccess._issue(
                question_id=str(latest_question.question_id),
                conversation_id=str(conversation.conversation_id),
                tenant_id=str(conversation.tenant_id),
                original_question=latest_question.original_question,
                read_context_ref=ReadContextRef.from_storage_dict(
                    conversation.read_context_ref or {}
                ),
            )
            latest_run = (
                QuestionRun.objects.filter(question=latest_question)
                .order_by("-run_number", "-created_at")
                .first()
            )
            current = (
                _run_state(str(latest_run.run_id), access=access)
                if latest_run is not None
                else None
            )
            items.append(
                {
                    "conversationId": str(conversation.conversation_id),
                    "firstQuestion": (
                        first_question.original_question
                        if first_question is not None
                        else latest_question.original_question
                    ),
                    "latestQuestionId": str(latest_question.question_id),
                    "currentRunId": (current or {}).get("runId"),
                    "status": str((current or {}).get("status") or "RUNNING"),
                    "runCount": QuestionRun.objects.filter(
                        question=latest_question
                    ).count(),
                    "updatedAt": _iso_datetime(
                        (
                            latest_run.created_at
                            if latest_run is not None
                            else latest_question.created_at
                        )
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
        question = _question(access)
        if question is None:
            return None
        latest_run = (
            QuestionRun.objects.filter(question=question)
            .order_by("-run_number", "-created_at")
            .first()
        )
        if latest_run is None:
            return _question_state_payload(question, current_run=None)
        current = _run_state(
            str(latest_run.run_id),
            access=access,
        )
        return _question_state_payload(question, current_run=current)

    def list_question_runs(
        self,
        *,
        access: AuthorizedQuestionAccess,
    ) -> list[dict[str, Any]]:
        access.require_valid()
        question = _question(access)
        if question is None:
            return []
        run_ids = (
            QuestionRun.objects.filter(question=question)
            .order_by("run_number", "created_at")
            .values_list("run_id", flat=True)
        )
        runs: list[dict[str, Any]] = []
        for run_id in run_ids:
            run = _run_state(
                str(run_id),
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
        run = _run_state(run_id, access=access)
        if run is None or str(run.get("questionId") or "") != access.question_id:
            return None
        return run


class DjangoQuestionLookupPort:
    def run_lookup(self, request, *, progress_sink=None):
        return self._adapter().run_lookup(request, progress_sink=progress_sink)

    def _adapter(self) -> LookupServiceQuestionLookupPort:
        return LookupServiceQuestionLookupPort(
            lookup_service=get_runtime(),
            conversation_context=lambda request: lookup_conversation_context(
                dict(request.conversation_context or {})
            ),
            runtime_context=lambda request: {
                **runtime_context_from_conversation(
                    dict(request.conversation_context or {})
                ),
                **dict(request.runtime_context or {}),
            },
            terminal_lineage_recorded=lambda request: run_has_terminal_result(
                request.run_id,
                tenant_id=request.tenant_id,
            ),
        )


def django_question_service() -> QuestionService:
    return QuestionService(
        lineage=DjangoQuestionLineagePort(),
        runs=DjangoQuestionLifecyclePort(),
        lookup=DjangoQuestionLookupPort(),
        state_reader=DjangoQuestionStateReaderPort(),
        adapter_ref=DJANGO_DRF_ADAPTER_REF,
        runtime_version=runtime_version_from_settings(),
    )


def django_run_work_service() -> RunWorkService:
    return RunWorkService(
        lineage=DjangoQuestionLineagePort(),
        runs=DjangoQuestionLifecyclePort(),
        lookup=DjangoQuestionLookupPort(),
    )


def _conversation_context(submission: RunSubmission) -> dict[str, Any]:
    output = dict(submission.conversation_context or {})
    if submission.runtime_context:
        output[RUN_CONTEXT_KEY] = dict(submission.runtime_context)
    return output


def _question(access: AuthorizedQuestionAccess) -> Question | None:
    rows = Question.objects.select_related("conversation").filter(
        question_id=access.question_id,
        conversation__tenant_id=access.tenant_id,
    )
    return rows.first()


def _run_state(
    run_id: str,
    *,
    access: AuthorizedQuestionAccess,
) -> dict[str, Any] | None:
    item = (
        RunWorkItem.objects.filter(
            run_id=run_id,
            tenant_id=access.tenant_id,
        )
        .order_by("created_at")
        .first()
    )
    if item is None:
        return None
    run = get_run_view(run_id, tenant_id=access.tenant_id)
    if run is None:
        return None
    if str(run.get("questionId") or "") != access.question_id:
        return None
    return with_lineage_usage(with_worker_snapshot(run))


def _conversation_owned_by_authority(
    *,
    conversation_id: str,
    authority: ReadAuthority,
) -> bool:
    conversation = Conversation.objects.filter(
        conversation_id=conversation_id,
        tenant_id=authority.tenant_id,
    ).first()
    return bool(
        conversation is not None
        and _conversation_read_context_matches(conversation, authority)
    )


def _conversation_read_context_matches(
    conversation: Conversation,
    authority: ReadAuthority,
) -> bool:
    return _read_context_ref_matches(
        conversation.read_context_ref,
        authority.read_context_ref,
    )


def _read_context_ref_matches(stored, expected) -> bool:
    return read_context_ref_matches(stored or {}, expected)


def _question_state_payload(
    question: Question,
    *,
    current_run: dict[str, Any] | None,
) -> dict[str, Any]:
    status = str((current_run or {}).get("status") or "RUNNING")
    return {
        "questionId": str(question.question_id),
        "conversationId": str(question.conversation_id),
        "tenantId": str(question.conversation.tenant_id),
        "status": status,
        "currentRunId": (current_run or {}).get("runId"),
        "question": question.original_question,
        "answer": (current_run or {}).get("answer"),
        "resultData": (current_run or {}).get("resultData"),
        "error": (current_run or {}).get("error"),
    }


def _iso_datetime(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


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
        selected_option_id=response.selected_option_id,
    )


def _queued_run_from_work_item(
    item,
    *,
    run_view: dict[str, Any] | None = None,
) -> QueuedRun:
    if run_view is None and run_has_terminal_result(
        item.run_id,
        tenant_id=item.tenant_id,
    ):
        run_view = get_run_view(item.run_id, tenant_id=item.tenant_id)
    return QueuedRun(
        submission=_submission_from_work_item(item),
        status=str((run_view or {}).get("status") or item.status),
        answer=(run_view or {}).get("answer"),
        result_data=(run_view or {}).get("resultData"),
        error=(run_view or {}).get("error") or item.last_error or None,
    )


def _terminal_run_from_lineage(item) -> QueuedRun | None:
    if not run_has_terminal_result(item.run_id, tenant_id=item.tenant_id):
        return None
    run = get_run_view(item.run_id, tenant_id=item.tenant_id)
    if run is None:
        raise RuntimeError("terminal Fervis run is missing interface view")
    reconcile_work_item_from_terminal_lineage(item)
    return _queued_run_from_work_item(
        get_work_item_for_run(item.run_id),
        run_view=run,
    )


def _submission_from_work_item(item) -> RunSubmission:
    return RunSubmission(
        conversation_id=item.conversation_id,
        tenant_id=item.tenant_id,
        question_id=_question_id_for_run(item.run_id),
        run_id=item.run_id,
        question=item.question,
        principal=QuestionPrincipal(
            principal_id=str(item.user_id),
            tenant_id=item.tenant_id,
            raw=None,
            read_context_ref=item.read_context_ref,
            delegated_credential=delegated_credential_from_runtime_context(
                item.runtime_context
            ),
        ),
        provider=item.provider,
        model_key=item.model_key,
        conversation_context=dict(item.conversation_context or {}),
        runtime_context=dict(item.runtime_context or {}),
        idempotency_key=item.idempotency_key,
        max_budget_usd=item.max_budget_usd,
        max_thinking_tokens=item.max_thinking_tokens,
    )


def _principal_user_id(principal: QuestionPrincipal) -> str:
    raw = principal.raw
    pk = getattr(raw, "pk", None)
    return str(pk or principal.principal_id)


def _question_id_for_run(run_id: str) -> str:
    run = QuestionRun.objects.select_related("question").filter(run_id=run_id).first()
    if run is None:
        return ""
    return str(run.question_id)
