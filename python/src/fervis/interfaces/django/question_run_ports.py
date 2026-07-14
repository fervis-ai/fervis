"""Django adapters for framework-neutral question lifecycle ports."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

from django.db import transaction
from django.utils import timezone

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
from fervis.run_work.queue.django.models import RunWorkStatus
from fervis.lineage.django.runtime_spine import (
    DJANGO_DRF_ADAPTER_REF,
    record_question_run_start,
    runtime_version_from_settings,
)
from fervis.lineage.enums import QuestionRunKind, RunResultKind
from fervis.lineage.models import (
    Conversation,
    ProgramInvocation,
    Question,
    QuestionRun,
    RunResult,
    ClarificationRequest,
    ClarificationResponse,
)
from fervis.lineage.run_spine import (
    QuestionRunStart as SpineQuestionRunStart,
    QuestionRunStartRequest,
    QuestionStart as SpineQuestionStart,
)
from fervis.lineage.django.runtime_failures import record_worker_runtime_error
from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.django.terminal_results import run_has_terminal_result
from fervis.lineage.views.django import DjangoLineageQuery
from fervis.host_api.contracts.authority import (
    ReadAuthority,
    ReadContextRef,
    read_context_ref_matches,
)
from fervis.host_api.credentials import (
    delegated_credential_from_runtime_context,
)
from fervis.lookup.orchestration.question_lookup_port import (
    LookupServiceQuestionLookupPort,
)
from fervis.lookup.clarification import clarification_response_ref
from fervis.lookup.orchestration.question_program_port import AnswerProgramQuestionPort
from fervis.lookup.orchestration.program_service import AnswerProgramService
from fervis.host_api.context import get_host_api_context
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    parse_stored_program_invocation,
)
from fervis.memory.lineage import (
    DEFAULT_RECENT_MEMORY_RUN_LIMIT,
    LineageMemoryArtifactService,
)
from fervis.questions.contracts import ExecutionMode, QuestionPrincipal
from fervis.questions.execution_specs import execution_spec_from_storage
from fervis.questions.execution_specs import execution_spec_to_storage_dict
from fervis.lookup.clarification.payload import clarification_from_payload
from fervis.lineage.recorder import ClarificationResponseWrite
from fervis.questions.projection import (
    QuestionMemoryRunSelection,
    QuestionRunProjection,
    QuestionRunStatus,
    QuestionRunSummary,
    project_question_runs,
    select_conversation_memory_runs,
)
from fervis.questions.clarification_state import pending_clarification_ids
from fervis.questions.ports import (
    AuthorizedQuestionAccess,
    ClarificationRunResponse,
    QueuedRun,
    QuestionRunRecord,
    ParsedQuestionRunSubmission,
    QuestionRunSubmissionKind,
    QuestionRunSubmissionResult,
    ResolveQuestionRunSpec,
    RunSubmission,
)
from fervis.lookup.clarification.response import parse_clarification_response
from fervis.lookup.clarification.model import ConversationResolutionResponse
from fervis.questions.service import QuestionService, clarification_successor_run
from fervis.run_work.service import RunWorkService
from fervis.run_work.contracts import run_wall_clock_duration_ms

from .composition import (
    get_runtime,
    lookup_conversation_context,
    runtime_context_from_conversation,
)
from .run_views import get_run_view


class DjangoQuestionLineagePort:
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
        if (
            context_run_id is not None
            and not QuestionRun.objects.filter(
                run_id=context_run_id,
                question__conversation_id=conversation_id,
                question__conversation__tenant_id=authority.tenant_id,
                run_result__result_kind=RunResultKind.ANSWERED.value,
            ).exists()
        ):
            raise PermissionError("context run is not an authorized answered run")
        if (
            continuation_run_id is not None
            and not QuestionRun.objects.filter(
                run_id=continuation_run_id,
                question__conversation_id=conversation_id,
                question__conversation__tenant_id=authority.tenant_id,
                run_result__isnull=False,
            ).exists()
        ):
            raise PermissionError("continuation run is not an authorized terminal run")
        selected_run_id = context_run_id or continuation_run_id
        artifacts = LineageMemoryArtifactService(DjangoLineageQuery()).for_runs(
            _primary_run_ids_for_conversation(
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

    def load_answered_program_invocation(
        self,
        *,
        run_id: str,
        access: AuthorizedQuestionAccess,
    ) -> StoredProgramInvocation | None:
        access.require_valid()
        record = (
            _answered_program_invocations(
                run_id=run_id,
                tenant_id=access.tenant_id,
            )
            .filter(run__question_id=access.question_id)
            .first()
        )
        if record is None:
            return None
        return _stored_program_invocation(record)

    def load_prior_answered_invocation(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None:
        record = (
            _answered_program_invocations(run_id=run_id, tenant_id=tenant_id)
            .filter(run__question__conversation_id=conversation_id)
            .first()
        )
        return _stored_program_invocation(record) if record is not None else None

    def load_program_invocation_for_execution(
        self,
        *,
        invocation_id: str,
        run_id: str,
        question_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None:
        record = (
            ProgramInvocation.objects.select_related("program")
            .filter(
                invocation_id=invocation_id,
                run_id=run_id,
                run__question_id=question_id,
                run__question__conversation__tenant_id=tenant_id,
                run__base_run__question_id=question_id,
                run__base_run__run_result__result_kind=RunResultKind.ANSWERED.value,
            )
            .first()
        )
        return _stored_program_invocation(record) if record is not None else None

    def find_idempotent_run(
        self,
        *,
        principal: QuestionPrincipal,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None:
        authority = ReadAuthority.from_principal(principal)
        item = find_idempotent_work_item(
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
        parsed = ParsedQuestionRunSubmission(submission=submission, record=record)
        submission = parsed.submission
        record = parsed.record
        try:
            with transaction.atomic():
                recorder = DjangoLineageRecorder()
                enqueued = enqueue_run_work_item(
                    submission=submission,
                )
                if not enqueued.created:
                    return QuestionRunSubmissionResult(
                        kind=QuestionRunSubmissionKind.EXISTING,
                        run=_queued_run_from_work_item(enqueued.item),
                    )
                record_question_run_start(
                    _spine_question_run_start(record),
                    recorder=recorder,
                )
                if record.program_revision is not None:
                    recorder.record_program_revision(record.program_revision)
                if record.program_invocation is not None:
                    recorder.record_program_invocation(record.program_invocation)
        except ActiveRunConflict as exc:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.ACTIVE_CONFLICT,
                run=_queued_run_from_work_item(get_work_item_for_run(exc.run_id)),
            )
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=_queued_run_from_work_item(enqueued.item),
        )

    def respond_to_clarification_atomically(
        self,
        resume: ClarificationRunResponse,
    ) -> QuestionRunSubmissionResult:
        authority = ReadAuthority.from_principal(resume.principal)
        with transaction.atomic():
            item = RunWorkItem.objects.select_for_update().get(run_id=resume.run_id)
            _require_resumable_clarification(item, resume=resume, authority=authority)
            clarification = ClarificationRequest.objects.select_for_update().get(
                clarification_id=resume.clarification_id,
                run_id=resume.run_id,
                responses__isnull=True,
            )
            spec = execution_spec_from_storage(
                item.spec_kind,
                item.execution_spec or {},
            )
            if not isinstance(spec, ResolveQuestionRunSpec):
                raise ValueError("clarification can resume only a question lookup")
            response = parse_clarification_response(
                clarification_from_payload(clarification.payload_json or {}),
                response_id=resume.response_id,
                response_text=resume.response_text,
                selected_option_id=resume.selected_option_id,
                suspended_question_text=spec.question,
            )
            DjangoLineageRecorder().record_clarification_response(
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
                resumed_spec = replace(
                    spec,
                    clarification_responses=(
                        *spec.clarification_responses,
                        response,
                    ),
                )
                item.execution_spec = execution_spec_to_storage_dict(resumed_spec)
                _resume_work_item(item, execution_mode=resume.execution_mode)
                queued_item = item
            else:
                current = _queued_run_from_work_item(item).submission
                submission, record = clarification_successor_run(
                    current,
                    response=resume,
                    annotation=response,
                )
                item.status = RunWorkStatus.SUPERSEDED
                item.lease_owner = None
                item.lease_expires_at = None
                item.next_attempt_at = None
                item.completed_at = timezone.now()
                item.save()
                enqueued = enqueue_run_work_item(submission=submission)
                record_question_run_start(
                    _spine_question_run_start(record),
                    recorder=DjangoLineageRecorder(),
                )
                queued_item = enqueued.item
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=_queued_run_from_work_item(queued_item),
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

    def wait_for_clarification(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> QueuedRun:
        updated = RunWorkItem.objects.filter(
            run_id=run_id,
            status=RunWorkStatus.RUNNING,
            lease_owner=worker_id,
            active_attempt=active_attempt,
        ).update(
            status=RunWorkStatus.WAITING_FOR_CLARIFICATION,
            completed_at=None,
            lease_owner=None,
            lease_expires_at=None,
            last_error="",
        )
        if updated != 1:
            raise ValueError("clarification wait requires the current run lease")
        item = get_work_item_for_run(run_id)
        return _queued_run_from_work_item(item)


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
            projection = _question_run_projection(latest_question)
            primary = (
                _run_state(projection.primary_run_id, access=access)
                if projection.primary_run_id is not None
                else None
            )
            active = (
                _run_state(projection.active_run_id, access=access)
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
                    "conversationId": str(conversation.conversation_id),
                    "firstQuestion": (
                        first_question.original_question
                        if first_question is not None
                        else latest_question.original_question
                    ),
                    "latestQuestionId": str(latest_question.question_id),
                    "primaryRunId": projection.primary_run_id,
                    "latestRunId": projection.latest_run_id,
                    "activeRunId": projection.active_run_id,
                    "status": str(projected_state["status"]),
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
        projection = _question_run_projection(question)
        primary = (
            _run_state(projection.primary_run_id, access=access)
            if projection.primary_run_id is not None
            else None
        )
        return _question_state_payload(
            question,
            primary_run=primary,
            projection=projection,
        )

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


class DjangoQuestionProgramPort:
    def run_program(self, request, *, progress_sink=None):
        return self._adapter().run_program(request, progress_sink=progress_sink)

    def _adapter(self) -> AnswerProgramQuestionPort:
        return AnswerProgramQuestionPort(
            program_service=AnswerProgramService(
                host_api_context=get_host_api_context(),
                lineage_recorder=DjangoLineageRecorder(),
            ),
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
        program=DjangoQuestionProgramPort(),
        state_reader=DjangoQuestionStateReaderPort(),
        adapter_ref=DJANGO_DRF_ADAPTER_REF,
        runtime_version=runtime_version_from_settings(),
    )


def django_run_work_service() -> RunWorkService:
    return RunWorkService(
        lineage=DjangoQuestionLineagePort(),
        runs=DjangoQuestionLifecyclePort(),
        lookup=DjangoQuestionLookupPort(),
        program=DjangoQuestionProgramPort(),
    )


def _question(access: AuthorizedQuestionAccess) -> Question | None:
    rows = Question.objects.select_related("conversation").filter(
        question_id=access.question_id,
        conversation__tenant_id=access.tenant_id,
    )
    return rows.first()


def _answered_program_invocations(*, run_id: str, tenant_id: str):
    return ProgramInvocation.objects.select_related("program").filter(
        run_id=run_id,
        run__question__conversation__tenant_id=tenant_id,
        run__run_result__result_kind=RunResultKind.ANSWERED.value,
    )


def _stored_program_invocation(
    record: ProgramInvocation,
) -> StoredProgramInvocation:
    return parse_stored_program_invocation(
        invocation_id=record.invocation_id,
        run_id=record.run_id,
        program_id=record.program_id,
        canonical_json=record.program.canonical_json,
        bindings_json=record.bindings_json,
        kind=record.kind,
        base_invocation_id=record.base_invocation_id,
        patch_id=record.patch_id,
        binding_patch_json=record.binding_patch_json,
        revision_id=record.revision_id,
    )


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
    return run


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
    primary_run: dict[str, Any] | None,
    projection: QuestionRunProjection,
) -> dict[str, Any]:
    if primary_run is None:
        raise RuntimeError("question projection is missing its primary run state")
    status = str(primary_run["status"])
    return {
        "questionId": str(question.question_id),
        "conversationId": str(question.conversation_id),
        "tenantId": str(question.conversation.tenant_id),
        "status": status,
        "primaryRunId": projection.primary_run_id,
        "latestRunId": projection.latest_run_id,
        "activeRunId": projection.active_run_id,
        "question": question.original_question,
        "answer": (primary_run or {}).get("answer"),
        "resultData": (primary_run or {}).get("resultData"),
        "error": (primary_run or {}).get("error"),
    }


def _question_run_projection(question: Question) -> QuestionRunProjection:
    runs = tuple(
        QuestionRun.objects.filter(question=question).order_by("run_number", "run_id")
    )
    if not runs:
        return project_question_runs(())
    result_kinds = {
        str(run_id): str(result_kind)
        for run_id, result_kind in RunResult.objects.filter(
            run_id__in={run.run_id for run in runs},
        ).values_list("run_id", "result_kind")
    }
    statuses = {
        str(run_id): str(status)
        for run_id, status in RunWorkItem.objects.filter(
            run_id__in={run.run_id for run in runs}
        ).values_list("run_id", "status")
    }
    run_ids = {str(run.run_id) for run in runs}
    missing_work_ids = run_ids - set(statuses)
    if missing_work_ids:
        raise RuntimeError("question run is missing its persisted work state")
    return project_question_runs(
        tuple(
            QuestionRunSummary(
                run_id=str(run.run_id),
                run_number=run.run_number,
                kind=QuestionRunKind(run.kind),
                status=QuestionRunStatus(statuses[str(run.run_id)]),
                answered=(
                    result_kinds.get(str(run.run_id)) == RunResultKind.ANSWERED.value
                ),
                terminal=str(run.run_id) in result_kinds,
            )
            for run in runs
        )
    )


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


def _primary_run_ids_for_conversation(
    conversation_id: str,
    *,
    context_run_id: str | None = None,
) -> tuple[str, ...]:
    questions = tuple(
        Question.objects.filter(conversation_id=conversation_id).order_by(
            "-conversation_sequence"
        )[:DEFAULT_RECENT_MEMORY_RUN_LIMIT]
    )
    context_question_id = (
        str(
            QuestionRun.objects.values_list("question_id", flat=True).get(
                run_id=context_run_id
            )
        )
        if context_run_id is not None
        else None
    )
    return select_conversation_memory_runs(
        tuple(
            QuestionMemoryRunSelection(
                question_id=str(question.question_id),
                primary_run_id=_question_run_projection(question).primary_run_id,
            )
            for question in reversed(questions)
        ),
        selected_run_id=context_run_id,
        selected_question_id=context_question_id,
    )


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
    result_data = (run_view or {}).get("resultData")
    if result_data is None and item.status == RunWorkStatus.WAITING_FOR_CLARIFICATION:
        result_data = _pending_clarification_result_data(item.run_id)
    return QueuedRun(
        submission=_submission_from_work_item(item),
        status=str((run_view or {}).get("status") or item.status),
        answer=(run_view or {}).get("answer"),
        result_data=result_data,
        error=(run_view or {}).get("error") or item.last_error or None,
        duration_ms=_queued_run_duration_ms(item=item, run_view=run_view),
        active_attempt=(
            int(item.active_attempt) if int(item.active_attempt or 0) > 0 else None
        ),
    )


def _pending_clarification_result_data(run_id: str) -> dict[str, Any]:
    request_rows = tuple(
        ClarificationRequest.objects.filter(run_id=run_id)
        .order_by("created_at", "clarification_id")
        .values_list("clarification_id", "payload_json")
    )
    response_ids = tuple(
        str(value)
        for value in ClarificationResponse.objects.filter(run_id=run_id).values_list(
            "clarification_id", flat=True
        )
    )
    pending_ids = frozenset(
        pending_clarification_ids(
            tuple(str(clarification_id) for clarification_id, _ in request_rows),
            response_ids,
        )
    )
    payloads = tuple(
        payload
        for clarification_id, payload in request_rows
        if str(clarification_id) in pending_ids
    )
    return {
        "kind": "needs_clarification",
        "details": {"clarifications": list(payloads)},
    }


def _queued_run_duration_ms(
    *,
    item,
    run_view: dict[str, Any] | None,
) -> int | None:
    projected_duration = (run_view or {}).get("durationMs")
    if projected_duration is not None:
        return int(projected_duration)
    created_at = item.created_at
    completed_at = getattr(item, "completed_at", None)
    return run_wall_clock_duration_ms(
        created_at=created_at,
        completed_at=completed_at,
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
    spec = execution_spec_from_storage(item.spec_kind, item.execution_spec or {})
    return RunSubmission(
        conversation_id=item.conversation_id,
        tenant_id=item.tenant_id,
        question_id=_question_id_for_run(item.run_id),
        run_id=item.run_id,
        principal=QuestionPrincipal(
            principal_id=str(item.user_id),
            tenant_id=item.tenant_id,
            raw=None,
            read_context_ref=item.read_context_ref,
            delegated_credential=delegated_credential_from_runtime_context(
                spec.runtime_context
            ),
        ),
        spec=spec,
        idempotency_key=item.idempotency_key,
        idempotency_authority_ref=item.idempotency_authority_ref,
        idempotency_scope=item.idempotency_scope,
    )


def _question_id_for_run(run_id: str) -> str:
    run = QuestionRun.objects.select_related("question").filter(run_id=run_id).first()
    if run is None:
        return ""
    return str(run.question_id)


def _require_resumable_clarification(
    item: RunWorkItem,
    *,
    resume: ClarificationRunResponse,
    authority: ReadAuthority,
) -> None:
    question_id = _question_id_for_run(resume.run_id)
    if (
        item.status != RunWorkStatus.WAITING_FOR_CLARIFICATION
        or question_id != resume.question_id
        or item.tenant_id != authority.tenant_id
        or not _read_context_ref_matches(
            item.read_context_ref, authority.read_context_ref
        )
    ):
        raise PermissionError("clarification does not belong to a resumable run")


def _resume_work_item(
    item: RunWorkItem,
    *,
    execution_mode: ExecutionMode,
) -> None:
    inline = execution_mode is ExecutionMode.INLINE
    now = timezone.now()
    next_attempt = int(item.attempt_count) + 1
    item.status = RunWorkStatus.RUNNING if inline else RunWorkStatus.QUEUED
    item.attempt_count = next_attempt if inline else item.attempt_count
    item.active_attempt = next_attempt if inline else 0
    item.max_attempts = max(int(item.max_attempts), next_attempt + 1)
    item.lease_owner = "inline" if inline else None
    item.lease_expires_at = now + timedelta(seconds=300) if inline else None
    item.next_attempt_at = None
    item.completed_at = None
    item.last_error = ""
    item.save()
