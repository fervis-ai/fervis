from datetime import UTC
from types import SimpleNamespace

import pytest

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.utils import timezone

from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.interfaces.django import question_run_ports
from fervis.run_work.queue.django.models import RunWorkItem
from fervis.run_work.queue.django.queue import claim_run_work_items
from fervis.interfaces.django.composition import RUN_CONTEXT_KEY
from fervis.lineage.enums import ClarificationBasis, RunStepKind, RunTriggerKind
from fervis.lineage.models import (
    ClarificationRequest,
    ClarificationResponse,
    Conversation,
    Question,
    QuestionRun,
    RunStep,
    RuntimeErrorDetail,
)
from fervis.questions.contracts import ExecutionMode, QuestionPrincipal
from fervis.questions.ports import (
    QuestionRunRecord,
    QuestionRunStart,
    QuestionRunSubmissionKind,
    LookupExecutionRequest,
    QuestionStart,
    RunSubmission,
)
from fervis.interfaces.django.question_run_ports import (
    DjangoQuestionLineagePort,
    DjangoQuestionLookupPort,
    DjangoQuestionLifecyclePort,
    DjangoQuestionStateReaderPort,
)

SEEDED_USER_ID = "1"


def _seeded_user():
    return get_user_model()._default_manager.get(pk=SEEDED_USER_ID)


@pytest.mark.django_db
def test_django_question_run_port_submits_initial_run_atomically(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()

    result = DjangoQuestionLifecyclePort().submit_question_run_atomically(
        submission=_submission(),
        record=_question_run_record(),
    )

    work_item = RunWorkItem.objects.get(run_id="run_1")
    run = QuestionRun.objects.get(run_id="run_1")
    question = Question.objects.get(question_id="question_1")
    assert result.kind is QuestionRunSubmissionKind.CREATED
    assert result.run.status == "QUEUED"
    assert {
        "work_item_user_id": work_item.user_id,
        "work_item_model_key": work_item.model_key,
        "work_item_budget": str(work_item.max_budget_usd),
        "run_trigger": run.trigger_kind,
        "question_sequence": question.conversation_sequence,
    } == {
        "work_item_user_id": str(user.pk),
        "work_item_model_key": "HAIKU",
        "work_item_budget": "0.2500",
        "run_trigger": RunTriggerKind.INITIAL.value,
        "question_sequence": 1,
    }


@pytest.mark.django_db
def test_django_question_access_uses_conversation_owner_not_work_item_context(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    owner = ReadContextRef(scheme="django_principal", key=str(user.pk))
    execution_context = ReadContextRef(
        scheme="django_principal",
        key="worker-execution-context",
    )
    port = DjangoQuestionLifecyclePort()

    port.submit_question_run_atomically(
        submission=_submission(),
        record=_question_run_record(read_context_ref=owner),
    )
    RunWorkItem.objects.filter(run_id="run_1").update(
        read_context_ref=execution_context.to_storage_dict(),
    )

    access = port.get_question(
        question_id="question_1",
        authority=ReadAuthority(tenant_id="tenant_1", read_context_ref=owner),
    )

    assert (
        Conversation.objects.get(conversation_id="conversation_1").read_context_ref
        == owner.to_storage_dict()
    )
    assert access is not None
    run = DjangoQuestionStateReaderPort().get_question_run(
        "run_1",
        access=access,
    )
    assert run is not None
    assert run["runNumber"] == 1
    assert run["triggerKind"] == RunTriggerKind.INITIAL.value
    assert run["error"] is None


@pytest.mark.django_db
def test_django_question_state_reader_lists_authorized_conversations(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    owner = ReadContextRef(scheme="django_principal", key=str(user.pk))
    other = ReadContextRef(scheme="django_principal", key="other-user")
    port = DjangoQuestionLifecyclePort()
    port.submit_question_run_atomically(
        submission=_submission(
            user=user,
            conversation_id="conversation_older",
            question_id="question_older",
            run_id="run_older",
            question="How many stores were open yesterday?",
        ),
        record=_question_run_record(
            conversation_id="conversation_older",
            question_id="question_older",
            run_id="run_older",
            question="How many stores were open yesterday?",
            read_context_ref=owner,
        ),
    )
    port.submit_question_run_atomically(
        submission=_submission(
            user=user,
            conversation_id="conversation_newer",
            question_id="question_newer_first",
            run_id="run_newer_first",
            question="How many stores are open this month?",
        ),
        record=_question_run_record(
            conversation_id="conversation_newer",
            question_id="question_newer_first",
            run_id="run_newer_first",
            question="How many stores are open this month?",
            read_context_ref=owner,
        ),
    )
    port.terminalize(
        run_id="run_newer_first",
        status="COMPLETED",
        answer="7",
        result_data={"kind": "answer"},
        error=None,
    )
    port.submit_question_run_atomically(
        submission=_submission(
            user=user,
            conversation_id="conversation_newer",
            question_id="question_newer_latest",
            run_id="run_newer_latest",
            question="What about today?",
        ),
        record=_question_run_record(
            conversation_id="conversation_newer",
            question_id="question_newer_latest",
            run_id="run_newer_latest",
            question="What about today?",
            read_context_ref=owner,
        ),
    )
    port.submit_question_run_atomically(
        submission=_submission(
            user=user,
            conversation_id="conversation_other",
            question_id="question_other",
            run_id="run_other",
            question="How many orders came in today?",
        ),
        record=_question_run_record(
            conversation_id="conversation_other",
            question_id="question_other",
            run_id="run_other",
            question="How many orders came in today?",
            read_context_ref=other,
        ),
    )
    QuestionRun.objects.filter(run_id="run_older").update(
        created_at=timezone.datetime(
            2026,
            6,
            26,
            9,
            0,
            tzinfo=UTC,
        )
    )
    Question.objects.filter(question_id="question_newer_first").update(
        conversation_sequence=1,
        created_at=timezone.datetime(
            2026,
            6,
            27,
            10,
            0,
            tzinfo=UTC,
        ),
    )
    Question.objects.filter(question_id="question_newer_latest").update(
        conversation_sequence=2,
        created_at=timezone.datetime(
            2026,
            6,
            27,
            10,
            10,
            tzinfo=UTC,
        ),
    )
    QuestionRun.objects.filter(run_id="run_newer_first").update(
        created_at=timezone.datetime(
            2026,
            6,
            27,
            10,
            0,
            tzinfo=UTC,
        )
    )
    QuestionRun.objects.filter(run_id="run_newer_latest").update(
        created_at=timezone.datetime(
            2026,
            6,
            27,
            10,
            15,
            tzinfo=UTC,
        )
    )

    conversations = DjangoQuestionStateReaderPort().list_conversations(
        authority=ReadAuthority(tenant_id="tenant_1", read_context_ref=owner),
    )

    assert conversations == [
        {
            "conversationId": "conversation_newer",
            "firstQuestion": "How many stores are open this month?",
            "latestQuestionId": "question_newer_latest",
            "currentRunId": "run_newer_latest",
            "status": "RUNNING",
            "runCount": 1,
            "updatedAt": "2026-06-27T10:15:00+00:00",
        },
        {
            "conversationId": "conversation_older",
            "firstQuestion": "How many stores were open yesterday?",
            "latestQuestionId": "question_older",
            "currentRunId": "run_older",
            "status": "RUNNING",
            "runCount": 1,
            "updatedAt": "2026-06-26T09:00:00+00:00",
        },
    ]


@pytest.mark.django_db
def test_django_question_run_port_creates_missing_conversation(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()

    result = DjangoQuestionLifecyclePort().submit_question_run_atomically(
        submission=_submission(
            user=user,
            question_id="question_missing_conversation",
            run_id="run_missing_conversation",
        ),
        record=_question_run_record(
            question_id="question_missing_conversation",
            run_id="run_missing_conversation",
        ),
    )

    assert result.kind is QuestionRunSubmissionKind.CREATED
    assert QuestionRun.objects.filter(run_id="run_missing_conversation").exists()


@pytest.mark.django_db
def test_django_question_run_port_replays_idempotent_submission_without_lineage_duplication(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()

    first = port.submit_question_run_atomically(
        submission=_submission(user=user, idempotency_key="same-key"),
        record=_question_run_record(),
    )
    second = port.submit_question_run_atomically(
        submission=_submission(
            user=user,
            question_id="question_2",
            run_id="run_2",
            idempotency_key="same-key",
        ),
        record=_question_run_record(question_id="question_2", run_id="run_2"),
    )

    assert first.kind is QuestionRunSubmissionKind.CREATED
    assert second.kind is QuestionRunSubmissionKind.EXISTING
    assert second.run.submission.run_id == "run_1"
    assert Question.objects.count() == 1
    assert QuestionRun.objects.count() == 1
    assert RunWorkItem.objects.count() == 1


@pytest.mark.django_db
def test_django_question_run_port_persists_clarification_response_continuation(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()
    port.submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )
    RunWorkItem.objects.filter(run_id="run_1").update(status="FAILED")
    run = QuestionRun.objects.get(run_id="run_1")
    step = RunStep.objects.create(
        step_id="step_1",
        run=run,
        sequence=1,
        step_key="question_contract",
        kind=RunStepKind.MODEL_TURN.value,
    )
    clarification = ClarificationRequest.objects.create(
        clarification_id="clar_1",
        run=run,
        step=step,
        basis=ClarificationBasis.MULTIPLE_MATCHING_ENTITIES.value,
        question_text="Which store?",
    )
    ClarificationResponse.objects.create(
        response_id="clar_response_1",
        run=run,
        clarification=clarification,
        evidence_ref="message:1",
        response_text="ABC Mall",
    )

    result = port.submit_question_run_atomically(
        submission=_submission(
            user=user,
            run_id="run_2",
            question="ABC Mall",
        ),
        record=QuestionRunRecord(
            run=QuestionRunStart(
                question_id="question_1",
                run_id="run_2",
                trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
                integrated_question="ABC Mall",
                adapter_ref="django_drf",
                runtime_version="test-runtime",
                previous_run_id=None,
                trigger_clarification_response_run_id="run_1",
                trigger_clarification_response_id="clar_response_1",
            )
        ),
    )

    continued = QuestionRun.objects.get(run_id="run_2")
    assert result.kind is QuestionRunSubmissionKind.CREATED
    assert continued.question_id == run.question_id
    assert continued.previous_run_id is None
    assert continued.trigger_clarification_response_run_id == "run_1"
    assert continued.trigger_clarification_response_id == "clar_response_1"


@pytest.mark.django_db
def test_django_question_run_port_finds_idempotent_run_without_memory_hydration(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()
    port.submit_question_run_atomically(
        submission=_submission(user=user, idempotency_key="same-key"),
        record=_question_run_record(),
    )

    replay = port.find_idempotent_run(
        authority=ReadAuthority(
            tenant_id="tenant_1",
            read_context_ref=ReadContextRef(
                scheme="django_principal",
                key=str(user.pk),
            ),
        ),
        conversation_id="conversation_1",
        idempotency_key="same-key",
    )

    assert replay is not None
    assert replay.submission.run_id == "run_1"


@pytest.mark.django_db
def test_django_question_run_port_replays_same_run_id_idempotently(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()

    first = port.submit_question_run_atomically(
        submission=_submission(user=user, idempotency_key="same-key"),
        record=_question_run_record(),
    )
    second = port.submit_question_run_atomically(
        submission=_submission(user=user, idempotency_key="same-key"),
        record=_question_run_record(),
    )

    assert first.kind is QuestionRunSubmissionKind.CREATED
    assert second.kind is QuestionRunSubmissionKind.EXISTING
    assert second.run.submission.run_id == "run_1"
    assert Question.objects.count() == 1
    assert QuestionRun.objects.count() == 1
    assert RunWorkItem.objects.count() == 1


@pytest.mark.django_db
def test_django_question_run_port_rejects_same_run_id_without_idempotency_key(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()

    first = port.submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )
    with pytest.raises(IntegrityError):
        port.submit_question_run_atomically(
            submission=_submission(user=user),
            record=_question_run_record(),
        )

    assert first.kind is QuestionRunSubmissionKind.CREATED
    assert Question.objects.count() == 1
    assert QuestionRun.objects.count() == 1
    assert RunWorkItem.objects.count() == 1


@pytest.mark.django_db
def test_django_question_run_port_rejects_same_terminal_run_id_without_idempotency_key(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()
    first = port.submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )
    terminal = port.terminalize(
        run_id="run_1",
        status="FAILED",
        answer=None,
        result_data=None,
        error="fervis_failed",
    )

    with pytest.raises(IntegrityError):
        port.submit_question_run_atomically(
            submission=_submission(user=user),
            record=_question_run_record(),
        )

    assert first.kind is QuestionRunSubmissionKind.CREATED
    assert terminal.status == "FAILED"
    assert Question.objects.count() == 1
    assert QuestionRun.objects.count() == 1
    assert RunWorkItem.objects.count() == 1


@pytest.mark.django_db
def test_django_question_run_port_reports_active_conflict_without_lineage_duplication(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()

    first = port.submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )
    second = port.submit_question_run_atomically(
        submission=_submission(user=user, question_id="question_2", run_id="run_2"),
        record=_question_run_record(question_id="question_2", run_id="run_2"),
    )

    assert first.kind is QuestionRunSubmissionKind.CREATED
    assert second.kind is QuestionRunSubmissionKind.ACTIVE_CONFLICT
    assert second.run.submission.run_id == "run_1"
    assert Question.objects.count() == 1
    assert QuestionRun.objects.count() == 1
    assert RunWorkItem.objects.count() == 1


@pytest.mark.django_db
def test_django_question_run_port_loads_executable_run_for_current_lease(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()
    port.submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )
    claimed = claim_run_work_items(worker_id="worker_1", batch_size=1, lease_seconds=30)

    run = port.load_executable_run(
        run_id="run_1",
        worker_id="worker_1",
        active_attempt=claimed[0].active_attempt,
    )

    assert run.status == "RUNNING"
    assert run.submission.run_id == "run_1"
    assert run.submission.principal.principal_id == str(user.pk)


@pytest.mark.django_db
def test_django_question_run_port_inline_submission_is_not_worker_claimable(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    port = DjangoQuestionLifecyclePort()

    result = port.submit_question_run_atomically(
        submission=_submission(user=user, execution_mode=ExecutionMode.INLINE),
        record=_question_run_record(),
    )
    claimed = claim_run_work_items(
        worker_id="worker_1",
        batch_size=1,
        lease_seconds=30,
    )
    work_item = RunWorkItem.objects.get(run_id="run_1")

    assert result.kind is QuestionRunSubmissionKind.CREATED
    assert result.run.status == "RUNNING"
    assert claimed == []
    assert (
        work_item.status,
        work_item.lease_owner,
        work_item.lease_expires_at is not None,
        work_item.active_attempt,
    ) == (
        "RUNNING",
        "inline",
        True,
        1,
    )


@pytest.mark.django_db
def test_django_question_run_port_expired_inline_submission_is_worker_claimable(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    DjangoQuestionLifecyclePort().submit_question_run_atomically(
        submission=_submission(user=user, execution_mode=ExecutionMode.INLINE),
        record=_question_run_record(),
    )
    RunWorkItem.objects.filter(run_id="run_1").update(
        lease_expires_at=timezone.now() - timezone.timedelta(seconds=1),
    )

    claimed = claim_run_work_items(
        worker_id="worker_1",
        batch_size=1,
        lease_seconds=30,
    )

    work_item = RunWorkItem.objects.get(run_id="run_1")
    assert [item.run_id for item in claimed] == ["run_1"]
    assert (
        work_item.lease_owner,
        work_item.active_attempt,
        work_item.attempt_count,
    ) == ("worker_1", 2, 2)


@pytest.mark.django_db
def test_django_question_run_lineage_context_is_tenant_scoped(
    api_client,
    fervis_foundation_reset,
):
    port = DjangoQuestionLineagePort()

    context = port.conversation_memory_context(
        conversation_id="conversation_1",
        authority=ReadAuthority(
            tenant_id="tenant_2",
            read_context_ref=ReadContextRef(
                scheme="django_principal",
                key=SEEDED_USER_ID,
            ),
        ),
    )

    assert context == {}


@pytest.mark.django_db
def test_django_question_run_lineage_records_failed_runtime_fallback(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    DjangoQuestionLifecyclePort().submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )

    DjangoQuestionLineagePort().record_failed_runtime_fallback(
        run_id="run_1",
        status="FAILED",
        answer=None,
        result_data=None,
        error="fervis_failed",
    )

    assert RuntimeErrorDetail.objects.get(run_id="run_1").message == "fervis_failed"


@pytest.mark.django_db
def test_django_question_run_lookup_port_adapts_runtime_request(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    user = _seeded_user()
    runtime = _FakeRuntime()
    monkeypatch.setattr(question_run_ports, "get_runtime", lambda: runtime)

    result = DjangoQuestionLookupPort().run_lookup(
        LookupExecutionRequest(
            run_id="run_1",
            conversation_id="conversation_1",
            tenant_id="tenant_1",
            question="How many stores are open?",
            read_context_ref=ReadContextRef(
                scheme="django_principal", key=str(user.pk)
            ),
            principal=user,
            provider="anthropic",
            model_key="HAIKU",
            conversation_context={
                "factArtifacts": [],
                RUN_CONTEXT_KEY: {"from_conversation": "yes"},
            },
            runtime_context={"from_request": "yes"},
            max_budget_usd="0.25",
            max_thinking_tokens=128,
            active_attempt=2,
        )
    )

    call = runtime.calls[0]
    assert result.status == "COMPLETED"
    assert result.answer == "42"
    assert result.terminal_lineage_recorded is False
    assert {
        "provider": call["provider"],
        "read_context_ref": call["read_context_ref"],
        "conversation_context": call["conversation_context"],
        "user_context": call["user_context"],
        "active_attempt": call["active_attempt"],
    } == {
        "provider": "anthropic_resolved",
        "read_context_ref": ReadContextRef(scheme="django_principal", key=str(user.pk)),
        "conversation_context": {"factArtifacts": []},
        "user_context": {"from_conversation": "yes", "from_request": "yes"},
        "active_attempt": 2,
    }


@pytest.mark.django_db
def test_queued_django_lookup_uses_submitting_subject_not_worker_identity(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    user = _seeded_user()
    worker_id = "worker-admin"
    runtime = _FakeRuntime()
    lifecycle = DjangoQuestionLifecyclePort()
    monkeypatch.setattr(question_run_ports, "get_runtime", lambda: runtime)

    submitted = lifecycle.submit_question_run_atomically(
        submission=_submission(user=user),
        record=_question_run_record(),
    )
    [work_item] = claim_run_work_items(
        worker_id=worker_id,
        batch_size=1,
        lease_seconds=300,
    )
    queued = lifecycle.load_executable_run(
        run_id=submitted.run.submission.run_id,
        worker_id=worker_id,
        active_attempt=work_item.active_attempt,
    )

    DjangoQuestionLookupPort().run_lookup(
        LookupExecutionRequest(
            run_id=queued.submission.run_id,
            conversation_id=queued.submission.conversation_id,
            tenant_id=queued.submission.tenant_id,
            question=queued.submission.question,
            read_context_ref=queued.submission.principal.read_context_ref,
            principal=queued.submission.principal.principal_id,
            provider=queued.submission.provider,
            model_key=queued.submission.model_key,
            conversation_context=queued.submission.conversation_context,
            runtime_context=queued.submission.runtime_context,
            max_budget_usd=queued.submission.max_budget_usd,
            max_thinking_tokens=queued.submission.max_thinking_tokens,
            active_attempt=work_item.active_attempt,
        )
    )

    assert runtime.calls[0]["read_context_ref"] == ReadContextRef(
        scheme="django_principal",
        key=str(user.pk),
    )
    assert runtime.calls[0]["read_context_ref"].key != worker_id


def _submission(
    *,
    user=None,
    conversation_id: str = "conversation_1",
    question_id: str = "question_1",
    run_id: str = "run_1",
    question: str = "How many stores are open?",
    idempotency_key: str | None = None,
    execution_mode: ExecutionMode = ExecutionMode.QUEUED,
) -> RunSubmission:
    user = user or _seeded_user()
    return RunSubmission(
        conversation_id=conversation_id,
        tenant_id="tenant_1",
        question_id=question_id,
        run_id=run_id,
        question=question,
        principal=QuestionPrincipal(
            principal_id=str(user.pk),
            tenant_id="tenant_1",
            raw=user,
            read_context_ref=ReadContextRef(
                scheme="django_principal", key=str(user.pk)
            ),
        ),
        provider="anthropic",
        model_key="HAIKU",
        execution_mode=execution_mode,
        conversation_context={"factArtifacts": []},
        runtime_context={"request_id": "request_1"},
        idempotency_key=idempotency_key,
        max_budget_usd="0.25",
        max_thinking_tokens=128,
    )


def _question_run_record(
    *,
    conversation_id: str = "conversation_1",
    question_id: str = "question_1",
    run_id: str = "run_1",
    question: str = "How many stores are open?",
    read_context_ref: ReadContextRef | None = None,
) -> QuestionRunRecord:
    return QuestionRunRecord(
        question=QuestionStart(
            conversation_id=conversation_id,
            tenant_id="tenant_1",
            read_context_ref=read_context_ref
            or ReadContextRef(scheme="django_principal", key=SEEDED_USER_ID),
            question_id=question_id,
            question=question,
            principal_id=SEEDED_USER_ID,
        ),
        run=QuestionRunStart(
            question_id=question_id,
            run_id=run_id,
            trigger_kind=RunTriggerKind.INITIAL,
            integrated_question=question,
            adapter_ref="django_drf",
            runtime_version="test-runtime",
        ),
    )


class _FakeProviderBackbone:
    def __init__(self) -> None:
        self.events = []

    def resolve_provider(self, provider, *, model_key):
        self.events.append(("resolve", provider, model_key))
        return f"{provider}_resolved"

    def trace(self, *, event_type, payload, correlation_id):
        self.events.append(("trace", event_type, payload, correlation_id))


class _FakeRuntime:
    def __init__(self) -> None:
        self.provider_backbone = _FakeProviderBackbone()
        self.calls = []

    def run_lookup(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            status="COMPLETED",
            answer="42",
            result_data={"value": 42},
            error="",
            usage={"model": {"input_tokens": 1}},
        )
