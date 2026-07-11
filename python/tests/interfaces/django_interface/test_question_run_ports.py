from datetime import UTC
from types import SimpleNamespace

import pytest

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.utils import timezone

from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.interfaces.django import question_run_ports
from fervis.run_work.queue.django.models import RunWorkItem
from fervis.run_work.queue.django.queue import (
    claim_run_work_items,
    mark_work_item_terminal,
)
from fervis.interfaces.django.composition import RUN_CONTEXT_KEY
from fervis.lineage.enums import (
    ProgramInvocationKind,
    QuestionRunKind,
    RunResultKind,
    RunStepKind,
    RunTriggerKind,
)
from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.recorder import RunResultWrite
from fervis.lineage.models import (
    ClarificationRequest,
    ClarificationResponse,
    Conversation,
    ProgramInvocation,
    ProgramRevision,
    Question,
    QuestionRun,
    RunStep,
    RuntimeErrorDetail,
)
from fervis.lookup.clarification import ClarificationNeed, ClarificationReason
from fervis.questions.contracts import (
    AskRequest,
    ExecutionMode,
    QuestionPrincipal,
    RerunQuestionRequest,
)
from fervis.questions.ports import (
    QuestionRunRecord,
    QuestionRunStart,
    QuestionRunSubmissionKind,
    LookupExecutionRequest,
    ResolveQuestionRunSpec,
    RunExecutionSpecKind,
    QuestionStart,
    RunSubmission,
)
from fervis.interfaces.django.question_run_ports import (
    DjangoQuestionLineagePort,
    DjangoQuestionLookupPort,
    DjangoQuestionLifecyclePort,
    DjangoQuestionStateReaderPort,
    django_question_service,
)
from fervis.lookup.answer_program import (
    BindingPatch,
    BindingProvenance,
    BindingProvenanceKind,
    CapabilityApplication,
    ParameterBinding,
    SetParameter,
    answer_program_id,
)
from fervis.lookup.answer_program.revisions import apply_capability
from fervis.lookup.answer_program.persistence import (
    program_invocation,
    program_invocation_bundle,
    program_revision_bundle,
)
from fervis.lookup.answer_program.values import FactValue

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
        "work_item_spec_kind": work_item.spec_kind,
        "work_item_model_key": work_item.execution_spec["model_key"],
        "work_item_budget": work_item.execution_spec["max_budget_usd"],
        "run_trigger": run.trigger_kind,
        "question_sequence": question.conversation_sequence,
    } == {
        "work_item_user_id": str(user.pk),
        "work_item_spec_kind": RunExecutionSpecKind.RESOLVE_QUESTION.value,
        "work_item_model_key": "HAIKU",
        "work_item_budget": "0.25",
        "run_trigger": RunTriggerKind.INITIAL.value,
        "question_sequence": 1,
    }


@pytest.mark.django_db
def test_django_rerun_persists_child_invocation_and_queue_atomically(
    api_client,
    fervis_foundation_reset,
):
    service, principal, base = _django_answered_program_base()

    rerun = service.rerun_question(_django_rerun_request(base, principal=principal))

    run = QuestionRun.objects.get(run_id=rerun.run_id)
    invocation = ProgramInvocation.objects.get(run_id=rerun.run_id)
    work_item = RunWorkItem.objects.get(run_id=rerun.run_id)
    assert rerun.status == "QUEUED"
    assert (
        run.kind,
        run.trigger_kind,
        run.base_run_id,
        work_item.spec_kind,
    ) == (
        QuestionRunKind.DETERMINISTIC.value,
        RunTriggerKind.RERUN.value,
        base.run_id,
        RunExecutionSpecKind.RERUN_PROGRAM.value,
    )
    assert invocation.patch_id.startswith("bp_")
    base_invocation = ProgramInvocation.objects.get(run_id=base.run_id)
    assert invocation.kind == ProgramInvocationKind.RERUN_PROGRAM.value
    assert invocation.base_invocation_id == base_invocation.invocation_id


@pytest.mark.django_db
def test_django_rerun_submission_rolls_back_every_child_record_on_failure(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    service, principal, base = _django_answered_program_base(
        program_name="capability",
        binding_set_name="capability",
        catalog_name="sales_channel",
    )
    before = (
        QuestionRun.objects.count(),
        RunWorkItem.objects.count(),
        ProgramInvocation.objects.count(),
        ProgramRevision.objects.count(),
    )

    def fail_on_child_invocation(_self, _bundle):
        raise RuntimeError("injected rerun invocation persistence failure")

    monkeypatch.setattr(
        DjangoLineageRecorder,
        "record_program_invocation",
        fail_on_child_invocation,
    )

    with pytest.raises(
        RuntimeError,
        match="injected rerun invocation persistence failure",
    ):
        service.rerun_question(
            _django_capability_rerun_request(base, principal=principal)
        )

    assert (
        QuestionRun.objects.count(),
        RunWorkItem.objects.count(),
        ProgramInvocation.objects.count(),
        ProgramRevision.objects.count(),
    ) == before


@pytest.mark.django_db
def test_django_capability_rerun_persists_one_revision_and_child_invocation(
    api_client,
    fervis_foundation_reset,
):
    service, principal, base = _django_answered_program_base(
        program_name="capability",
        binding_set_name="capability",
        catalog_name="sales_channel",
    )

    rerun = service.rerun_question(
        _django_capability_rerun_request(base, principal=principal)
    )

    invocation = ProgramInvocation.objects.get(run_id=rerun.run_id)
    revision = ProgramRevision.objects.get(revision_id=invocation.revision_id)
    assert rerun.status == "QUEUED"
    assert invocation.patch_id is None
    assert invocation.program_id == revision.revised_program_id
    assert revision.base_program_id != revision.revised_program_id
    assert revision.capability_id == "filter_by_sale_channel"


@pytest.mark.django_db
def test_django_persists_declared_program_revision(
    api_client,
    fervis_foundation_reset,
):
    from tests.testkit.answer_program_fixtures import load_answer_program_fixture

    user = _seeded_user()
    principal = QuestionPrincipal(
        principal_id=str(user.pk),
        tenant_id="tenant_1",
        raw=user,
        read_context_ref=ReadContextRef(
            scheme="django_principal",
            key=str(user.pk),
        ),
    )
    base = django_question_service().ask(
        AskRequest(
            question="How many sales did we make today?",
            principal=principal,
        )
    )
    fixture = load_answer_program_fixture(
        program="capability",
        binding_set="capability",
        catalog="sales_channel",
    )
    program, bindings = fixture.program, fixture.bindings
    recorder = DjangoLineageRecorder()
    recorder.record_program_invocation(
        program_invocation_bundle(
            program=program,
            invocation=program_invocation(
                run_id=base.run_id,
                program_id=answer_program_id(program),
                bindings=bindings,
                kind=ProgramInvocationKind.COMPILED_QUESTION,
            ),
        )
    )
    revision = apply_capability(
        program=program,
        bindings=bindings,
        application=CapabilityApplication(
            capability_id="filter_by_sale_channel",
            binding=ParameterBinding(
                parameter_id="semantic.sale_channels",
                value=FactValue.string_set(
                    id="capability.sale_channels",
                    values=("STORE",),
                ),
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.SEMANTIC_CHOICE,
                    refs=("governance:sales.channel",),
                ),
            ),
        ),
    )

    recorder.record_program_revision(
        program_revision_bundle(
            revision=revision,
        )
    )

    stored = ProgramRevision.objects.get(revision_id=revision.revision_id)
    assert stored.base_program_id == revision.base_program_id
    assert stored.revised_program_id == revision.revised_program_id
    assert stored.capability_id == "filter_by_sale_channel"


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
            read_context_ref=other,
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
            "primaryRunId": "run_newer_latest",
            "latestRunId": "run_newer_latest",
            "activeRunId": "run_newer_latest",
            "status": "QUEUED",
            "runCount": 1,
            "updatedAt": "2026-06-27T10:15:00+00:00",
        },
        {
            "conversationId": "conversation_older",
            "firstQuestion": "How many stores were open yesterday?",
            "latestQuestionId": "question_older",
            "primaryRunId": "run_older",
            "latestRunId": "run_older",
            "activeRunId": "run_older",
            "status": "QUEUED",
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
def test_django_question_projection_rejects_missing_work_state(
    api_client,
    fervis_foundation_reset,
):
    user = _seeded_user()
    owner = ReadContextRef(scheme="django_principal", key=str(user.pk))
    lifecycle = DjangoQuestionLifecyclePort()
    lifecycle.submit_question_run_atomically(
        submission=_submission(
            user=user,
            question_id="question_missing_work",
            run_id="run_missing_work",
        ),
        record=_question_run_record(
            question_id="question_missing_work",
            run_id="run_missing_work",
            read_context_ref=owner,
        ),
    )
    RunWorkItem.objects.filter(run_id="run_missing_work").delete()
    access = lifecycle.get_question(
        question_id="question_missing_work",
        authority=ReadAuthority(tenant_id="tenant_1", read_context_ref=owner),
    )
    assert access is not None

    with pytest.raises(
        RuntimeError,
        match="question run is missing its persisted work state",
    ):
        DjangoQuestionStateReaderPort().get_question_state(access=access)


@pytest.mark.django_db
def test_django_question_run_port_returns_idempotent_submission_without_lineage_duplication(
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
        need=ClarificationNeed.TARGET_REFERENCE.value,
        reason=ClarificationReason.MULTIPLE_MATCHING_ENTITIES.value,
        payload_json={
            "id": "clar_1",
            "need": "target_reference",
            "reason": "multiple_matching_entities",
            "requestedFactId": "question_contract",
            "question": "Which matching store should I use?",
            "subjects": [
                {
                    "kind": "question_input",
                    "id": "store",
                    "label": "store",
                    "sourceText": "",
                    "options": [],
                }
            ],
            "evidence": [],
        },
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
                kind=QuestionRunKind.MODEL_ASSISTED,
                trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
                adapter_ref="django_drf",
                runtime_version="test-runtime",
                base_run_id="run_1",
                trigger_clarification_response_id="clar_response_1",
            )
        ),
    )

    continued = QuestionRun.objects.get(run_id="run_2")
    assert result.kind is QuestionRunSubmissionKind.CREATED
    assert continued.question_id == run.question_id
    assert continued.base_run_id == "run_1"
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

    existing = port.find_idempotent_run(
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

    assert existing is not None
    assert existing.submission.run_id == "run_1"


@pytest.mark.django_db
def test_django_question_run_port_returns_same_run_id_idempotently(
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
            question=queued.submission.spec.question,
            read_context_ref=queued.submission.principal.read_context_ref,
            principal=queued.submission.principal.principal_id,
            provider=queued.submission.spec.provider,
            model_key=queued.submission.spec.model_key,
            conversation_context=queued.submission.spec.conversation_context,
            runtime_context=queued.submission.spec.runtime_context,
            max_budget_usd=queued.submission.spec.max_budget_usd,
            max_thinking_tokens=queued.submission.spec.max_thinking_tokens,
            active_attempt=work_item.active_attempt,
        )
    )

    assert runtime.calls[0]["read_context_ref"] == ReadContextRef(
        scheme="django_principal",
        key=str(user.pk),
    )
    assert runtime.calls[0]["read_context_ref"].key != worker_id


def _django_answered_program_base(
    *,
    program_name: str = "invocation",
    binding_set_name: str = "invocation",
    catalog_name: str = "sales",
):
    from tests.testkit.answer_program_fixtures import load_answer_program_fixture

    user = _seeded_user()
    principal = QuestionPrincipal(
        principal_id=str(user.pk),
        tenant_id="tenant_1",
        raw=user,
        read_context_ref=ReadContextRef(
            scheme="django_principal",
            key=str(user.pk),
        ),
    )
    service = django_question_service()
    base = service.ask(
        AskRequest(
            question="How many sales did we make today?",
            principal=principal,
            execution_mode=ExecutionMode.QUEUED,
        )
    )
    fixture = load_answer_program_fixture(
        program=program_name,
        binding_set=binding_set_name,
        catalog=catalog_name,
    )
    program, bindings = fixture.program, fixture.bindings
    recorder = DjangoLineageRecorder()
    recorder.record_program_invocation(
        program_invocation_bundle(
            program=program,
            invocation=program_invocation(
                run_id=base.run_id,
                program_id=answer_program_id(program),
                bindings=bindings,
                kind=ProgramInvocationKind.COMPILED_QUESTION,
            ),
        )
    )
    recorder.record_run_result(
        RunResultWrite(
            run_result_id="base_answered",
            run_id=base.run_id,
            result_kind=RunResultKind.ANSWERED,
        )
    )
    mark_work_item_terminal(run_id=base.run_id, status="COMPLETED")
    return service, principal, base


def _django_rerun_request(
    base,
    *,
    principal: QuestionPrincipal,
) -> RerunQuestionRequest:
    return RerunQuestionRequest(
        question_id=base.question_id,
        base_run_id=base.run_id,
        patch=BindingPatch(
            operations=(
                SetParameter(
                    parameter_id="semantic.sale_states",
                    value=FactValue.string_set(
                        id="patch.sale_states",
                        values=("COMPLETED", "PLACED"),
                    ),
                ),
            )
        ),
        principal=principal,
    )


def _django_capability_rerun_request(
    base,
    *,
    principal: QuestionPrincipal,
) -> RerunQuestionRequest:
    return RerunQuestionRequest(
        question_id=base.question_id,
        base_run_id=base.run_id,
        capability_application=CapabilityApplication(
            capability_id="filter_by_sale_channel",
            binding=ParameterBinding(
                parameter_id="semantic.sale_channels",
                value=FactValue.string_set(
                    id="capability.sale_channels",
                    values=("STORE",),
                ),
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.SEMANTIC_CHOICE,
                    refs=("governance:sales.channel",),
                ),
            ),
        ),
        principal=principal,
    )


def _submission(
    *,
    user=None,
    conversation_id: str = "conversation_1",
    question_id: str = "question_1",
    run_id: str = "run_1",
    question: str = "How many stores are open?",
    idempotency_key: str | None = None,
    execution_mode: ExecutionMode = ExecutionMode.QUEUED,
    read_context_ref: ReadContextRef | None = None,
) -> RunSubmission:
    user = user or _seeded_user()
    return RunSubmission(
        conversation_id=conversation_id,
        tenant_id="tenant_1",
        question_id=question_id,
        run_id=run_id,
        principal=QuestionPrincipal(
            principal_id=str(user.pk),
            tenant_id="tenant_1",
            raw=user,
            read_context_ref=ReadContextRef(
                scheme="django_principal", key=str(user.pk)
            ) if read_context_ref is None else read_context_ref,
        ),
        spec=ResolveQuestionRunSpec(
            question=question,
            provider="anthropic",
            model_key="HAIKU",
            conversation_context={"factArtifacts": []},
            runtime_context={"request_id": "request_1"},
            max_budget_usd="0.25",
            max_thinking_tokens=128,
        ),
        execution_mode=execution_mode,
        idempotency_key=idempotency_key,
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
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.INITIAL,
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
