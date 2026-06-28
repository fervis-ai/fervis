from __future__ import annotations

import pytest
from django.db import IntegrityError, transaction

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    SourceReadStatus,
)
from fervis.lineage.models import (
    CatalogEndpoint,
    Conversation,
    ModelCall,
    ModelCallUsage,
    Question,
    QuestionRun,
    RunArtifact,
    RunStep,
    SourceRead,
)

pytestmark = pytest.mark.django_db


def test_lineage_spine_persists_source_read_without_response_body() -> None:
    conversation = Conversation.objects.create(
        conversation_id="cv_1",
        tenant_id="tenant_1",
    )
    question = Question.objects.create(
        question_id="q_1",
        conversation=conversation,
        conversation_sequence=1,
        original_question="How many stores are open?",
    )
    run = QuestionRun.objects.create(
        run_id="run_1",
        question=question,
        run_number=1,
        trigger_kind=RunTriggerKind.INITIAL.value,
        integrated_question="How many stores are open?",
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )
    step = RunStep.objects.create(
        step_id="step_execute",
        run=run,
        sequence=1,
        step_key=RunStepKey.EXECUTE.value,
        kind=RunStepKind.DETERMINISTIC.value,
    )
    catalog_endpoint = CatalogEndpoint.objects.create(
        catalog_endpoint_id="11111111-1111-4111-8111-111111111111",
        run=run,
        catalog_endpoint_key="django_retail_ops_list_store_list:test",
        endpoint_name="list_store_list",
        framework_kind="django_drf",
        source_namespace_kind="django_app",
        source_namespace_path_json=["retail_ops"],
        route_method="GET",
        route_path_template="/v1/stores/",
        handler_ref="apps.retail_ops.views.StoreListView",
    )
    source_read = SourceRead.objects.create(
        source_read_id="source_read_1",
        run=run,
        step=step,
        catalog_endpoint=catalog_endpoint,
        args_json={"is_open": True},
        status=SourceReadStatus.SUCCEEDED.value,
        row_count=2,
        completeness_json={"complete": True},
        response_hash="sha256:stores",
    )

    loaded = SourceRead.objects.select_related(
        "catalog_endpoint", "run__question__conversation", "step"
    ).get(source_read_id=source_read.source_read_id)

    assert loaded.artifact_id is None
    assert loaded.catalog_endpoint_id == "11111111-1111-4111-8111-111111111111"
    assert loaded.catalog_endpoint.catalog_endpoint_key == "django_retail_ops_list_store_list:test"
    assert loaded.catalog_endpoint.source_namespace_path_json == ["retail_ops"]
    assert loaded.args_json == {"is_open": True}
    assert loaded.row_count == 2
    assert loaded.run.question.conversation.tenant_id == "tenant_1"
    assert loaded.step.step_key == RunStepKey.EXECUTE.value


def test_lineage_persistence_rejects_duplicate_step_sequence_per_run() -> None:
    run = _create_run()
    RunStep.objects.create(
        step_id="step_1",
        run=run,
        sequence=1,
        step_key=RunStepKey.GROUNDING.value,
        kind=RunStepKind.MODEL_TURN.value,
    )

    with pytest.raises(IntegrityError), transaction.atomic():
        RunStep.objects.create(
            step_id="step_2",
            run=run,
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING.value,
            kind=RunStepKind.MODEL_TURN.value,
        )


def test_model_usage_and_artifacts_are_scoped_to_a_model_call() -> None:
    run = _create_run()
    step = RunStep.objects.create(
        step_id="step_model",
        run=run,
        sequence=1,
        step_key=RunStepKey.SOURCE_BINDING.value,
        kind=RunStepKind.MODEL_TURN.value,
    )
    model_call = ModelCall.objects.create(
        model_call_id="call_1",
        run=run,
        step=step,
        call_index=1,
        provider="openai",
        model_key="test-model",
        status=ModelCallStatus.SUCCEEDED.value,
        prompt_chars=120,
        schema_chars=80,
    )
    ModelCallUsage.objects.create(
        usage_id="usage_1",
        run=run,
        model_call=model_call,
        usage_kind=ModelUsageKind.INPUT_TOKENS.value,
        quantity=30,
        unit=ModelUsageUnit.TOKENS.value,
        provider_usage_key="input_tokens",
        cost_micros=12,
        currency="USD",
    )
    RunArtifact.objects.create(
        artifact_id="artifact_prompt",
        run=run,
        step=step,
        model_call=model_call,
        artifact_kind=ArtifactKind.PROMPT.value,
        content_hash="sha256:prompt",
        content="prompt text",
        content_type="text/plain",
        size_bytes=11,
    )

    loaded = ModelCall.objects.prefetch_related("usage_rows", "artifacts").get(
        model_call_id=model_call.model_call_id
    )

    assert [item.quantity for item in loaded.usage_rows.all()] == [30]
    assert [item.artifact_kind for item in loaded.artifacts.all()] == [
        ArtifactKind.PROMPT.value
    ]


def test_lineage_persistence_accepts_clarification_triggered_run() -> None:
    previous_run = _create_run(run_id="run_test_1", conversation_id="cv_test_1")
    clarification_run = QuestionRun.objects.create(
        run_id="run_test_2",
        question=previous_run.question,
        run_number=2,
        trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE.value,
        trigger_clarification_response_run=previous_run,
        trigger_clarification_response_id="response_1",
        integrated_question="Question with clarification.",
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )

    assert clarification_run.trigger_clarification_response_run_id == "run_test_1"
    assert clarification_run.trigger_clarification_response_id == "response_1"


def test_lineage_persistence_rejects_blank_external_artifact_reference() -> None:
    run = _create_run()
    step = RunStep.objects.create(
        step_id="step_execute",
        run=run,
        sequence=1,
        step_key=RunStepKey.EXECUTE.value,
        kind=RunStepKind.DETERMINISTIC.value,
    )

    with pytest.raises(IntegrityError), transaction.atomic():
        RunArtifact.objects.create(
            artifact_id="artifact_blank_ref",
            run=run,
            step=step,
            artifact_kind=ArtifactKind.DETERMINISTIC_OUTPUT.value,
            content_hash="sha256:empty-ref",
            storage_ref="",
            content_type="application/json",
            size_bytes=1,
        )


def _create_run(
    *, run_id: str = "run_test", conversation_id: str = "cv_test"
) -> QuestionRun:
    conversation = Conversation.objects.create(
        conversation_id=conversation_id,
        tenant_id="tenant_test",
    )
    question = Question.objects.create(
        question_id=f"q_{run_id}",
        conversation=conversation,
        conversation_sequence=1,
        original_question="Question?",
    )
    return QuestionRun.objects.create(
        run_id=run_id,
        question=question,
        run_number=1,
        trigger_kind=RunTriggerKind.INITIAL.value,
        integrated_question="Question?",
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )
