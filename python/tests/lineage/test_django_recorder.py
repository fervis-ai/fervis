from __future__ import annotations

import pytest

from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.django.store import (
    DjangoLineageRecorderStore,
    lineage_model_by_record_key,
)
from fervis.lineage.views.django import DjangoLineageQuery
from fervis.lineage.enums import (
    AnswerValueKind,
    ArtifactKind,
    FactResultKind,
    MemoryArtifactSourceKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    PresentationKind,
    ProgramInvocationKind,
    QuestionRunKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lookup.clarification import ClarificationNeed, ClarificationReason
from fervis.lineage.models import (
    Answer,
    AnswerProgram,
    AnswerOutput,
    AnswerPresentation,
    ClarificationRequest,
    ClarificationResponse,
    Conversation,
    ExecutionProofGraph,
    FactResult,
    MemoryArtifact,
    ModelCall,
    ModelCallUsage,
    ProgramInvocation,
    RequestedFact,
    RunArtifact,
    RunResult,
    RunStep,
    RuntimeErrorDetail,
    SourceRead,
)
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.recorder import (
    AnswerOutputWrite,
    AnswerPresentationWrite,
    AnswerWrite,
    AnsweredRunResultWrite,
    AnswerProgramWrite,
    CatalogEndpointWrite,
    ClarificationRequestWrite,
    ClarificationResponseWrite,
    ConversationWrite,
    ExecutionProofGraphWrite,
    FactualTerminalRunResultWrite,
    FactResultWrite,
    LineageRecorderConflict,
    MemoryArtifactWrite,
    ModelCallAuditWrite,
    ModelCallUsageWrite,
    ModelCallWrite,
    ProgramInvocationBundleWrite,
    ProgramInvocationWrite,
    QuestionRunWrite,
    QuestionWrite,
    RequestedFactWrite,
    RunArtifactWrite,
    RunResultWrite,
    RunStepWrite,
    RuntimeErrorResultWrite,
    RuntimeErrorWrite,
    SourceReadWrite,
)
from fervis.lineage.records import RECORD_SPECS_BY_KEY
from fervis.lineage.recorder_core import LineageRecorder

pytestmark = pytest.mark.django_db

_CATALOG_ENDPOINT_ID = "11111111-1111-4111-8111-111111111111"


def test_django_lineage_store_maps_every_shared_lineage_record() -> None:
    assert set(lineage_model_by_record_key()) == set(RECORD_SPECS_BY_KEY)


def test_django_lineage_recorder_ensure_conversation_rejects_conflicting_root() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    recorder.ensure_conversation(
        ConversationWrite(conversation_id="cv_1", tenant_id="tenant_1")
    )

    with pytest.raises(LineageRecorderConflict, match="already exists"):
        recorder.ensure_conversation(
            ConversationWrite(conversation_id="cv_1", tenant_id="tenant_2")
        )

    conversation = Conversation.objects.get(conversation_id="cv_1")

    assert conversation.tenant_id == "tenant_1"


def test_django_lineage_recorder_ensure_conversation_is_idempotent() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    conversation = ConversationWrite(conversation_id="cv_1", tenant_id="tenant_1")

    recorder.ensure_conversation(conversation)
    recorder.ensure_conversation(conversation)

    assert Conversation.objects.filter(conversation_id="cv_1").count() == 1


def test_django_program_and_invocation_persistence_is_atomic(monkeypatch) -> None:
    spine_recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(spine_recorder)
    store = DjangoLineageRecorderStore()
    recorder = LineageRecorder(store)
    persist = store.get_or_insert_row

    def fail_on_invocation(row):
        if row.key == "program_invocation":
            raise RuntimeError("injected invocation persistence failure")
        return persist(row)

    monkeypatch.setattr(store, "get_or_insert_row", fail_on_invocation)

    with pytest.raises(RuntimeError, match="injected invocation persistence failure"):
        _record_program_invocation(recorder)

    assert AnswerProgram.objects.count() == 0
    assert ProgramInvocation.objects.count() == 0


def test_django_lineage_recorder_persists_model_call_usage_and_artifact() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_model",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_model_call(
        ModelCallWrite(
            model_call_id="call_1",
            run_id="run_1",
            step_id="step_model",
            call_index=1,
            provider="openai",
            model_key="test-model",
            status=ModelCallStatus.SUCCEEDED,
            prompt_chars=100,
            schema_chars=50,
            tool_spec_chars=75,
        )
    )
    recorder.record_model_call_usage(
        ModelCallUsageWrite(
            usage_id="usage_1",
            run_id="run_1",
            model_call_id="call_1",
            usage_kind=ModelUsageKind.INPUT_TOKENS,
            quantity=25,
            unit=ModelUsageUnit.TOKENS,
            provider_usage_key="input_tokens",
            cost_micros=12,
            currency="USD",
            price_basis_json={"unit": "token"},
        )
    )
    recorder.record_artifact(
        RunArtifactWrite(
            artifact_id="artifact_prompt",
            run_id="run_1",
            step_id="step_model",
            model_call_id="call_1",
            artifact_kind=ArtifactKind.PROMPT,
            content_hash="sha256:prompt",
            content="prompt text",
            content_type="text/plain",
            size_bytes=11,
        )
    )

    model_call = ModelCall.objects.get(model_call_id="call_1")
    usage = ModelCallUsage.objects.get(usage_id="usage_1")
    artifact = RunArtifact.objects.get(artifact_id="artifact_prompt")

    assert model_call.prompt_chars == 100
    assert model_call.tool_spec_chars == 75
    assert model_call.provider == "openai"
    assert model_call.status == ModelCallStatus.SUCCEEDED.value
    assert usage.quantity == 25
    assert usage.currency == "USD"
    assert usage.price_basis_json == {"unit": "token"}
    assert artifact.model_call_id == "call_1"
    assert artifact.content == "prompt text"


def test_django_lineage_recorder_rejects_conflicting_model_call_index() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_model",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_model_call(
        ModelCallWrite(
            model_call_id="call_1",
            run_id="run_1",
            step_id="step_model",
            call_index=1,
            provider="openai",
            model_key="test-model",
            status=ModelCallStatus.SUCCEEDED,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="already exists"):
        recorder.record_model_call(
            ModelCallWrite(
                model_call_id="call_2",
                run_id="run_1",
                step_id="step_model",
                call_index=1,
                provider="openai",
                model_key="test-model",
                status=ModelCallStatus.SUCCEEDED,
            )
        )


def test_django_lineage_recorder_records_model_call_audit_idempotently() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_model",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    audit = ModelCallAuditWrite(
        model_call=ModelCallWrite(
            model_call_id="call_1",
            run_id="run_1",
            step_id="step_model",
            call_index=1,
            provider="openai",
            model_key="test-model",
            status=ModelCallStatus.SUCCEEDED,
            prompt_chars=100,
            schema_chars=50,
            tool_spec_chars=75,
        ),
        usage_rows=(
            ModelCallUsageWrite(
                usage_id="usage_1",
                run_id="run_1",
                model_call_id="call_1",
                usage_kind=ModelUsageKind.INPUT_TOKENS,
                quantity=25,
                unit=ModelUsageUnit.TOKENS,
                provider_usage_key="inputTokens",
            ),
        ),
        artifacts=(
            RunArtifactWrite(
                artifact_id="artifact_prompt",
                run_id="run_1",
                step_id="step_model",
                model_call_id="call_1",
                artifact_kind=ArtifactKind.PROMPT,
                content_hash="sha256:prompt",
                content="prompt text",
                content_type="text/plain",
                size_bytes=11,
            ),
        ),
    )

    recorder.record_model_call_audit(audit)
    recorder.record_model_call_audit(audit)

    assert ModelCall.objects.count() == 1
    assert ModelCallUsage.objects.count() == 1
    assert RunArtifact.objects.count() == 1


def test_django_lineage_recorder_rejects_artifact_model_call_from_other_step() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    for step in (
        RunStepWrite(
            step_id="step_source_binding",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        ),
        RunStepWrite(
            step_id="step_fact_planning",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.FACT_PLANNING,
            kind=RunStepKind.MODEL_TURN,
        ),
    ):
        recorder.record_step(step)
    recorder.record_model_call(
        ModelCallWrite(
            model_call_id="call_1",
            run_id="run_1",
            step_id="step_source_binding",
            call_index=1,
            provider="openai",
            model_key="test-model",
            status=ModelCallStatus.SUCCEEDED,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="model call step must match"):
        recorder.record_artifact(
            RunArtifactWrite(
                artifact_id="artifact_wrong_step",
                run_id="run_1",
                step_id="step_fact_planning",
                model_call_id="call_1",
                artifact_kind=ArtifactKind.PROMPT,
                content_hash="sha256:prompt",
                content="prompt text",
                content_type="text/plain",
                size_bytes=11,
            )
        )


def test_django_lineage_recorder_rejects_model_call_for_deterministic_step() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="not a model_turn"):
        recorder.record_model_call(
            ModelCallWrite(
                model_call_id="call_1",
                run_id="run_1",
                step_id="step_execute",
                call_index=1,
                provider="openai",
                model_key="test-model",
                status=ModelCallStatus.SUCCEEDED,
            )
        )


def test_django_lineage_recorder_rejects_model_call_for_missing_step() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)

    with pytest.raises(LineageRecorderConflict, match="model call step does not exist"):
        recorder.record_model_call(
            ModelCallWrite(
                model_call_id="call_1",
                run_id="run_1",
                step_id="missing_step",
                call_index=1,
                provider="openai",
                model_key="test-model",
                status=ModelCallStatus.SUCCEEDED,
            )
        )


def test_django_lineage_recorder_records_steps_idempotently() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    step = RunStepWrite(
        step_id="step_model",
        run_id="run_1",
        sequence=1,
        step_key=RunStepKey.SOURCE_BINDING,
        kind=RunStepKind.MODEL_TURN,
        input_summary_json={"purpose": "source_binding"},
    )

    recorder.record_step(step)
    recorder.record_step(step)

    with pytest.raises(LineageRecorderConflict, match="different lineage fields"):
        recorder.record_step(
            RunStepWrite(
                step_id="step_model",
                run_id="run_1",
                sequence=1,
                step_key=RunStepKey.SOURCE_BINDING,
                kind=RunStepKind.MODEL_TURN,
                input_summary_json={"purpose": "changed"},
            )
        )


def test_django_lineage_recorder_persists_source_read_without_response_body() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    _record_catalog_endpoint(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    source_read_write = SourceReadWrite(
        source_read_id="source_read_1",
        run_id="run_1",
        step_id="step_execute",
        catalog_endpoint_id=_CATALOG_ENDPOINT_ID,
        status=SourceReadStatus.SUCCEEDED,
        row_count=2,
        completeness_json={"complete": True},
        response_hash="sha256:stores",
        args_json={"is_open": True},
    )
    recorder.record_source_read(source_read_write)
    recorder.record_source_read(source_read_write)

    source_read = SourceRead.objects.select_related("step").get(
        source_read_id="source_read_1"
    )

    assert source_read.artifact_id is None
    assert source_read.catalog_endpoint_id == _CATALOG_ENDPOINT_ID
    assert source_read.catalog_endpoint.source_namespace_path_json == ["retail_ops"]
    assert source_read.args_json == {"is_open": True}
    assert source_read.response_hash == "sha256:stores"
    assert source_read.step.step_key == RunStepKey.EXECUTE.value
    assert SourceRead.objects.filter(source_read_id="source_read_1").count() == 1


def test_django_lineage_recorder_rejects_source_read_endpoint_from_other_run() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    _record_empty_run(recorder, question_id="q_2", run_id="run_2", sequence=2)
    recorder.record_catalog_endpoint(
        CatalogEndpointWrite(
            catalog_endpoint_id="22222222-2222-4222-8222-222222222222",
            run_id="run_2",
            catalog_endpoint_key="django_retail_ops_list_store_list:test",
            endpoint_name="list_store_list",
            framework_kind="django_drf",
            source_namespace_kind="django_app",
            source_namespace_path_json=("retail_ops",),
            route_method="GET",
            route_path_template="/v1/stores/",
            handler_ref="apps.retail_ops.views.StoreListView",
        )
    )

    with pytest.raises(
        LineageRecorderConflict,
        match="source read catalog endpoint must belong to run 'run_1'",
    ):
        recorder.record_source_read(
            SourceReadWrite(
                source_read_id="source_read_1",
                run_id="run_1",
                step_id="step_execute",
                catalog_endpoint_id="22222222-2222-4222-8222-222222222222",
                status=SourceReadStatus.SUCCEEDED,
                row_count=2,
                completeness_json={"complete": True},
                response_hash="sha256:stores",
            )
        )


def test_django_lineage_recorder_records_execution_step_with_source_reads_idempotently() -> (
    None
):
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    _record_catalog_endpoint(recorder)
    step = RunStepWrite(
        step_id="step_execute",
        run_id="run_1",
        sequence=1,
        step_key=RunStepKey.EXECUTE,
        kind=RunStepKind.DETERMINISTIC,
        output_summary_json={"relationCount": 2},
    )
    source_reads = (
        SourceReadWrite(
            source_read_id="source_read_1",
            run_id="run_1",
            step_id="step_execute",
            catalog_endpoint_id=_CATALOG_ENDPOINT_ID,
            status=SourceReadStatus.SUCCEEDED,
            row_count=2,
            completeness_json={"complete": True},
            response_hash="sha256:stores",
            args_json={"is_open": True},
        ),
    )

    catalog_endpoints = (_catalog_endpoint_write(),)

    recorder.record_step_with_source_context(
        step,
        catalog_endpoints,
        source_reads,
    )
    recorder.record_step_with_source_context(
        step,
        catalog_endpoints,
        source_reads,
    )

    assert RunStep.objects.filter(step_id="step_execute").count() == 1
    assert SourceRead.objects.filter(source_read_id="source_read_1").count() == 1


def test_django_lineage_recorder_rejects_source_read_artifact_from_other_step() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    _record_catalog_endpoint(recorder)
    for step in (
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        ),
        RunStepWrite(
            step_id="step_render",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.RENDER,
            kind=RunStepKind.DETERMINISTIC,
        ),
    ):
        recorder.record_step(step)
    recorder.record_artifact(
        RunArtifactWrite(
            artifact_id="artifact_source_response",
            run_id="run_1",
            step_id="step_render",
            artifact_kind=ArtifactKind.SOURCE_RESPONSE,
            content_hash="sha256:source-response",
            content="[]",
            content_type="application/json",
            size_bytes=2,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="artifact step must match"):
        recorder.record_source_read(
            SourceReadWrite(
                source_read_id="source_read_1",
                run_id="run_1",
                step_id="step_execute",
                catalog_endpoint_id=_CATALOG_ENDPOINT_ID,
                status=SourceReadStatus.SUCCEEDED,
                row_count=2,
                completeness_json={"complete": True},
                response_hash="sha256:stores",
                artifact_id="artifact_source_response",
            )
        )


def test_django_lineage_recorder_persists_answer_lineage_primitives_idempotently() -> (
    None
):
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    _record_catalog_endpoint(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_contract",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_compile",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=3,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_render",
            run_id="run_1",
            sequence=4,
            step_key=RunStepKey.RENDER,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    _record_answered_source_read(recorder)
    requested_fact = RequestedFactWrite(
        requested_fact_id="fact_1",
        run_id="run_1",
        produced_by_step_id="step_contract",
        fact_key="fact_1",
        description="open store count",
        answer_expression_family="scalar_aggregate",
        requested_fact_json={"description": "open store count"},
        answer_requests_json={"outputs": ["answer_1"]},
    )
    fact_result = FactResultWrite(
        fact_result_id="fact_result_1",
        run_id="run_1",
        requested_fact_id="fact_1",
        produced_by_step_id="step_execute",
        result_kind=FactResultKind.ANSWERED,
        evidence_refs_json=["source_read:source_read_1"],
    )
    proof_graph = ExecutionProofGraphWrite(
        proof_graph_id="proof_1",
        run_id="run_1",
        fact_result_id="fact_result_1",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        payload_schema="fervis.execution_proof_graph",
        payload_schema_rev=1,
        payload_json=_answer_proof_graph_payload(),
    )
    run_result = RunResultWrite(
        run_result_id="result_1",
        run_id="run_1",
        result_kind=RunResultKind.ANSWERED,
    )
    answer = AnswerWrite(
        answer_id="answer_1",
        run_id="run_1",
        run_result_id="result_1",
    )
    output = AnswerOutputWrite(
        answer_output_id="answer_output_1",
        run_id="run_1",
        answer_id="answer_1",
        fact_result_id="fact_result_1",
        output_key="answer_1",
        value_kind=AnswerValueKind.NUMBER,
        value_json={"kind": "number", "value": 2},
        proof_node_refs_json=[_answer_proof_node_ref()],
    )
    presentation = AnswerPresentationWrite(
        presentation_id="presentation_1",
        run_id="run_1",
        answer_id="answer_1",
        presentation_kind=PresentationKind.TEXT,
        render_step_id="step_render",
        rendered_value="2",
    )

    recorder.record_run_result(run_result)
    for _ in range(2):
        recorder.record_requested_fact(requested_fact)
        recorder.record_fact_result(fact_result)
        recorder.record_execution_proof_graph(proof_graph)
        recorder.record_answer(answer)
        recorder.record_answer_output(output)
        recorder.record_answer_presentation(presentation)

    assert RequestedFact.objects.filter(requested_fact_id="fact_1").count() == 1
    assert FactResult.objects.get(fact_result_id="fact_result_1").result_kind == (
        FactResultKind.ANSWERED.value
    )
    assert ExecutionProofGraph.objects.get(proof_graph_id="proof_1").payload_json == (
        _answer_proof_graph_payload()
    )
    assert Answer.objects.get(answer_id="answer_1").run_result_id == "result_1"
    assert AnswerOutput.objects.get(answer_output_id="answer_output_1").value_json == {
        "kind": "number",
        "value": 2,
    }
    assert (
        AnswerPresentation.objects.get(presentation_id="presentation_1").rendered_value
        == "2"
    )


def test_django_lineage_recorder_records_answered_result_atomically() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    _record_program_invocation(recorder)
    for step in (
        RunStepWrite(
            step_id="step_contract",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        ),
        RunStepWrite(
            step_id="step_compile",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        ),
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=3,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        ),
    ):
        recorder.record_step(step)
    _record_answered_source_read(recorder)

    recorder.record_answered_result(
        AnsweredRunResultWrite(
            result=RunResultWrite(
                run_result_id="result_1",
                run_id="run_1",
                result_kind=RunResultKind.ANSWERED,
            ),
            requested_facts=(
                RequestedFactWrite(
                    requested_fact_id="fact_1",
                    run_id="run_1",
                    produced_by_step_id="step_contract",
                    fact_key="fact_1",
                    answer_expression_family="scalar_aggregate",
                ),
            ),
            fact_results=(
                FactResultWrite(
                    fact_result_id="fact_result_1",
                    run_id="run_1",
                    requested_fact_id="fact_1",
                    produced_by_step_id="step_execute",
                    result_kind=FactResultKind.ANSWERED,
                ),
            ),
            proof_graphs=(
                ExecutionProofGraphWrite(
                    proof_graph_id="proof_1",
                    run_id="run_1",
                    fact_result_id="fact_result_1",
                    compile_step_id="step_compile",
                    execute_step_id="step_execute",
                    payload_schema="fervis.execution_proof_graph",
                    payload_schema_rev=1,
                    payload_json=_answer_proof_graph_payload(),
                ),
            ),
            answer=AnswerWrite(
                answer_id="answer_1",
                run_id="run_1",
                run_result_id="result_1",
            ),
            outputs=(
                AnswerOutputWrite(
                    answer_output_id="answer_output_1",
                    run_id="run_1",
                    answer_id="answer_1",
                    fact_result_id="fact_result_1",
                    output_key="answer_1",
                    value_kind=AnswerValueKind.NUMBER,
                    value_json={"kind": "number", "value": 2},
                    proof_node_refs_json=[_answer_proof_node_ref()],
                ),
            ),
            memory_artifacts=(
                MemoryArtifactWrite(
                    memory_artifact_id="memory_artifact_1",
                    run_id="run_1",
                    produced_by_step_id="step_execute",
                    source_kind=MemoryArtifactSourceKind.FACT_RESULT,
                    fact_result_id="fact_result_1",
                    payload_schema="fervis.memory_artifact",
                    payload_schema_rev=1,
                    payload_json={
                        "sourceKind": "fact_result",
                        "artifactId": "memory_artifact_1",
                        "outcome": "answered",
                        "addresses": [
                            {
                                "address": "value.answer_1",
                                "kind": "value",
                                "value": {"type": "decimal", "value": "2"},
                            }
                        ],
                        "provenance": {"runId": "run_1"},
                    },
                ),
            ),
        )
    )

    assert Answer.objects.get(answer_id="answer_1").run_result_id == "result_1"
    assert AnswerOutput.objects.get(answer_output_id="answer_output_1").value_json == {
        "kind": "number",
        "value": 2,
    }
    assert MemoryArtifact.objects.get(
        memory_artifact_id="memory_artifact_1"
    ).payload_json["addresses"][0]["address"] == "value.answer_1"


def test_django_lineage_recorder_records_answered_result_idempotently() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisite_steps(recorder)
    answered_result = _answered_result_write()

    recorder.record_answered_result(answered_result)
    recorder.record_answered_result(answered_result)

    assert RunResult.objects.filter(run_result_id="result_1").count() == 1
    assert RequestedFact.objects.filter(requested_fact_id="fact_1").count() == 1
    assert FactResult.objects.filter(fact_result_id="fact_result_1").count() == 1
    assert Answer.objects.filter(answer_id="answer_1").count() == 1
    assert AnswerOutput.objects.filter(answer_output_id="answer_output_1").count() == 1


def test_django_lineage_recorder_rolls_back_answered_result_on_late_failure() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisite_steps(recorder)

    with pytest.raises(LineageRecorderConflict, match="missing from proof graph"):
        recorder.record_answered_result(
            AnsweredRunResultWrite(
                result=RunResultWrite(
                    run_result_id="result_1",
                    run_id="run_1",
                    result_kind=RunResultKind.ANSWERED,
                ),
                requested_facts=(
                    RequestedFactWrite(
                        requested_fact_id="fact_1",
                        run_id="run_1",
                        produced_by_step_id="step_contract",
                        fact_key="fact_1",
                        answer_expression_family="scalar_aggregate",
                    ),
                ),
                fact_results=(
                    FactResultWrite(
                        fact_result_id="fact_result_1",
                        run_id="run_1",
                        requested_fact_id="fact_1",
                        produced_by_step_id="step_execute",
                        result_kind=FactResultKind.ANSWERED,
                    ),
                ),
                proof_graphs=(
                    ExecutionProofGraphWrite(
                        proof_graph_id="proof_1",
                        run_id="run_1",
                        fact_result_id="fact_result_1",
                        compile_step_id="step_compile",
                        execute_step_id="step_execute",
                        payload_schema="fervis.execution_proof_graph",
                        payload_schema_rev=1,
                        payload_json=_answer_proof_graph_payload(),
                    ),
                ),
                answer=AnswerWrite(
                    answer_id="answer_1",
                    run_id="run_1",
                    run_result_id="result_1",
                ),
                outputs=(
                    AnswerOutputWrite(
                        answer_output_id="answer_output_1",
                        run_id="run_1",
                        answer_id="answer_1",
                        fact_result_id="fact_result_1",
                        output_key="answer_1",
                        value_kind=AnswerValueKind.NUMBER,
                        value_json={"kind": "number", "value": 2},
                        proof_node_refs_json=["missing_node"],
                    ),
                ),
            )
        )

    assert RunResult.objects.filter(run_result_id="result_1").count() == 0
    assert RequestedFact.objects.filter(requested_fact_id="fact_1").count() == 0
    assert FactResult.objects.filter(fact_result_id="fact_result_1").count() == 0
    assert ExecutionProofGraph.objects.filter(proof_graph_id="proof_1").count() == 0
    assert Answer.objects.filter(answer_id="answer_1").count() == 0


def test_django_lineage_recorder_preserves_no_data_terminal_proof() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisite_steps(recorder)

    recorder.record_factual_terminal_result(
        FactualTerminalRunResultWrite(
            result=RunResultWrite(
                run_result_id="result_1",
                run_id="run_1",
                result_kind=RunResultKind.FACTUAL_TERMINAL,
            ),
            requested_facts=(
                RequestedFactWrite(
                    requested_fact_id="fact_1",
                    run_id="run_1",
                    produced_by_step_id="step_contract",
                    fact_key="fact_1",
                    answer_expression_family="scalar_aggregate",
                ),
            ),
            fact_results=(
                FactResultWrite(
                    fact_result_id="fact_result_1",
                    run_id="run_1",
                    requested_fact_id="fact_1",
                    produced_by_step_id="step_execute",
                    result_kind=FactResultKind.NO_DATA,
                ),
            ),
            proof_graphs=(
                ExecutionProofGraphWrite(
                    proof_graph_id="proof_1",
                    run_id="run_1",
                    fact_result_id="fact_result_1",
                    compile_step_id="step_compile",
                    execute_step_id="step_execute",
                    payload_schema="fervis.execution_proof_graph",
                    payload_schema_rev=1,
                    payload_json=_answer_proof_graph_payload(),
                ),
            ),
        )
    )

    assert FactResult.objects.get(fact_result_id="fact_result_1").result_kind == (
        FactResultKind.NO_DATA.value
    )
    assert ExecutionProofGraph.objects.get(proof_graph_id="proof_1").fact_result_id == (
        "fact_result_1"
    )


def test_django_lineage_recorder_persists_clarification_primitives() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_contract",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    requested_fact = RequestedFactWrite(
        requested_fact_id="fact_1",
        run_id="run_1",
        produced_by_step_id="step_contract",
        fact_key="fact_1",
        answer_expression_family="scalar_value",
    )
    fact_result = FactResultWrite(
        fact_result_id="fact_result_1",
        run_id="run_1",
        requested_fact_id="fact_1",
        produced_by_step_id="step_contract",
        result_kind=FactResultKind.NEEDS_CLARIFICATION,
        evidence_refs_json=["known_input:input_1"],
    )
    clarification = ClarificationRequestWrite(
        clarification_id="clarification_1",
        run_id="run_1",
        fact_result_id="fact_result_1",
        payload_json={
            "id": "clarification_1",
            "need": "target_reference",
            "reason": "multiple_matching_entities",
            "requestedFactId": "fact_1",
            "question": "Which matching area should I use?",
            "subjects": [
                {
                    "kind": "question_input",
                    "id": "input_1",
                    "label": "area",
                    "sourceText": "London",
                    "options": [{"id": "area:1", "label": "London"}],
                }
            ],
            "evidence": [{"kind": "known_input", "id": "known_input:input_1"}],
        },
    )
    response = ClarificationResponseWrite(
        response_id="response_1",
        run_id="run_1",
        clarification_id="clarification_1",
        selected_option_id="area:1",
        response_text="London",
        evidence_ref="clarification_response:response_1",
    )

    recorder.record_requested_fact(requested_fact)
    recorder.record_fact_result(fact_result)
    recorder.record_clarification_request(clarification)
    recorder.record_clarification_response(response)

    saved_clarification = ClarificationRequest.objects.get(
        clarification_id="clarification_1"
    )
    saved_response = ClarificationResponse.objects.get(response_id="response_1")

    assert saved_clarification.fact_result_id == "fact_result_1"
    assert saved_clarification.need == ClarificationNeed.TARGET_REFERENCE
    assert saved_clarification.reason == ClarificationReason.MULTIPLE_MATCHING_ENTITIES
    assert saved_clarification.payload_json["question"] == (
        "Which matching area should I use?"
    )
    assert saved_response.clarification_id == "clarification_1"
    assert saved_response.selected_option_id == "area:1"


def test_django_lineage_recorder_rejects_cross_run_lineage_references() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.ensure_conversation(
        ConversationWrite(conversation_id="cv_2", tenant_id="tenant_1")
    )
    recorder.record_question(
        QuestionWrite(
            question_id="q_2",
            conversation_id="cv_2",
            conversation_sequence=1,
            original_question="How many sales are open?",
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id="run_2",
            question_id="q_2",
            run_number=1,
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.INITIAL,
            adapter_ref="django_drf:test",
            runtime_version="test-runtime",
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_other_run",
            run_id="run_2",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="must belong to run 'run_1'"):
        recorder.record_requested_fact(
            RequestedFactWrite(
                requested_fact_id="fact_1",
                run_id="run_1",
                produced_by_step_id="step_other_run",
                fact_key="fact_1",
                answer_expression_family="scalar_aggregate",
            )
        )


def test_django_lineage_recorder_rejects_cross_run_memory_artifact_refs() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisites(recorder)
    recorder.ensure_conversation(
        ConversationWrite(conversation_id="cv_2", tenant_id="tenant_1")
    )
    recorder.record_question(
        QuestionWrite(
            question_id="q_2",
            conversation_id="cv_2",
            conversation_sequence=1,
            original_question="How many sales are open?",
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id="run_2",
            question_id="q_2",
            run_number=1,
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.INITIAL,
            adapter_ref="django_drf:test",
            runtime_version="test-runtime",
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_other_run",
            run_id="run_2",
            sequence=1,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_requested_fact(
        RequestedFactWrite(
            requested_fact_id="fact_2",
            run_id="run_2",
            produced_by_step_id="step_other_run",
            fact_key="fact_1",
            answer_expression_family="scalar_aggregate",
        )
    )
    recorder.record_fact_result(
        FactResultWrite(
            fact_result_id="fact_result_2",
            run_id="run_2",
            requested_fact_id="fact_2",
            produced_by_step_id="step_other_run",
            result_kind=FactResultKind.ANSWERED,
        )
    )

    with pytest.raises(
        LineageRecorderConflict,
        match="memory artifact fact result must belong to run 'run_1'",
    ):
        recorder.record_memory_artifact(
            MemoryArtifactWrite(
                memory_artifact_id="memory_artifact_cross_run",
                run_id="run_1",
                produced_by_step_id="step_execute",
                source_kind=MemoryArtifactSourceKind.FACT_RESULT,
                fact_result_id="fact_result_2",
                payload_schema="fervis.memory_artifact",
                payload_schema_rev=1,
                payload_json={
                    "sourceKind": "fact_result",
                    "artifactId": "memory_artifact_cross_run",
                    "outcome": "answered",
                    "addresses": [
                        {
                            "address": "value.answer_1",
                            "kind": "value",
                            "value": {"type": "number", "value": "1"},
                        }
                    ],
                },
            )
        )


def test_django_lineage_query_maps_memory_artifact_rows() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisites(recorder)
    recorder.record_memory_artifact(
        MemoryArtifactWrite(
            memory_artifact_id="memory_artifact_1",
            run_id="run_1",
            produced_by_step_id="step_execute",
            source_kind=MemoryArtifactSourceKind.FACT_RESULT,
            fact_result_id="fact_result_1",
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "fact_result",
                "artifactId": "memory_artifact_1",
                "outcome": "answered",
                "addresses": [
                    {
                        "address": "value.answer_1",
                        "kind": "value",
                        "value": {"type": "number", "value": "1"},
                    }
                ],
            },
        )
    )

    rows = DjangoLineageQuery().lineage_rows_for_run_ids(("run_1",))

    assert len(rows.memory_artifacts) == 1
    memory_artifact = rows.memory_artifacts[0]
    assert memory_artifact.memory_artifact_id == "memory_artifact_1"
    assert memory_artifact.source_kind is MemoryArtifactSourceKind.FACT_RESULT
    assert memory_artifact.fact_result_id == "fact_result_1"
    assert memory_artifact.requested_fact_id is None
    assert memory_artifact.payload_json["artifactId"] == "memory_artifact_1"


def test_django_lineage_query_loads_memory_for_selected_runs() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisites(recorder)
    recorder.record_memory_artifact(
        MemoryArtifactWrite(
            memory_artifact_id="memory_artifact_1",
            run_id="run_1",
            produced_by_step_id="step_execute",
            source_kind=MemoryArtifactSourceKind.FACT_RESULT,
            fact_result_id="fact_result_1",
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "fact_result",
                "artifactId": "memory_artifact_1",
                "outcome": "answered",
                "addresses": [
                    {
                        "address": "value.answer_1",
                        "kind": "value",
                        "value": {"type": "number", "value": "1"},
                    }
                ],
            },
        )
    )
    _record_empty_run(recorder, question_id="q_2", run_id="run_2", sequence=2)
    _record_empty_run(recorder, question_id="q_3", run_id="run_3", sequence=3)

    rows = DjangoLineageQuery().memory_artifact_rows_for_run_ids(
        ("run_1",),
    )

    assert [row.memory_artifact_id for row in rows] == ["memory_artifact_1"]


def test_django_lineage_recorder_rejects_missing_clarification_trigger_response() -> (
    None
):
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)

    with pytest.raises(
        LineageRecorderConflict,
        match="clarification trigger response does not exist",
    ):
        recorder.start_run(
            QuestionRunWrite(
                run_id="run_2",
                question_id="q_1",
                run_number=2,
                kind=QuestionRunKind.MODEL_ASSISTED,
                trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
                base_run_id="run_1",
                trigger_clarification_response_id="response_missing",
                adapter_ref="django_drf:test",
                runtime_version="test-runtime",
            )
        )


def test_django_lineage_recorder_rejects_answer_for_non_answered_result() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_run_result(
        RunResultWrite(
            run_result_id="result_1",
            run_id="run_1",
            result_kind=RunResultKind.RUNTIME_ERROR,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="answer run result must be"):
        recorder.record_answer(
            AnswerWrite(
                answer_id="answer_1",
                run_id="run_1",
                run_result_id="result_1",
            )
        )


def test_lineage_write_contract_rejects_unversioned_payloads() -> None:
    with pytest.raises(ValueError, match="payload_schema"):
        FactResultWrite(
            fact_result_id="fact_result_1",
            run_id="run_1",
            requested_fact_id="fact_1",
            produced_by_step_id="step_execute",
            result_kind=FactResultKind.ANSWERED,
            payload_json={"value": 2},
        )


def test_lineage_write_contract_rejects_unversioned_answer_presentation_payload() -> (
    None
):
    with pytest.raises(ValueError, match="payload_schema"):
        AnswerPresentationWrite(
            presentation_id="presentation_1",
            run_id="run_1",
            answer_id="answer_1",
            presentation_kind=PresentationKind.TABLE,
            render_step_id="step_render",
            payload_json={"rows": []},
        )


def test_django_lineage_recorder_rejects_answer_output_without_proof_graph() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_contract",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_run_result(
        RunResultWrite(
            run_result_id="result_1",
            run_id="run_1",
            result_kind=RunResultKind.ANSWERED,
        )
    )
    recorder.record_requested_fact(
        RequestedFactWrite(
            requested_fact_id="fact_1",
            run_id="run_1",
            produced_by_step_id="step_contract",
            fact_key="fact_1",
            answer_expression_family="scalar_aggregate",
        )
    )
    recorder.record_fact_result(
        FactResultWrite(
            fact_result_id="fact_result_1",
            run_id="run_1",
            requested_fact_id="fact_1",
            produced_by_step_id="step_execute",
            result_kind=FactResultKind.ANSWERED,
        )
    )
    recorder.record_answer(
        AnswerWrite(answer_id="answer_1", run_id="run_1", run_result_id="result_1")
    )

    with pytest.raises(LineageRecorderConflict, match="proof graph does not exist"):
        recorder.record_answer_output(
            AnswerOutputWrite(
                answer_output_id="answer_output_1",
                run_id="run_1",
                answer_id="answer_1",
                fact_result_id="fact_result_1",
                output_key="answer_1",
                value_kind=AnswerValueKind.NUMBER,
                value_json={"kind": "number", "value": 2},
                proof_node_refs_json=[_answer_proof_node_ref()],
            )
        )


def test_django_lineage_recorder_rejects_unknown_answer_output_proof_ref() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisites(recorder)

    with pytest.raises(LineageRecorderConflict, match="missing from proof graph"):
        recorder.record_answer_output(
            AnswerOutputWrite(
                answer_output_id="answer_output_1",
                run_id="run_1",
                answer_id="answer_1",
                fact_result_id="fact_result_1",
                output_key="answer_1",
                value_kind=AnswerValueKind.NUMBER,
                value_json={"kind": "number", "value": 2},
                proof_node_refs_json=["missing_node"],
            )
        )


def test_django_lineage_recorder_rejects_missing_source_read_proof_ref() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    for step in (
        RunStepWrite(
            step_id="step_contract",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        ),
        RunStepWrite(
            step_id="step_compile",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        ),
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=3,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        ),
    ):
        recorder.record_step(step)
    recorder.record_requested_fact(
        RequestedFactWrite(
            requested_fact_id="fact_1",
            run_id="run_1",
            produced_by_step_id="step_contract",
            fact_key="fact_1",
            answer_expression_family="scalar_aggregate",
        )
    )
    recorder.record_fact_result(
        FactResultWrite(
            fact_result_id="fact_result_1",
            run_id="run_1",
            requested_fact_id="fact_1",
            produced_by_step_id="step_execute",
            result_kind=FactResultKind.ANSWERED,
        )
    )

    with pytest.raises(LineageRecorderConflict, match="proof graph source read"):
        recorder.record_execution_proof_graph(
            ExecutionProofGraphWrite(
                proof_graph_id="proof_1",
                run_id="run_1",
                fact_result_id="fact_result_1",
                compile_step_id="step_compile",
                execute_step_id="step_execute",
                payload_schema="fervis.execution_proof_graph",
                payload_schema_rev=1,
                payload_json=_answer_proof_graph_payload(),
            )
        )


def test_django_lineage_recorder_rejects_presentation_from_non_render_step() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_answered_lineage_prerequisites(recorder)

    with pytest.raises(LineageRecorderConflict, match="must be a render"):
        recorder.record_answer_presentation(
            AnswerPresentationWrite(
                presentation_id="presentation_1",
                run_id="run_1",
                answer_id="answer_1",
                presentation_kind=PresentationKind.TEXT,
                render_step_id="step_compile",
                rendered_value="2",
            )
        )


def test_django_lineage_recorder_persists_runtime_error_terminal_atomically() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_model",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_runtime_error_result(
        RuntimeErrorResultWrite(
            result=RunResultWrite(
                run_result_id="result_1",
                run_id="run_1",
                result_kind=RunResultKind.RUNTIME_ERROR,
            ),
            error=RuntimeErrorWrite(
                runtime_error_detail_id="error_1",
                run_id="run_1",
                run_result_id="result_1",
                failed_step_id="step_model",
                error_kind=RuntimeErrorKind.PROVIDER_RUNTIME_FAILED,
                message="provider timed out",
            ),
        )
    )

    error = RuntimeErrorDetail.objects.select_related("run_result").get(
        runtime_error_detail_id="error_1"
    )

    assert error.run_result.result_kind == RunResultKind.RUNTIME_ERROR.value
    assert error.failed_step_id == "step_model"
    assert error.message == "provider timed out"


def test_django_lineage_recorder_rolls_back_runtime_error_terminal_on_late_failure() -> (
    None
):
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)

    with pytest.raises(LineageRecorderConflict, match="runtime error step"):
        recorder.record_runtime_error_result(
            RuntimeErrorResultWrite(
                result=RunResultWrite(
                    run_result_id="result_1",
                    run_id="run_1",
                    result_kind=RunResultKind.RUNTIME_ERROR,
                ),
                error=RuntimeErrorWrite(
                    runtime_error_detail_id="error_1",
                    run_id="run_1",
                    run_result_id="result_1",
                    failed_step_id="missing_step",
                    error_kind=RuntimeErrorKind.PROVIDER_RUNTIME_FAILED,
                    message="provider timed out",
                ),
            )
        )

    assert RunResult.objects.filter(run_result_id="result_1").count() == 0
    assert (
        RuntimeErrorDetail.objects.filter(runtime_error_detail_id="error_1").count()
        == 0
    )


def _record_run_spine(recorder: LineageRecorderPort) -> None:
    recorder.ensure_conversation(
        ConversationWrite(conversation_id="cv_1", tenant_id="tenant_1")
    )
    recorder.record_question(
        QuestionWrite(
            question_id="q_1",
            conversation_id="cv_1",
            conversation_sequence=1,
            original_question="How many stores are open?",
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id="run_1",
            question_id="q_1",
            run_number=1,
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.INITIAL,
            adapter_ref="django_drf:test",
            runtime_version="test-runtime",
        )
    )


def _record_empty_run(
    recorder: LineageRecorderPort,
    *,
    question_id: str,
    run_id: str,
    sequence: int,
) -> None:
    recorder.record_question(
        QuestionWrite(
            question_id=question_id,
            conversation_id="cv_1",
            conversation_sequence=sequence,
            original_question="Follow-up?",
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id=run_id,
            question_id=question_id,
            run_number=1,
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.INITIAL,
            adapter_ref="django_drf:test",
            runtime_version="test-runtime",
        )
    )


def _record_answered_lineage_prerequisites(recorder: LineageRecorderPort) -> None:
    _record_answered_lineage_prerequisite_steps(recorder)
    recorder.record_run_result(
        RunResultWrite(
            run_result_id="result_1",
            run_id="run_1",
            result_kind=RunResultKind.ANSWERED,
        )
    )
    recorder.record_requested_fact(
        RequestedFactWrite(
            requested_fact_id="fact_1",
            run_id="run_1",
            produced_by_step_id="step_contract",
            fact_key="fact_1",
            answer_expression_family="scalar_aggregate",
        )
    )
    recorder.record_fact_result(
        FactResultWrite(
            fact_result_id="fact_result_1",
            run_id="run_1",
            requested_fact_id="fact_1",
            produced_by_step_id="step_execute",
            result_kind=FactResultKind.ANSWERED,
        )
    )
    recorder.record_execution_proof_graph(
        ExecutionProofGraphWrite(
            proof_graph_id="proof_1",
            run_id="run_1",
            fact_result_id="fact_result_1",
            compile_step_id="step_compile",
            execute_step_id="step_execute",
            payload_schema="fervis.execution_proof_graph",
            payload_schema_rev=1,
            payload_json=_answer_proof_graph_payload(),
        )
    )
    recorder.record_answer(
        AnswerWrite(answer_id="answer_1", run_id="run_1", run_result_id="result_1")
    )


def _record_answered_lineage_prerequisite_steps(
    recorder: LineageRecorderPort,
) -> None:
    _record_run_spine(recorder)
    _record_program_invocation(recorder)
    for step in (
        RunStepWrite(
            step_id="step_contract",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        ),
        RunStepWrite(
            step_id="step_compile",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        ),
        RunStepWrite(
            step_id="step_execute",
            run_id="run_1",
            sequence=3,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        ),
    ):
        recorder.record_step(step)
    _record_answered_source_read(recorder)


def _record_program_invocation(recorder: LineageRecorderPort) -> None:
    recorder.record_program_invocation(
        ProgramInvocationBundleWrite(
            program=AnswerProgramWrite(
                program_id="ap_test",
                schema_revision=1,
                canonical_json="{}",
            ),
            invocation=ProgramInvocationWrite(
                invocation_id="pi_test",
                run_id="run_1",
                program_id="ap_test",
                bindings_json="{}",
                kind=ProgramInvocationKind.COMPILED_QUESTION,
            ),
        )
    )


def test_content_addressed_program_is_shared_by_independent_run_invocations() -> None:
    recorder: LineageRecorderPort = DjangoLineageRecorder()
    _record_run_spine(recorder)
    _record_empty_run(recorder, question_id="q_2", run_id="run_2", sequence=2)
    program = AnswerProgramWrite(
        program_id="ap_shared",
        schema_revision=1,
        canonical_json='{"program":"shared"}',
    )
    for run_id in ("run_1", "run_2"):
        recorder.record_program_invocation(
            ProgramInvocationBundleWrite(
                program=program,
                invocation=ProgramInvocationWrite(
                    invocation_id=f"pi_{run_id}",
                    run_id=run_id,
                    program_id=program.program_id,
                    bindings_json="{}",
                    kind=ProgramInvocationKind.COMPILED_QUESTION,
                ),
            )
        )

    assert AnswerProgram.objects.filter(program_id="ap_shared").count() == 1
    assert ProgramInvocation.objects.filter(program_id="ap_shared").count() == 2


def _record_answered_source_read(recorder: LineageRecorderPort) -> None:
    _record_catalog_endpoint(recorder)
    recorder.record_source_read(
        SourceReadWrite(
            source_read_id="source_read_1",
            run_id="run_1",
            step_id="step_execute",
            catalog_endpoint_id=_CATALOG_ENDPOINT_ID,
            status=SourceReadStatus.SUCCEEDED,
            row_count=2,
            completeness_json={"complete": True},
            response_hash="sha256:stores",
        )
    )


def _record_catalog_endpoint(recorder: LineageRecorderPort) -> None:
    recorder.record_catalog_endpoint(_catalog_endpoint_write())


def _catalog_endpoint_write() -> CatalogEndpointWrite:
    return CatalogEndpointWrite(
        catalog_endpoint_id=_CATALOG_ENDPOINT_ID,
        run_id="run_1",
        catalog_endpoint_key="django_retail_ops_list_store_list:test",
        endpoint_name="list_store_list",
        framework_kind="django_drf",
        source_namespace_kind="django_app",
        source_namespace_path_json=("retail_ops",),
        route_method="GET",
        route_path_template="/v1/stores/",
        handler_ref="apps.retail_ops.views.StoreListView",
        domain_resource_names_json=("store",),
    )


def _answered_result_write() -> AnsweredRunResultWrite:
    return AnsweredRunResultWrite(
        result=RunResultWrite(
            run_result_id="result_1",
            run_id="run_1",
            result_kind=RunResultKind.ANSWERED,
        ),
        requested_facts=(
            RequestedFactWrite(
                requested_fact_id="fact_1",
                run_id="run_1",
                produced_by_step_id="step_contract",
                fact_key="fact_1",
                answer_expression_family="scalar_aggregate",
            ),
        ),
        fact_results=(
            FactResultWrite(
                fact_result_id="fact_result_1",
                run_id="run_1",
                requested_fact_id="fact_1",
                produced_by_step_id="step_execute",
                result_kind=FactResultKind.ANSWERED,
            ),
        ),
        proof_graphs=(
            ExecutionProofGraphWrite(
                proof_graph_id="proof_1",
                run_id="run_1",
                fact_result_id="fact_result_1",
                compile_step_id="step_compile",
                execute_step_id="step_execute",
                payload_schema="fervis.execution_proof_graph",
                payload_schema_rev=1,
                payload_json=_answer_proof_graph_payload(),
            ),
        ),
        answer=AnswerWrite(
            answer_id="answer_1",
            run_id="run_1",
            run_result_id="result_1",
        ),
        outputs=(
            AnswerOutputWrite(
                answer_output_id="answer_output_1",
                run_id="run_1",
                answer_id="answer_1",
                fact_result_id="fact_result_1",
                output_key="answer_1",
                value_kind=AnswerValueKind.NUMBER,
                value_json={"kind": "number", "value": 2},
                proof_node_refs_json=[_answer_proof_node_ref()],
            ),
        ),
    )


def _answer_proof_node_ref() -> str:
    return "answer_output:fact_1:answer_1"


def _answer_proof_graph_payload() -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": "relation:source_1",
                "kind": "relation",
                "proof_refs": ["source_read:source_read_1"],
            },
            {
                "id": _answer_proof_node_ref(),
                "kind": "answer_output",
                "proof_refs": [],
            },
        ],
        "edges": [
            {
                "source": "relation:source_1",
                "target": _answer_proof_node_ref(),
                "role": "produces",
            },
        ],
    }
