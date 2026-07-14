from __future__ import annotations

import pytest

from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.host_api.contracts import FrameworkKind, SourceNamespaceKind
from fervis.lineage.enums import (
    AnswerValueKind,
    ArtifactKind,
    FactResultKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    ProgramInvocationKind,
    QuestionRunKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    SourceReadStatus,
)
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.recorder import (
    AnswerOutputWrite,
    AnswerWrite,
    AnsweredRunResultWrite,
    AnswerProgramWrite,
    ClarificationRequestWrite,
    ClarificationResponseWrite,
    ConversationWrite,
    CatalogEndpointWrite,
    ExecutionProofGraphWrite,
    FactResultWrite,
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
    SourceReadWrite,
)
from fervis.observability.usage import RuntimeUsageService
from fervis.observability.django import DjangoObservabilityQuery

pytestmark = pytest.mark.django_db


def test_django_observability_query_maps_model_call_rows() -> None:
    recorder = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_source_binding",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_model_call_audit(
        ModelCallAuditWrite(
            model_call=ModelCallWrite(
                model_call_id="call_1",
                run_id="run_1",
                step_id="step_source_binding",
                call_index=1,
                provider="openai",
                model_key="gpt-test",
                status=ModelCallStatus.SUCCEEDED,
                reasoning_effort="medium",
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
                    quantity=20,
                    unit=ModelUsageUnit.TOKENS,
                    provider_usage_key="input_tokens",
                    cost_micros=1000,
                    currency="USD",
                ),
            ),
            artifacts=(
                RunArtifactWrite(
                    artifact_id="artifact_prompt",
                    run_id="run_1",
                    step_id="step_source_binding",
                    model_call_id="call_1",
                    artifact_kind=ArtifactKind.PROMPT,
                    content_hash="sha256:prompt",
                    content="prompt text",
                    content_type="text/plain",
                    size_bytes=11,
                ),
            ),
        )
    )

    rows = DjangoObservabilityQuery().model_calls_for_run("run_1")

    assert len(rows) == 1
    row = rows[0]
    assert row.model_call_id == "call_1"
    assert row.step_key == RunStepKey.SOURCE_BINDING
    assert row.step_sequence == 2
    assert row.status == ModelCallStatus.SUCCEEDED
    assert row.reasoning_effort == "medium"
    assert row.usage_rows[0].usage_kind == ModelUsageKind.INPUT_TOKENS
    assert row.usage_rows[0].cost_micros == 1000
    assert row.artifacts[0].artifact_kind == ArtifactKind.PROMPT
    assert row.artifacts[0].has_content is True
    assert not hasattr(row.artifacts[0], "content")


def test_django_observability_query_answer_scope_includes_previous_runs() -> None:
    recorder = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_run_1_source_binding",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_model_call_audit(
        ModelCallAuditWrite(
            model_call=ModelCallWrite(
                model_call_id="call_run_1",
                run_id="run_1",
                step_id="step_run_1_source_binding",
                call_index=1,
                provider="openai",
                model_key="gpt-test",
                status=ModelCallStatus.SUCCEEDED,
            ),
            usage_rows=(
                ModelCallUsageWrite(
                    usage_id="usage_run_1",
                    run_id="run_1",
                    model_call_id="call_run_1",
                    usage_kind=ModelUsageKind.INPUT_TOKENS,
                    quantity=10,
                    unit=ModelUsageUnit.TOKENS,
                    provider_usage_key="input_tokens",
                    cost_micros=100,
                    currency="USD",
                ),
            ),
            artifacts=(
                RunArtifactWrite(
                    artifact_id="artifact_run_1_prompt",
                    run_id="run_1",
                    step_id="step_run_1_source_binding",
                    model_call_id="call_run_1",
                    artifact_kind=ArtifactKind.PROMPT,
                    content_hash="sha256:run1",
                    content="run 1 prompt",
                    content_type="text/plain",
                    size_bytes=12,
                ),
            ),
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id="run_2",
            question_id="question_1",
            run_number=2,
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.RETRY,
            base_run_id="run_1",
            adapter_ref="django_drf:test",
            runtime_version="test",
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_contract",
            run_id="run_2",
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_compile",
            run_id="run_2",
            sequence=2,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id="run_2",
            sequence=3,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    _record_catalog_endpoint(recorder, run_id="run_2")
    recorder.record_source_read(
        SourceReadWrite(
            source_read_id="source_read_1",
            run_id="run_2",
            step_id="step_execute",
            catalog_endpoint_id="11111111-1111-4111-8111-111111111111",
            status=SourceReadStatus.SUCCEEDED,
            row_count=2,
            completeness_json={"complete": True},
            response_hash="sha256:stores",
        )
    )
    _record_program_invocation(recorder, run_id="run_2")
    recorder.record_answered_result(
        AnsweredRunResultWrite(
            result=RunResultWrite(
                run_result_id="result_1",
                run_id="run_2",
                result_kind=RunResultKind.ANSWERED,
            ),
            requested_facts=(
                RequestedFactWrite(
                    requested_fact_id="fact_1",
                    run_id="run_2",
                    produced_by_step_id="step_contract",
                    fact_key="fact_1",
                    answer_expression_family="scalar_aggregate",
                ),
            ),
            fact_results=(
                FactResultWrite(
                    fact_result_id="fact_result_1",
                    run_id="run_2",
                    requested_fact_id="fact_1",
                    produced_by_step_id="step_execute",
                    result_kind=FactResultKind.ANSWERED,
                ),
            ),
            proof_graphs=(
                ExecutionProofGraphWrite(
                    proof_graph_id="proof_1",
                    run_id="run_2",
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
                run_id="run_2",
                run_result_id="result_1",
            ),
            outputs=(
                AnswerOutputWrite(
                    answer_output_id="answer_output_1",
                    run_id="run_2",
                    answer_id="answer_1",
                    fact_result_id="fact_result_1",
                    output_key="answer_1",
                    value_kind=AnswerValueKind.NUMBER,
                    value_json={"kind": "number", "value": 2},
                    proof_node_refs_json=[_answer_proof_node_ref()],
                ),
            ),
        )
    )

    report = RuntimeUsageService(DjangoObservabilityQuery()).for_answer("answer_1")

    assert [call.model_call_id for call in report.calls] == ["call_run_1"]
    assert report.cost_micros_by_currency == {"USD": 100}
    assert report.calls[0].artifacts == ()


def test_django_observability_query_answer_scope_includes_clarification_lineage() -> (
    None
):
    recorder = DjangoLineageRecorder()
    _record_run_spine(recorder)
    recorder.record_step(
        RunStepWrite(
            step_id="step_clarification_source_binding",
            run_id="run_1",
            sequence=1,
            step_key=RunStepKey.SOURCE_BINDING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_model_call_audit(
        ModelCallAuditWrite(
            model_call=ModelCallWrite(
                model_call_id="call_clarification_run",
                run_id="run_1",
                step_id="step_clarification_source_binding",
                call_index=1,
                provider="openai",
                model_key="gpt-test",
                status=ModelCallStatus.SUCCEEDED,
            ),
            usage_rows=(
                ModelCallUsageWrite(
                    usage_id="usage_clarification_run",
                    run_id="run_1",
                    model_call_id="call_clarification_run",
                    usage_kind=ModelUsageKind.INPUT_TOKENS,
                    quantity=11,
                    unit=ModelUsageUnit.TOKENS,
                    provider_usage_key="input_tokens",
                    cost_micros=110,
                    currency="USD",
                ),
            ),
        )
    )
    recorder.record_clarification_request(
        ClarificationRequestWrite(
            clarification_id="clarification_1",
            run_id="run_1",
            step_id="step_clarification_source_binding",
            payload_json={
                "id": "clarification_1",
                "need": "target_reference",
                "reason": "multiple_matching_entities",
                "owner": "grounding",
                "continuation": {
                        "kind": "grounding",
                        "knownInputId": "store",
                        "acceptsFreeText": False,
                },
                "requestedFactId": "clarification_fact_1",
                "question": "Which matching store should I use?",
                "subjects": [
                    {
                        "kind": "question_input",
                        "id": "store",
                        "label": "store",
                        "sourceText": "",
                        "options": [
                            {
                                "id": "store_1",
                                "label": "Store 1",
                                "entityKind": "store",
                                "keyId": "primary_key",
                                "keyComponents": [
                                    {"componentId": "store_id", "value": "store_1"}
                                ],
                                "matchedField": "store_id",
                                "matchedValue": "store_1",
                                "resolverReadId": "list_stores",
                            }
                        ],
                    }
                ],
                "evidence": [],
            },
        )
    )
    recorder.record_clarification_response(
        ClarificationResponseWrite(
            response_id="clarification_response_1",
            run_id="run_1",
            clarification_id="clarification_1",
            evidence_ref="clarification_response:clarification_response_1",
            selected_option_id="store_1",
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_run_1_contract",
            run_id="run_1",
            sequence=2,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_run_1_compile",
            run_id="run_1",
            sequence=3,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_run_1_execute",
            run_id="run_1",
            sequence=4,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    _record_catalog_endpoint(recorder, run_id="run_1")
    recorder.record_source_read(
        SourceReadWrite(
            source_read_id="source_read_1",
            run_id="run_1",
            step_id="step_run_1_execute",
            catalog_endpoint_id="11111111-1111-4111-8111-111111111111",
            status=SourceReadStatus.SUCCEEDED,
            row_count=2,
            completeness_json={"complete": True},
            response_hash="sha256:stores",
        )
    )
    _record_program_invocation(recorder, run_id="run_1")
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
                    produced_by_step_id="step_run_1_contract",
                    fact_key="fact_1",
                    answer_expression_family="scalar_aggregate",
                ),
            ),
            fact_results=(
                FactResultWrite(
                    fact_result_id="fact_result_1",
                    run_id="run_1",
                    requested_fact_id="fact_1",
                    produced_by_step_id="step_run_1_execute",
                    result_kind=FactResultKind.ANSWERED,
                ),
            ),
            proof_graphs=(
                ExecutionProofGraphWrite(
                    proof_graph_id="proof_1",
                    run_id="run_1",
                    fact_result_id="fact_result_1",
                    compile_step_id="step_run_1_compile",
                    execute_step_id="step_run_1_execute",
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
    )

    report = RuntimeUsageService(DjangoObservabilityQuery()).for_answer("answer_1")

    assert [call.model_call_id for call in report.calls] == ["call_clarification_run"]
    assert report.cost_micros_by_currency == {"USD": 110}


def _record_run_spine(recorder: LineageRecorderPort) -> None:
    recorder.ensure_conversation(
        ConversationWrite(conversation_id="conversation_1", tenant_id="tenant_1")
    )
    recorder.record_question(
        QuestionWrite(
            question_id="question_1",
            conversation_id="conversation_1",
            conversation_sequence=1,
            original_question="How many stores are open?",
        )
    )
    recorder.start_run(
        QuestionRunWrite(
            run_id="run_1",
            question_id="question_1",
            run_number=1,
            kind=QuestionRunKind.MODEL_ASSISTED,
            trigger_kind=RunTriggerKind.INITIAL,
            adapter_ref="django_drf:test",
            runtime_version="test",
        )
    )


def _record_program_invocation(
    recorder: LineageRecorderPort,
    *,
    run_id: str,
) -> None:
    recorder.record_program_invocation(
        ProgramInvocationBundleWrite(
            program=AnswerProgramWrite(
                program_id=f"{run_id}:program",
                schema_revision=1,
                canonical_json="{}",
            ),
            invocation=ProgramInvocationWrite(
                invocation_id=f"{run_id}:invocation",
                run_id=run_id,
                program_id=f"{run_id}:program",
                bindings_json="{}",
                kind=ProgramInvocationKind.COMPILED_QUESTION,
            ),
        )
    )


def _record_catalog_endpoint(recorder: LineageRecorderPort, *, run_id: str) -> None:
    recorder.record_catalog_endpoint(
        CatalogEndpointWrite(
            catalog_endpoint_id="11111111-1111-4111-8111-111111111111",
            run_id=run_id,
            catalog_endpoint_key="django_retail_ops_list_store_list:test",
            endpoint_name="list_store_list",
            framework_kind=FrameworkKind.DJANGO_DRF,
            source_namespace_kind=SourceNamespaceKind.DJANGO_APP,
            source_namespace_path_json=("retail_ops",),
            route_method="GET",
            route_path_template="/v1/stores/",
            handler_ref="apps.retail_ops.views.StoreListView",
            domain_resource_names_json=("store",),
        )
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
