from __future__ import annotations

import pytest

from fervis.lineage.enums import (
    ArtifactKind,
    ConversationOriginKind,
    MemoryArtifactSourceKind,
    RunTriggerKind,
    SourceReadStatus,
)
from fervis.lineage.recorder import (
    ConversationWrite,
    MemoryArtifactWrite,
    QuestionRunWrite,
    RunArtifactWrite,
    SourceReadWrite,
)


def test_recorder_contract_accepts_clarification_trigger_shape() -> None:
    run = QuestionRunWrite(
        run_id="run_2",
        question_id="q_1",
        run_number=2,
        trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
        trigger_clarification_response_run_id="run_1",
        trigger_clarification_response_id="response_1",
        integrated_question="Question with clarification.",
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )

    assert run.trigger_clarification_response_run_id == "run_1"


def test_recorder_contract_accepts_complete_conversation_fork_shape() -> None:
    conversation = ConversationWrite(
        conversation_id="cv_2",
        tenant_id="tenant_1",
        origin_kind=ConversationOriginKind.FORK,
        parent_conversation_id="cv_1",
        forked_after_question_id="q_1",
        forked_after_run_id="run_1",
    )

    assert conversation.parent_conversation_id == "cv_1"


def test_recorder_contract_rejects_initial_conversation_with_fork_fields() -> None:
    with pytest.raises(ValueError, match="parent_conversation_id must be absent"):
        ConversationWrite(
            conversation_id="cv_1",
            tenant_id="tenant_1",
            origin_kind=ConversationOriginKind.INITIAL,
            parent_conversation_id="cv_0",
        )


def test_recorder_contract_rejects_incomplete_conversation_fork_shape() -> None:
    with pytest.raises(ValueError, match="forked_after_question_id is required"):
        ConversationWrite(
            conversation_id="cv_2",
            tenant_id="tenant_1",
            origin_kind=ConversationOriginKind.FORK,
            parent_conversation_id="cv_1",
            forked_after_run_id="run_1",
        )


def test_recorder_contract_rejects_conflicting_run_trigger_shape() -> None:
    with pytest.raises(ValueError, match="previous_run_id must be absent"):
        QuestionRunWrite(
            run_id="run_2",
            question_id="q_1",
            run_number=2,
            trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
            previous_run_id="run_1",
            trigger_clarification_response_run_id="run_1",
            trigger_clarification_response_id="response_1",
            integrated_question="Question with clarification.",
            adapter_ref="django_drf:test",
            runtime_version="test-runtime",
        )


def test_recorder_contract_rejects_incomplete_successful_source_read() -> None:
    with pytest.raises(ValueError, match="response_hash is required"):
        SourceReadWrite(
            source_read_id="source_read_1",
            run_id="run_1",
            step_id="step_execute",
            catalog_endpoint_id="11111111-1111-4111-8111-111111111111",
            status=SourceReadStatus.SUCCEEDED,
            row_count=1,
            completeness_json={"complete": True},
            response_hash="",
        )


def test_recorder_contract_rejects_model_artifact_without_model_call() -> None:
    with pytest.raises(ValueError, match="artifacts require model_call_id"):
        RunArtifactWrite(
            artifact_id="artifact_prompt",
            run_id="run_1",
            step_id="step_model",
            artifact_kind=ArtifactKind.PROMPT,
            content_hash="sha256:prompt",
            content="prompt text",
            content_type="text/plain",
            size_bytes=11,
        )


def test_recorder_contract_rejects_blank_external_artifact_reference() -> None:
    with pytest.raises(ValueError, match="storage_ref cannot be blank"):
        RunArtifactWrite(
            artifact_id="artifact_external",
            run_id="run_1",
            step_id="step_execute",
            artifact_kind=ArtifactKind.DETERMINISTIC_OUTPUT,
            content_hash="sha256:artifact",
            storage_ref="",
            content_type="application/json",
            size_bytes=10,
        )


def test_recorder_contract_rejects_memory_artifact_source_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="payload sourceKind must match"):
        MemoryArtifactWrite(
            memory_artifact_id="memory_1",
            run_id="run_1",
            produced_by_step_id="step_execute",
            source_kind=MemoryArtifactSourceKind.FACT_RESULT,
            fact_result_id="fact_result_1",
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "known_input",
                "artifactId": "memory_1",
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


def test_recorder_contract_rejects_memory_artifact_payload_id_mismatch() -> None:
    with pytest.raises(ValueError, match="payload artifactId must match"):
        MemoryArtifactWrite(
            memory_artifact_id="memory_1",
            run_id="run_1",
            produced_by_step_id="step_execute",
            source_kind=MemoryArtifactSourceKind.FACT_RESULT,
            fact_result_id="fact_result_1",
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "fact_result",
                "artifactId": "memory_2",
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


def test_recorder_contract_rejects_memory_artifact_missing_outcome() -> None:
    with pytest.raises(ValueError, match="outcome is required"):
        MemoryArtifactWrite(
            memory_artifact_id="memory_1",
            run_id="run_1",
            produced_by_step_id="step_execute",
            source_kind=MemoryArtifactSourceKind.FACT_RESULT,
            fact_result_id="fact_result_1",
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "fact_result",
                "artifactId": "memory_1",
                "addresses": [
                    {
                        "address": "value.answer_1",
                        "kind": "value",
                        "value": {"type": "number", "value": "1"},
                    }
                ],
            },
        )


def test_recorder_contract_accepts_requested_fact_memory_without_addresses() -> None:
    artifact = MemoryArtifactWrite(
        memory_artifact_id="memory_requested_fact_1",
        run_id="run_1",
        produced_by_step_id="step_question_contract",
        source_kind=MemoryArtifactSourceKind.REQUESTED_FACT,
        requested_fact_id="fact_1",
        payload_schema="fervis.memory_artifact",
        payload_schema_rev=1,
        payload_json={
            "sourceKind": "requested_fact",
            "artifactId": "memory_requested_fact_1",
            "outcome": "answered",
            "provenance": {
                "question_contract": {
                    "answer_requests": [
                        {"id": "fact_1", "answer_fact": "store count"}
                    ]
                }
            },
        },
    )

    assert artifact.source_kind is MemoryArtifactSourceKind.REQUESTED_FACT


def test_recorder_contract_rejects_requested_fact_memory_with_addresses() -> None:
    with pytest.raises(
        ValueError, match="requested_fact memory artifacts cannot carry addresses"
    ):
        MemoryArtifactWrite(
            memory_artifact_id="memory_requested_fact_1",
            run_id="run_1",
            produced_by_step_id="step_question_contract",
            source_kind=MemoryArtifactSourceKind.REQUESTED_FACT,
            requested_fact_id="fact_1",
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "requested_fact",
                "artifactId": "memory_requested_fact_1",
                "outcome": "answered",
                "addresses": [
                    {
                        "address": "entity.input_1",
                        "kind": "entity",
                        "resource": "area",
                        "identity": {"area_id": "area_1"},
                    }
                ],
            },
        )


def test_recorder_contract_accepts_known_input_memory_with_addresses() -> None:
    artifact = MemoryArtifactWrite(
        memory_artifact_id="memory_known_input_1",
        run_id="run_1",
        produced_by_step_id="step_grounding",
        source_kind=MemoryArtifactSourceKind.KNOWN_INPUT,
        payload_schema="fervis.memory_artifact",
        payload_schema_rev=1,
        payload_json={
            "sourceKind": "known_input",
            "artifactId": "memory_known_input_1",
            "outcome": "answered",
            "addresses": [
                {
                    "address": "entity.input_1",
                    "kind": "entity",
                    "resource": "area",
                    "identity": {"area_id": "area_1"},
                }
            ],
        },
    )

    assert artifact.source_kind is MemoryArtifactSourceKind.KNOWN_INPUT


def test_recorder_contract_rejects_known_input_memory_without_addresses() -> None:
    with pytest.raises(ValueError, match="known_input memory artifacts require addresses"):
        MemoryArtifactWrite(
            memory_artifact_id="memory_known_input_1",
            run_id="run_1",
            produced_by_step_id="step_grounding",
            source_kind=MemoryArtifactSourceKind.KNOWN_INPUT,
            payload_schema="fervis.memory_artifact",
            payload_schema_rev=1,
            payload_json={
                "sourceKind": "known_input",
                "artifactId": "memory_known_input_1",
                "outcome": "answered",
            },
        )
