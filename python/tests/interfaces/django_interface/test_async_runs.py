from types import SimpleNamespace
import json
import uuid

import pytest
from django.db import IntegrityError
from django.db import transaction
from django.utils import timezone

from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.lookup.errors import ErrorCode
from fervis.observability.event_contracts import EventPayloadKey
from fervis.interfaces.django.run_views import get_run_view
from fervis.run_work.queue.django.models import (
    RunWorkItem,
    RunWorkStatus,
)
from fervis.interfaces.django.worker import process_run_batch
from fervis.run_work.queue.django.queue import enqueue_run_work_item
from fervis.run_work.queue.django.queue import claim_run_work_items
from fervis.run_work.queue.django.queue import StaleRunLease
from fervis.run_work.worker import WorkerRunResult
from fervis.interfaces.django.runs import get_run
from fervis.interfaces.django.runs import process_run_work
from fervis.interfaces.django import runs as runs_module
from fervis.interfaces.django.question_run_ports import (
    DjangoQuestionLineagePort,
)
from fervis.lineage.enums import (
    AnswerValueKind,
    ClarificationBasis,
    FactResultKind,
    MemoryArtifactSourceKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
)
from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.models import (
    Answer,
    AnswerOutput,
    ClarificationRequest,
    Conversation,
    FactResult,
    MemoryArtifact,
    Question,
    QuestionRun,
    RequestedFact,
    RunResult,
    RunStep,
    RuntimeErrorDetail,
)
from fervis.lineage.recorder import (
    ModelCallAuditWrite,
    ModelCallUsageWrite,
    ModelCallWrite,
    RunStepWrite,
)
from tests.interfaces.django_interface.helpers import (
    install_test_model_adapter,
    post_fervis_question,
    question_run_detail_url,
)
from fervis.lookup.orchestration.service import LookupService
from fervis.lookup.orchestration.result import PlannerRunResult


@pytest.mark.django_db
def test_django_question_run_lineage_port_uses_canonical_lineage_artifacts():
    read_context_ref = ReadContextRef(scheme="django_principal", key="user-memory")
    conversation = Conversation.objects.create(
        conversation_id="conversation_memory_lineage",
        tenant_id="tenant-1",
        read_context_ref=read_context_ref.to_storage_dict(),
    )
    _answered_lineage_run(
        conversation=conversation,
        question_id="question_1",
        run_id="run_1",
        sequence=1,
        question="First question",
        value="first",
    )
    _work_item_for_memory_run(
        run_id="run_1",
        read_context_ref=read_context_ref,
    )
    _answered_lineage_run(
        conversation=conversation,
        question_id="question_2",
        run_id="run_2",
        sequence=2,
        question="Second question",
        value="second",
    )
    _work_item_for_memory_run(
        run_id="run_2",
        read_context_ref=read_context_ref,
    )

    context = DjangoQuestionLineagePort().conversation_memory_context(
        conversation_id="conversation_memory_lineage",
        authority=ReadAuthority(
            tenant_id="tenant-1",
            read_context_ref=read_context_ref,
        ),
    )

    assert [
        (artifact["artifactId"], artifact["sourceQuestion"], artifact["sourceAnswer"])
        for artifact in context["factArtifacts"]
    ] == [
        ("run_1.memory", "First question", "first"),
        ("run_2.memory", "Second question", "second"),
    ]


def _answered_lineage_run(
    *,
    conversation: Conversation,
    question_id: str,
    run_id: str,
    sequence: int,
    question: str,
    value: str,
) -> None:
    question_row = Question.objects.create(
        question_id=question_id,
        conversation=conversation,
        conversation_sequence=sequence,
        original_question=question,
    )
    run = QuestionRun.objects.create(
        run_id=run_id,
        question=question_row,
        run_number=1,
        trigger_kind=RunTriggerKind.INITIAL.value,
        integrated_question=question,
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )
    step = RunStep.objects.create(
        step_id=f"{run_id}.question_contract",
        run=run,
        sequence=1,
        step_key=RunStepKey.QUESTION_CONTRACT.value,
        kind=RunStepKind.MODEL_TURN.value,
    )
    run_result = RunResult.objects.create(
        run_result_id=f"{run_id}.result",
        run=run,
        result_kind=RunResultKind.ANSWERED.value,
    )
    requested_fact = RequestedFact.objects.create(
        requested_fact_id=f"{run_id}.fact",
        run=run,
        produced_by_step=step,
        fact_key="fact_1",
        description=f"answer for {question}",
        answer_expression_family="scalar",
        requested_fact_json={
            "id": "fact_1",
            "answer_fact": f"answer for {question}",
            "answer_outputs": [{"id": "answer_1", "description": "answer"}],
        },
    )
    fact_result = FactResult.objects.create(
        fact_result_id=f"{run_id}.fact_result",
        run=run,
        requested_fact=requested_fact,
        produced_by_step=step,
        result_kind=FactResultKind.ANSWERED.value,
    )
    answer = Answer.objects.create(
        answer_id=f"{run_id}.answer",
        run=run,
        run_result=run_result,
    )
    AnswerOutput.objects.create(
        answer_output_id=f"{run_id}.answer_output",
        run=run,
        answer=answer,
        fact_result=fact_result,
        output_key="answer_1",
        value_kind=AnswerValueKind.TEXT.value,
        value_json={"kind": "text", "value": value},
        proof_node_refs_json=[f"answer_output:{run_id}:answer_1"],
    )
    MemoryArtifact.objects.create(
        memory_artifact_id=f"{run_id}.memory",
        run=run,
        produced_by_step=step,
        source_kind=MemoryArtifactSourceKind.FACT_RESULT.value,
        fact_result=fact_result,
        payload_schema="fervis.memory_artifact",
        payload_schema_rev=1,
        payload_json={
            "sourceKind": "fact_result",
            "artifactId": f"{run_id}.memory",
            "outcome": "answered",
            "addresses": [
                {
                    "address": "value.answer_1",
                    "kind": "value",
                    "value": {"type": "text", "value": value},
                }
            ],
            "provenance": {"runId": run_id},
            "sourceQuestion": question,
            "sourceAnswer": value,
        },
    )


def _work_item_for_memory_run(
    *,
    run_id: str,
    read_context_ref: ReadContextRef,
) -> None:
    RunWorkItem.objects.create(
        run_id=run_id,
        conversation_id="conversation_memory_lineage",
        tenant_id="tenant-1",
        user_id=str(read_context_ref.key),
        provider="anthropic",
        model_key="HAIKU",
        question=f"question for {run_id}",
        conversation_context={},
        runtime_context={},
        read_context_ref=read_context_ref.to_storage_dict(),
        status=RunWorkStatus.COMPLETED,
        max_budget_usd=0.5,
    )


def _record_terminal_run_result(run_id: str) -> None:
    run = QuestionRun.objects.get(run_id=run_id)
    RunResult.objects.create(
        run_result_id=f"{run_id}.terminal",
        run=run,
        result_kind=RunResultKind.ANSWERED.value,
    )


@pytest.mark.django_db
def test_enqueued_message_creates_canonical_lineage_root(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)

    response = _post_message(
        api_client,
        conversation["conversationId"],
        "How many stores are open?",
    )

    run_id = response.json()["currentRunId"]
    question = Question.objects.get(conversation_id=conversation["conversationId"])
    run = QuestionRun.objects.get(run_id=run_id)
    assert {
        "question_text": question.original_question,
        "origin_message_ref_present": bool(question.origin_message_ref),
        "conversation_sequence": question.conversation_sequence,
        "run_question_id": run.question_id,
        "run_number": run.run_number,
        "trigger_kind": run.trigger_kind,
        "integrated_question": run.integrated_question,
    } == {
        "question_text": "How many stores are open?",
        "origin_message_ref_present": True,
        "conversation_sequence": 1,
        "run_question_id": question.question_id,
        "run_number": 1,
        "trigger_kind": RunTriggerKind.INITIAL.value,
        "integrated_question": "How many stores are open?",
    }


@pytest.mark.django_db
def test_runtime_model_turn_failure_records_canonical_run_step(
    api_client,
    fervis_foundation_reset,
):
    class StopAfterFirstModelCallAdapter:
        provider_name = "anthropic"

        def generate(
            self,
            *,
            model_id: str | None = None,
            prompt: str,
            max_thinking_tokens: int,
            system_prompt: str = "",
            output_mode=None,
            tool_specs=(),
        ):
            del (
                model_id,
                prompt,
                max_thinking_tokens,
                system_prompt,
                output_mode,
                tool_specs,
            )
            raise RuntimeError("stop after deterministic preflight")

    install_test_model_adapter(StopAfterFirstModelCallAdapter())
    conversation = _create_conversation(api_client)
    response = _post_message(
        api_client,
        conversation["conversationId"],
        "list all active staff alphabetically",
    )
    run_id = response.json()["currentRunId"]

    process_run_batch(
        worker_id="test-worker",
        batch_size=1,
        lease_seconds=300,
    )

    step = RunStep.objects.get(run_id=run_id)
    assert {
        "step_key": step.step_key,
        "kind": step.kind,
        "sequence": step.sequence,
        "error_code": step.error_json.get(EventPayloadKey.ERROR_CODE),
    } == {
        "step_key": RunStepKey.QUESTION_CONTRACT.value,
        "kind": RunStepKind.MODEL_TURN.value,
        "sequence": 1,
        "error_code": ErrorCode.PROVIDER_RUNTIME_FAILED,
    }


def _create_conversation(api_client):
    del api_client
    return {"conversationId": str(uuid.uuid4())}


def _post_message(api_client, conversation_id, message="list stores", payload=None):
    return post_fervis_question(
        api_client,
        message,
        conversation_id=conversation_id,
        payload=payload,
    )


def _record_lineage_model_usage(
    run_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    cost_micros: int,
) -> None:
    recorder = DjangoLineageRecorder()
    step_id = f"{run_id}.fact_planning"
    call_id = f"{run_id}.model_call"
    recorder.record_step(
        RunStepWrite(
            step_id=step_id,
            run_id=run_id,
            sequence=1,
            step_key=RunStepKey.FACT_PLANNING,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_model_call_audit(
        ModelCallAuditWrite(
            model_call=ModelCallWrite(
                model_call_id=call_id,
                run_id=run_id,
                step_id=step_id,
                call_index=1,
                provider="test",
                model_key="HAIKU",
                status=ModelCallStatus.SUCCEEDED,
            ),
            usage_rows=(
                _model_usage(
                    run_id,
                    call_id,
                    usage_id=f"{call_id}.input",
                    usage_kind=ModelUsageKind.INPUT_TOKENS,
                    quantity=input_tokens,
                    cost_micros=cost_micros,
                ),
                _model_usage(
                    run_id,
                    call_id,
                    usage_id=f"{call_id}.output",
                    usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                    quantity=output_tokens,
                    cost_micros=0,
                ),
                _model_usage(
                    run_id,
                    call_id,
                    usage_id=f"{call_id}.thinking",
                    usage_kind=ModelUsageKind.THINKING_TOKENS,
                    quantity=thinking_tokens,
                    cost_micros=0,
                ),
            ),
        )
    )


def _model_usage(
    run_id: str,
    model_call_id: str,
    *,
    usage_id: str,
    usage_kind: ModelUsageKind,
    quantity: int,
    cost_micros: int,
) -> ModelCallUsageWrite:
    return ModelCallUsageWrite(
        usage_id=usage_id,
        run_id=run_id,
        model_call_id=model_call_id,
        usage_kind=usage_kind,
        quantity=quantity,
        unit=ModelUsageUnit.TOKENS,
        provider_usage_key=usage_kind.value,
        cost_micros=cost_micros,
        currency="USD",
    )


def _cycle_summary(cycle):
    return {
        "claimed_count": cycle.claimed_count,
        "completed_count": cycle.completed_count,
        "failed_count": cycle.failed_count,
    }


@pytest.mark.django_db
def test_message_post_enqueues_without_running_planner_runtime(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("planner runtime must not run inside POST /messages/")

    monkeypatch.setattr(LookupService, "run_lookup", fail_if_called)
    conversation = _create_conversation(api_client)

    response = _post_message(api_client, conversation["conversationId"])

    body = response.json()
    assert (
        response.status_code,
        body["status"],
        body["answer"],
        RunWorkItem.objects.filter(run_id=body["currentRunId"]).exists(),
    ) == (202, "RUNNING", None, True)


@pytest.mark.django_db
def test_worker_records_question_contract_failure_before_execution(
    api_client,
    fervis_foundation_reset,
):
    class StopAfterFirstModelCallAdapter:
        provider_name = "anthropic"

        def generate(
            self,
            *,
            model_id: str | None = None,
            prompt: str,
            max_thinking_tokens: int,
            system_prompt: str = "",
            output_mode=None,
            tool_specs=(),
        ):
            del model_id, system_prompt
            raise RuntimeError("stop after deterministic preflight")

    install_test_model_adapter(StopAfterFirstModelCallAdapter())
    conversation = _create_conversation(api_client)
    response = _post_message(
        api_client,
        conversation["conversationId"],
        "list all active staff alphabetically",
    )
    run_id = response.json()["currentRunId"]

    process_run_batch(
        worker_id="test-worker",
        batch_size=1,
        lease_seconds=300,
    )

    run = get_run(run_id)
    failure = RuntimeErrorDetail.objects.get(run_id=run_id)
    assert {
        "run_status": run["status"],
        "run_error": run["error"],
        "step_names": [step["toolName"] for step in run["steps"]],
        "step_error": run["steps"][0]["responseBody"]["errorCode"],
        "failure_error_code": failure.error_kind,
        "failure_message": failure.message,
    } == {
        "run_status": "FAILED",
        "run_error": ErrorCode.PROVIDER_RUNTIME_FAILED,
        "step_names": ["question_contract"],
        "step_error": ErrorCode.PROVIDER_RUNTIME_FAILED,
        "failure_error_code": ErrorCode.PROVIDER_RUNTIME_FAILED,
        "failure_message": ErrorCode.PROVIDER_RUNTIME_FAILED,
    }


@pytest.mark.django_db
def test_worker_fails_before_answer_synthesis_when_budget_is_exceeded(
    api_client,
    fervis_foundation_reset,
):
    class ImpossiblePlanAdapter:
        provider_name = "anthropic"

        def generate(
            self,
            *,
            model_id: str | None = None,
            prompt: str,
            max_thinking_tokens: int,
            system_prompt: str = "",
            output_mode=None,
            tool_specs=(),
        ):
            del model_id, system_prompt
            tool_name = tool_specs[0].name if tool_specs else ""
            if tool_name == "submit_answer_request_contract":
                return {
                    "answer": json.dumps(
                        {
                            "tool": "submit_answer_request_contract",
                            "arguments": {
                                "kind": "question_contract",
                                "answer_requests_count": 1,
                                "question_inputs": [],
                                "answer_requests": [
                                    {
                                        "answer_fact": "restricted fact",
                                        "answer_expression": {"family": "scalar_value"},
                                        "answer_subject": {
                                            "subject_text": "restricted fact",
                                            "instance_interpretation": {
                                                "kind": "NORMAL_BUSINESS_INSTANCE"
                                            },
                                        },
                                        "answer_population": {
                                            "population_label": "restricted fact",
                                            "counted_unit": "restricted fact",
                                            "membership_tests": [
                                                {
                                                    "test_id": "pop_test_1",
                                                    "kind": "SUBJECT_IDENTITY",
                                                    "polarity": "MUST_PASS",
                                                        "test_question": (
                                                            "Does the row/value represent restricted fact?"
                                                        ),
                                                        "owned_question_input_refs": [],
                                                    }
                                                ],
                                            },
                                        "answer_outputs": [
                                            {
                                                "description": "restricted fact",
                                            }
                                        ],
                                        "used_question_inputs": [],
                                    }
                                ],
                                "question_input_inventory_check": {
                                    "all_input_like_phrases_declared": True,
                                },
                            },
                        }
                    ),
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "thinkingTokens": 0,
                        "costUsd": 0.02,
                        "inputCostUsd": 0.02,
                        "outputCostUsd": 0,
                        "thinkingCostUsd": 0,
                    },
                }
            raise AssertionError(f"unexpected tool: {tool_name}")

    install_test_model_adapter(ImpossiblePlanAdapter())
    conversation = _create_conversation(api_client)
    response = _post_message(
        api_client,
        conversation["conversationId"],
        "what is the restricted fact?",
        payload={"maxBudgetUsd": 0.01},
    )
    run_id = response.json()["currentRunId"]

    process_run_batch(
        worker_id="test-worker",
        batch_size=1,
        lease_seconds=300,
    )

    run = get_run(run_id)
    failure = RuntimeErrorDetail.objects.get(run_id=run_id)
    assert {
        "run_status": run["status"],
        "run_error": run["error"],
        "failure_error_code": failure.error_kind,
    } == {
        "run_status": "FAILED",
        "run_error": "max_budget_exceeded",
        "failure_error_code": "policy_limit_exceeded",
    }


@pytest.mark.django_db
def test_worker_terminal_result_data_excludes_audit_handles(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    def fact_bearing_failure(self, **kwargs):
        del self, kwargs
        return PlannerRunResult(
            status="FAILED",
            answer=None,
            result_data={
                "rows": [
                    {
                        "name": "Location Alpha",
                        "proofRefs": ["read:list_location_list"],
                    }
                ],
                "proofRefs": ["read:list_location_list"],
            },
            usage={"inputTokens": 1, "outputTokens": 1, "thinkingTokens": 0},
            error="terminal_synthesis_failed",
        )

    monkeypatch.setattr(LookupService, "run_lookup", fact_bearing_failure)
    conversation = _create_conversation(api_client)
    response = _post_message(
        api_client,
        conversation["conversationId"],
        "what was the metric total?",
    )
    run_id = response.json()["currentRunId"]

    process_run_batch(
        worker_id="test-worker",
        batch_size=1,
        lease_seconds=300,
    )

    public_run = get_run(run_id)
    run = get_run_view(run_id)
    assert {
        "public_status": public_run["status"],
        "public_error": public_run["error"],
        "run_view_result_data": run["resultData"],
    } == {
        "public_status": "FAILED",
        "public_error": "terminal_synthesis_failed",
        "run_view_result_data": None,
    }


@pytest.mark.django_db
def test_run_view_projects_needs_clarification_result_data_in_canonical_shape(
    fervis_foundation_reset,
):
    conversation = Conversation.objects.create(
        conversation_id="conversation_clarification_result_data",
        tenant_id="tenant-1",
        read_context_ref=ReadContextRef(
            scheme="django_principal",
            key="user-1",
        ).to_storage_dict(),
    )
    question = Question.objects.create(
        question_id="question_clarification_result_data",
        conversation=conversation,
        conversation_sequence=1,
        original_question="How many sales happened there?",
    )
    run = QuestionRun.objects.create(
        run_id="run_clarification_result_data",
        question=question,
        run_number=1,
        trigger_kind=RunTriggerKind.INITIAL.value,
        integrated_question="How many sales happened there?",
        adapter_ref="django_drf:test",
        runtime_version="test-runtime",
    )
    step = RunStep.objects.create(
        step_id="run_clarification_result_data.grounding",
        run=run,
        sequence=1,
        step_key=RunStepKey.GROUNDING.value,
        kind=RunStepKind.MODEL_TURN.value,
    )
    RunResult.objects.create(
        run_result_id="run_clarification_result_data.result",
        run=run,
        result_kind=RunResultKind.FACTUAL_TERMINAL.value,
    )
    requested_fact = RequestedFact.objects.create(
        requested_fact_id="run_clarification_result_data.fact",
        run=run,
        produced_by_step=step,
        fact_key="fact_1",
        description="sales count",
        answer_expression_family="scalar",
        requested_fact_json={
            "id": "fact_1",
            "answer_fact": "sales count",
            "answer_outputs": [{"id": "answer_1", "description": "answer"}],
        },
    )
    fact_result = FactResult.objects.create(
        fact_result_id="run_clarification_result_data.fact_result",
        run=run,
        requested_fact=requested_fact,
        produced_by_step=step,
        result_kind=FactResultKind.NEEDS_CLARIFICATION.value,
    )
    ClarificationRequest.objects.create(
        clarification_id="clarification_1",
        run=run,
        fact_result=fact_result,
        basis=ClarificationBasis.UNRESOLVED_REFERENCE.value,
        question_text="Which store do you mean?",
        options_json=[{"id": "store_1", "label": "ABC Mall"}],
        evidence_refs_json=["known_input:store"],
    )

    run_view = get_run_view("run_clarification_result_data")

    assert run_view is not None
    assert run_view["status"] == "NEEDS_CLARIFICATION"
    assert run_view["resultData"] == {
        "kind": "needs_clarification",
        "details": {
            "clarifications": [
                {
                    "id": "clarification_1",
                    "basis": "unresolved_reference",
                    "question": "Which store do you mean?",
                    "requestedFactId": "run_clarification_result_data.fact",
                    "knownInputId": "store",
                    "availableOptions": [{"id": "store_1", "label": "ABC Mall"}],
                    "evidenceRefs": ["known_input:store"],
                    "factResultId": "run_clarification_result_data.fact_result",
                    "stepId": None,
                }
            ]
        },
    }


@pytest.mark.django_db
def test_fervis_worker_processes_queued_run_to_terminal(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    response = _post_message(api_client, conversation["conversationId"])
    question = response.json()
    run_id = question["currentRunId"]

    cycle = process_run_batch(
        worker_id="test-worker",
        batch_size=1,
        lease_seconds=300,
    )

    poll = api_client.get(
        question_run_detail_url(question["questionId"], run_id),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )
    assert (
        cycle.claimed_count,
        poll.status_code,
        poll.json()["status"] in {"COMPLETED", "NEEDS_CLARIFICATION", "FAILED"},
    ) == (1, 200, True)


@pytest.mark.django_db
def test_same_conversation_rejects_second_active_run(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    first = _post_message(api_client, conversation["conversationId"], "first")

    second = _post_message(api_client, conversation["conversationId"], "second")

    assert (
        first.status_code,
        second.status_code,
        second.json()["error"]["context"]["activeRunId"],
    ) == (202, 409, first.json()["currentRunId"])


@pytest.mark.django_db
def test_created_run_response_exposes_worker_snapshot(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)

    response = _post_message(api_client, conversation["conversationId"], "first")

    question = response.json()
    detail = api_client.get(
        question_run_detail_url(question["questionId"], question["currentRunId"]),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )

    worker = detail.json()["worker"]
    assert (response.status_code, worker["status"], worker["attemptCount"]) == (
        202,
        "QUEUED",
        0,
    )


@pytest.mark.django_db
def test_idempotent_integrity_race_returns_existing_same_key(
    fervis_foundation_reset,
    monkeypatch,
):
    existing = RunWorkItem.objects.create(
        run_id="existing-race-run",
        conversation_id="conversation-race",
        tenant_id="tenant-race",
        user_id="user-race",
        provider="anthropic",
        model_key="HAIKU",
        question="already completed",
        conversation_context={},
        read_context_ref={
            "scheme": "django_principal",
            "key": "user-race",
            "tenant_key": None,
        },
        idempotency_key="same-key",
        status=RunWorkStatus.COMPLETED,
        max_budget_usd=0.5,
    )
    original_filter = RunWorkItem.objects.filter
    calls = {"idempotency": 0}

    def racing_create(**kwargs):
        raise IntegrityError("duplicate idempotency key")

    def filter_with_hidden_first_idempotent_lookup(*args, **kwargs):
        if kwargs.get("idempotency_key") == "same-key":
            calls["idempotency"] += 1
            if calls["idempotency"] == 1:
                return RunWorkItem.objects.none()
        return original_filter(*args, **kwargs)

    monkeypatch.setattr(RunWorkItem.objects, "create", racing_create)
    monkeypatch.setattr(
        RunWorkItem.objects,
        "filter",
        filter_with_hidden_first_idempotent_lookup,
    )

    enqueued = enqueue_run_work_item(
        run_id="new-race-run",
        conversation_id="conversation-race",
        tenant_id="tenant-race",
        user_id="user-race",
        question="same question",
        provider="anthropic",
        model_key="HAIKU",
        execution_mode="queued",
        conversation_context={},
        runtime_context={},
        read_context_ref={
            "scheme": "django_principal",
            "key": "user-race",
            "tenant_key": None,
        },
        idempotency_key="same-key",
        max_budget_usd=0.5,
        max_thinking_tokens=64,
    )

    assert (enqueued.item, enqueued.created, RunWorkItem.objects.count()) == (
        existing,
        False,
        1,
    )


@pytest.mark.django_db
def test_enqueue_integrity_error_does_not_break_outer_transaction(
    fervis_foundation_reset,
):
    RunWorkItem.objects.create(
        run_id="duplicate-run",
        conversation_id="other-conversation",
        tenant_id="tenant-race",
        user_id="user-race",
        provider="anthropic",
        model_key="HAIKU",
        question="already created",
        conversation_context={},
        max_budget_usd=0.5,
    )

    with transaction.atomic():
        with pytest.raises(IntegrityError):
            enqueue_run_work_item(
                run_id="duplicate-run",
                conversation_id="conversation-race",
                tenant_id="tenant-race",
                user_id="user-race",
                question="same run id",
                provider="anthropic",
                model_key="HAIKU",
                execution_mode="queued",
                conversation_context={},
                runtime_context={},
                read_context_ref={
                    "scheme": "django_principal",
                    "key": "user-race",
                    "tenant_key": None,
                },
                idempotency_key=None,
                max_budget_usd=0.5,
                max_thinking_tokens=64,
            )
        assert RunWorkItem.objects.count() == 1


@pytest.mark.django_db
def test_enqueue_run_work_item_persists_runtime_context(
    fervis_foundation_reset,
):
    enqueued = enqueue_run_work_item(
        run_id="runtime-context-run",
        conversation_id="conversation-runtime-context",
        tenant_id="tenant-runtime-context",
        user_id="user-runtime-context",
        question="same question",
        provider="anthropic",
        model_key="HAIKU",
        execution_mode="queued",
        conversation_context={},
        runtime_context={
            "caseId": "case-1",
            "goldsetRunId": "goldset-1",
            "certificationRunId": "goldset-1",
        },
        read_context_ref={
            "scheme": "django_principal",
            "key": "user-runtime-context",
            "tenant_key": None,
        },
        idempotency_key=None,
        max_budget_usd=0.5,
        max_thinking_tokens=64,
    )

    assert enqueued.item.runtime_context == {
        "caseId": "case-1",
        "goldsetRunId": "goldset-1",
        "certificationRunId": "goldset-1",
    }


@pytest.mark.django_db
def test_terminal_lineage_reconciles_stale_active_work_item(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    first = _post_message(api_client, conversation["conversationId"], "first").json()
    work_item = RunWorkItem.objects.get(run_id=first["currentRunId"])
    work_item.status = RunWorkStatus.RUNNING
    work_item.save(update_fields=["status", "updated_at"])
    _record_terminal_run_result(first["currentRunId"])

    second = _post_message(api_client, conversation["conversationId"], "second")

    work_item.refresh_from_db()
    assert (second.status_code, work_item.status) == (
        202,
        RunWorkStatus.COMPLETED,
    )


@pytest.mark.django_db
def test_expired_running_lease_is_claimed_by_next_worker(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    RunWorkItem.objects.filter(run_id=run["currentRunId"]).update(
        status=RunWorkStatus.RUNNING,
        lease_owner="worker-a",
        lease_expires_at=timezone.now() - timezone.timedelta(minutes=1),
    )

    claimed = claim_run_work_items(
        worker_id="worker-b",
        batch_size=1,
        lease_seconds=300,
    )

    work_item = RunWorkItem.objects.get(run_id=run["currentRunId"])
    assert (
        [item.run_id for item in claimed],
        work_item.status,
        work_item.lease_owner,
        work_item.attempt_count,
    ) == ([run["currentRunId"]], RunWorkStatus.RUNNING, "worker-b", 1)


@pytest.mark.django_db
def test_stale_worker_cannot_execute_after_expired_lease_is_reclaimed(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    first_claim = claim_run_work_items(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
    )
    first_attempt = first_claim[0].active_attempt
    RunWorkItem.objects.filter(run_id=run["currentRunId"]).update(
        lease_expires_at=timezone.now() - timezone.timedelta(minutes=1),
    )

    second_claim = claim_run_work_items(
        worker_id="worker-b",
        batch_size=1,
        lease_seconds=300,
    )

    with pytest.raises(StaleRunLease):
        process_run_work(
            run_id=run["currentRunId"],
            worker_id="worker-a",
            active_attempt=first_attempt,
        )

    work_item = RunWorkItem.objects.get(run_id=run["currentRunId"])
    assert (
        [item.run_id for item in first_claim],
        [item.run_id for item in second_claim],
        work_item.status,
        work_item.lease_owner,
        work_item.active_attempt,
    ) == (
        [run["currentRunId"]],
        [run["currentRunId"]],
        RunWorkStatus.RUNNING,
        "worker-b",
        second_claim[0].active_attempt,
    )


@pytest.mark.django_db
def test_stale_worker_failure_does_not_overwrite_terminal_work_item(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    first_claim = claim_run_work_items(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
    )
    first_attempt = first_claim[0].active_attempt
    _record_terminal_run_result(run["currentRunId"])
    RunWorkItem.objects.filter(run_id=run["currentRunId"]).update(
        status=RunWorkStatus.COMPLETED,
        completed_at=timezone.now(),
        lease_owner=None,
        lease_expires_at=None,
        last_error="",
    )

    with pytest.raises(StaleRunLease):
        runs_module.fail_run_work(
            run_id=run["currentRunId"],
            worker_id="worker-a",
            active_attempt=first_attempt,
            error="stale worker failure",
        )

    work_item = RunWorkItem.objects.get(run_id=run["currentRunId"])
    assert work_item.last_error == ""


@pytest.mark.django_db
def test_current_worker_failure_reconciles_existing_terminal_lineage(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    claimed = claim_run_work_items(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
    )
    _record_terminal_run_result(run["currentRunId"])

    terminal = runs_module.fail_run_work(
        run_id=run["currentRunId"],
        worker_id="worker-a",
        active_attempt=claimed[0].active_attempt,
        error="worker_crashed",
    )

    work_item = RunWorkItem.objects.get(run_id=run["currentRunId"])
    assert (
        terminal["status"],
        work_item.status,
        work_item.lease_owner,
        work_item.last_error,
    ) == (
        "COMPLETED",
        RunWorkStatus.COMPLETED,
        None,
        "",
    )


@pytest.mark.django_db
def test_expired_running_lease_stops_at_max_attempts(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    RunWorkItem.objects.filter(run_id=run["currentRunId"]).update(
        status=RunWorkStatus.RUNNING,
        lease_owner="worker-a",
        lease_expires_at=timezone.now() - timezone.timedelta(minutes=1),
        attempt_count=2,
        active_attempt=2,
        max_attempts=2,
    )
    _record_lineage_model_usage(
        run["currentRunId"],
        input_tokens=2,
        output_tokens=3,
        thinking_tokens=1,
        cost_micros=20_000,
    )

    claimed = claim_run_work_items(
        worker_id="worker-b",
        batch_size=1,
        lease_seconds=300,
    )

    work_item = RunWorkItem.objects.get(run_id=run["currentRunId"])
    terminal_run = get_run(run["currentRunId"])
    failure = RuntimeErrorDetail.objects.get(run_id=run["currentRunId"])
    assert {
        "claimed": claimed,
        "work_status": work_item.status,
        "work_completed": work_item.completed_at is not None,
        "work_lease_owner": work_item.lease_owner,
        "run_status": terminal_run["status"],
        "run_error": terminal_run["error"],
        "run_input_tokens": terminal_run["usage"]["inputTokens"],
        "run_cost": terminal_run["usage"]["costUsd"],
        "failure_error_code": failure.error_kind,
        "failure_message": failure.message,
    } == {
        "claimed": [],
        "work_status": RunWorkStatus.FAILED,
        "work_completed": True,
        "work_lease_owner": None,
        "run_status": "FAILED",
        "run_error": "run_max_attempts_exceeded",
        "run_input_tokens": 2,
        "run_cost": 0.02,
        "failure_error_code": "infrastructure_failed",
        "failure_message": "run_max_attempts_exceeded",
    }


@pytest.mark.django_db
def test_worker_failure_terminal_payload_preserves_recorded_usage(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    claimed = claim_run_work_items(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
    )
    _record_lineage_model_usage(
        run["currentRunId"],
        input_tokens=10,
        output_tokens=5,
        thinking_tokens=1,
        cost_micros=25_000,
    )

    failed = runs_module.fail_run_work(
        run_id=run["currentRunId"],
        worker_id="worker-a",
        active_attempt=claimed[0].active_attempt,
        error="worker_crashed",
    )
    derived_run_view = get_run_view(run["currentRunId"])

    assert (
        [item.run_id for item in claimed],
        failed["status"],
        failed["usage"]["inputTokens"],
        failed["usage"]["outputTokens"],
        failed["usage"]["thinkingTokens"],
        failed["usage"]["costUsd"],
        "usage" in (derived_run_view or {}),
    ) == ([run["currentRunId"]], "FAILED", 10, 5, 1, 0.025, False)


@pytest.mark.django_db
def test_already_terminal_worker_return_preserves_recorded_usage(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    run = _post_message(api_client, conversation["conversationId"], "first").json()
    claimed = claim_run_work_items(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
    )
    _record_lineage_model_usage(
        run["currentRunId"],
        input_tokens=8,
        output_tokens=4,
        thinking_tokens=2,
        cost_micros=18_000,
    )

    runs_module.fail_run_work(
        run_id=run["currentRunId"],
        worker_id="worker-a",
        active_attempt=claimed[0].active_attempt,
        error="worker_crashed",
    )

    terminal = process_run_work(
        run_id=run["currentRunId"],
        worker_id="worker-a",
        active_attempt=claimed[0].active_attempt,
    )

    assert (
        terminal["status"],
        terminal["usage"]["inputTokens"],
        terminal["usage"]["outputTokens"],
        terminal["usage"]["thinkingTokens"],
        terminal["usage"]["costUsd"],
    ) == ("FAILED", 8, 4, 2, 0.018)


def test_worker_batch_does_not_abort_when_failure_terminalization_loses_lease(
    monkeypatch,
):
    from fervis.run_work.queue.django import runner as worker_runner

    item = SimpleNamespace(
        run_id="run-stale",
        conversation_id="conversation-1",
        tenant_id="tenant-1",
        provider="fake",
        model_key="HAIKU",
        question="stale worker question",
        active_attempt=1,
    )
    monkeypatch.setattr(
        worker_runner,
        "claim_run_work_items",
        lambda **kwargs: [item],
    )
    monkeypatch.setattr(worker_runner, "queue_counts", lambda: {})
    cycle = worker_runner.process_run_batch(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
        worker=_FailingWorker(
            fail_error=StaleRunLease(
                run_id="run-stale",
                worker_id="worker-a",
                active_attempt=1,
            )
        ),
    )

    assert _cycle_summary(cycle) == {
        "claimed_count": 1,
        "completed_count": 0,
        "failed_count": 0,
    }


def test_worker_cycle_counts_failed_runtime_failure(monkeypatch):
    from fervis.run_work.queue.django import runner as worker_runner

    item = SimpleNamespace(
        run_id="run-failed",
        conversation_id="conversation-1",
        tenant_id="tenant-1",
        provider="fake",
        model_key="HAIKU",
        question="failed runtime question",
        active_attempt=1,
    )
    monkeypatch.setattr(
        worker_runner,
        "claim_run_work_items",
        lambda **kwargs: [item],
    )
    monkeypatch.setattr(worker_runner, "queue_counts", lambda: {})

    cycle = worker_runner.process_run_batch(
        worker_id="worker-a",
        batch_size=1,
        lease_seconds=300,
        worker=_StaticWorker(
            process_result={
                "status": "FAILED",
                "error": "planning_failed",
                "answer": "planning failed",
            }
        ),
    )

    assert _cycle_summary(cycle) == {
        "claimed_count": 1,
        "completed_count": 0,
        "failed_count": 1,
    }


class _StaticWorker:
    def __init__(self, *, process_result):
        self.process_result = process_result

    def process_run_work(self, **kwargs):
        active_attempt = int(kwargs["active_attempt"])
        return WorkerRunResult(
            run_id=str(kwargs["run_id"]),
            active_attempt=active_attempt,
            status=str(self.process_result["status"]),
            error=self.process_result.get("error"),
        )

    def fail_run_work(self, **kwargs):
        active_attempt = int(kwargs["active_attempt"])
        return WorkerRunResult(
            run_id=str(kwargs["run_id"]),
            active_attempt=active_attempt,
            status="FAILED",
            error=kwargs.get("error"),
        )


class _FailingWorker(_StaticWorker):
    def __init__(self, *, fail_error):
        super().__init__(process_result={})
        self.fail_error = fail_error

    def process_run_work(self, **kwargs):
        del kwargs
        raise RuntimeError("worker failed")

    def fail_run_work(self, **kwargs):
        del kwargs
        raise self.fail_error


@pytest.mark.django_db
def test_terminal_failed_work_item_does_not_starve_new_queued_runs(
    api_client,
    fervis_foundation_reset,
):
    conversation = _create_conversation(api_client)
    failed = _post_message(api_client, conversation["conversationId"], "first").json()
    RunWorkItem.objects.filter(run_id=failed["currentRunId"]).update(
        status=RunWorkStatus.FAILED,
        attempt_count=1,
        max_attempts=2,
    )
    second = _post_message(api_client, conversation["conversationId"], "second").json()

    cycle = process_run_batch(
        worker_id="test-worker",
        batch_size=1,
        lease_seconds=300,
    )

    assert (
        cycle.claimed_count,
        RunWorkItem.objects.get(run_id=second["currentRunId"]).status
        in {RunWorkStatus.COMPLETED, RunWorkStatus.FAILED},
    ) == (1, True)
