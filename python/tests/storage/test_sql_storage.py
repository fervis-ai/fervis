from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import sqlalchemy as sa
import pytest

from fervis.interfaces.agent.actions import inspect_question_action
from fervis.lookup.orchestration.result import LookupResult
from fervis.lookup.orchestration.request import LookupRequest
from fervis.host_api.contracts.authority import ReadContextRef
from fervis.host_api.credentials import (
    CapturedHeaderCredentialPolicy,
    capture_header_credential,
    overlay_from_header_credential,
)
from fervis.project.persistence.schema import metadata
from fervis.project.persistence.sqlite_engine import create_sqlite_engine
from fervis.interfaces.common.admission import ConfiguredModelPolicy
from fervis.interfaces.cli.contracts import FervisCliPorts
from fervis.interfaces.cli.dispatch import (
    run_fervis,
    run_auth_command,
    run_init_command,
    run_migrate_command,
)
from fervis.project import discover_project
from fervis.project.configuration import load_fervis_project_config
from fervis.lineage.recorder_core import LineageRecorder
from fervis.lineage.recorder import (
    ClarificationRequestWrite,
    ExecutionProofGraphWrite,
    FactResultWrite,
    LineageRecorderConflict,
    ModelCallAuditWrite,
    ModelCallWrite,
    RequestedFactWrite,
    RunStepWrite,
)
from fervis.lineage.enums import (
    ClarificationBasis,
    FactResultKind,
    ModelCallStatus,
    RunStepKey,
    RunStepKind,
)
from fervis.questions import (
    AskRequest,
    ContinueQuestionRequest,
    ExecutionMode,
    QuestionPrincipal,
)
from fervis.lineage.enums import RunTriggerKind
from fervis.run_work.events import CollectingQuestionRunEventSink
from fervis.run_work.queued_execution import LocalQueuedRunFollower
from fervis.run_work.worker import RunWorkBatchProcessor, RunWorkServiceWorker
from fervis.run_work import QueuedRunRequest
from fervis.questions.ports import (
    LookupExecutionRequest,
    LookupExecutionResult,
    QuestionLookupPort,
)
from fervis.storage.sql.bundle import sql_storage_bundle
from fervis.storage.sql.lineage_store import SQLLineageRecorderStore
from fervis.storage.sql.rows import now_utc
from fervis.storage.sql.terminal import (
    record_runtime_error_result,
    terminal_result_for_run,
)
from fervis.storage.sql.transaction import rollback_sql_transaction
from fervis.storage.sql.work_items import SQLWorkItemQueue
from tests.testkit.terminal_lineage import (
    TerminalAnswerWriter,
    make_terminal_answer_writer,
)


API_DIR = Path(__file__).resolve().parents[3]


def test_sql_storage_queued_ask_writes_lineage_and_work_item(tmp_path: Path) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    read_context_ref = ReadContextRef(scheme="fastapi_principal", key="user_7")

    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(
                principal_id="u1",
                tenant_id="t1",
                read_context_ref=read_context_ref,
            ),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
            idempotency_key="orders-today",
        )
    )

    assert result.status == "QUEUED"
    assert _count_rows(root, "fervis_conversation") == 1
    assert _count_rows(root, "fervis_question") == 1
    assert _count_rows(root, "fervis_question_run") == 1
    assert _count_rows(root, "fervis_run_work_item") == 1
    assert (
        _work_item_read_context_ref(root, result.run_id)
        == read_context_ref.to_storage_dict()
    )
    assert (
        _conversation_read_context_ref(root, result.conversation_id)
        == read_context_ref.to_storage_dict()
    )
    assert bundle.lineage_query.run_by_id(result.run_id).run_id == result.run_id


def test_sql_storage_round_trips_read_context_ref_into_lookup_request(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    lookup = _CapturingLookup()
    bundle = _storage_bundle(root, lookup=lookup)
    read_context_ref = ReadContextRef(scheme="fastapi_principal", key="user_7")
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(
                principal_id="u1",
                tenant_id="t1",
                read_context_ref=read_context_ref,
            ),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    [item] = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )

    bundle.run_work.process_queued_run(
        QueuedRunRequest(
            run_id=result.run_id,
            worker_id="worker-1",
            active_attempt=item.active_attempt,
        )
    )

    assert lookup.requests[0].read_context_ref == read_context_ref


def test_sql_storage_round_trips_delegated_credential_into_lookup_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    lookup = _CapturingLookup()
    bundle = _storage_bundle(root, lookup=lookup)
    monkeypatch.setenv("FERVIS_READ_CREDENTIAL_KEY", "test-secret")
    policy = CapturedHeaderCredentialPolicy(headers=("Authorization",))
    credential = capture_header_credential(
        request_headers={"Authorization": "Bearer queued-token"},
        policy=policy,
    )
    assert credential is not None
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(
                principal_id="u1",
                tenant_id="t1",
                read_context_ref=ReadContextRef(
                    scheme="fastapi_principal",
                    key="user_7",
                ),
                delegated_credential=credential,
            ),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    [item] = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )

    bundle.run_work.process_queued_run(
        QueuedRunRequest(
            run_id=result.run_id,
            worker_id="worker-1",
            active_attempt=item.active_attempt,
        )
    )

    stored_context = json.dumps(_work_item_runtime_context(root, result.run_id))
    assert "queued-token" not in stored_context
    replay = overlay_from_header_credential(
        lookup.requests[0].delegated_credential,
        policy=policy,
    )
    assert replay.headers == {"Authorization": "Bearer queued-token"}


def test_sql_storage_question_state_is_scoped_to_submitting_read_context(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    owner = ReadContextRef(scheme="fastapi_principal", key="user_7")
    other_read_context = ReadContextRef(scheme="fastapi_principal", key="user_8")
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(
                principal_id="u1",
                tenant_id="t1",
                read_context_ref=owner,
            ),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )

    owned_state = bundle.questions.get_question_state(
        result.question_id,
        principal=QuestionPrincipal(
            principal_id="u1",
            tenant_id="t1",
            read_context_ref=owner,
        ),
    )
    other_state = bundle.questions.get_question_state(
        result.question_id,
        principal=QuestionPrincipal(
            principal_id="u2",
            tenant_id="t1",
            read_context_ref=other_read_context,
        ),
    )

    assert owned_state is not None
    assert other_state is None


def test_sql_storage_lists_authorized_conversations_for_web_client(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    owner = _principal("owner", "owner")
    other = _principal("other", "other")
    older = bundle.questions.ask(
        AskRequest(
            question="How many orders came in yesterday?",
            principal=owner,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="conversation-older",
        )
    )
    newer = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=owner,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="conversation-newer",
        )
    )
    bundle.questions.ask(
        AskRequest(
            question="How many payments came in today?",
            principal=other,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="conversation-other",
        )
    )
    _set_run_created_at(root, older.run_id, "2026-06-26T09:00:00+00:00")
    _set_run_created_at(root, newer.run_id, "2026-06-27T10:15:00+00:00")

    conversations = bundle.questions.list_conversations(principal=owner)

    assert conversations == [
        {
            "conversationId": "conversation-newer",
            "firstQuestion": "How many orders came in today?",
            "latestQuestionId": newer.question_id,
            "currentRunId": newer.run_id,
            "status": "QUEUED",
            "runCount": 1,
            "updatedAt": "2026-06-27T10:15:00+00:00",
        },
        {
            "conversationId": "conversation-older",
            "firstQuestion": "How many orders came in yesterday?",
            "latestQuestionId": older.question_id,
            "currentRunId": older.run_id,
            "status": "QUEUED",
            "runCount": 1,
            "updatedAt": "2026-06-26T09:00:00+00:00",
        },
    ]


def test_sql_storage_question_state_uses_conversation_owner_not_work_item_context(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    owner = ReadContextRef(scheme="fastapi_principal", key="user_7")
    worker_context = ReadContextRef(scheme="fastapi_principal", key="worker-context")
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(
                principal_id="u1",
                tenant_id="t1",
                read_context_ref=owner,
            ),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    _set_work_item_read_context_ref(root, result.run_id, worker_context)

    state = bundle.questions.get_question_state(
        result.question_id,
        principal=QuestionPrincipal(
            principal_id="u1",
            tenant_id="t1",
            read_context_ref=owner,
        ),
    )

    assert state is not None
    assert state["questionId"] == result.question_id


def test_sql_storage_rejects_cross_subject_conversation_reuse(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    owner = _principal("owner", "owner")
    other = _principal("other", "other")

    owner_result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=owner,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="shared-conversation",
        )
    )

    with pytest.raises(PermissionError, match="conversation"):
        bundle.questions.ask(
            AskRequest(
                question="How many payments came in today?",
                principal=other,
                execution_mode=ExecutionMode.QUEUED,
                conversation_id="shared-conversation",
            )
        )
    assert (
        bundle.questions.get_question_state(
            owner_result.question_id,
            principal=other,
        )
        is None
    )


def test_sql_storage_idempotency_replay_is_scoped_to_read_context(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)

    owner_result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=_principal("owner", "owner"),
            execution_mode=ExecutionMode.QUEUED,
            idempotency_key="shared-idempotency-key",
        )
    )

    other_result = bundle.questions.ask(
        AskRequest(
            question="How many payments came in today?",
            principal=_principal("other", "other"),
            execution_mode=ExecutionMode.QUEUED,
            idempotency_key="shared-idempotency-key",
        )
    )

    assert other_result.run_id != owner_result.run_id
    assert other_result.question_id != owner_result.question_id
    assert other_result.conversation_id != owner_result.conversation_id


def test_sql_storage_clarification_continuation_authorizes_trigger_run(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    owner_result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=_principal("owner", "owner"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="owner-conversation",
        )
    )
    recorder = LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    recorder.record_step(
        RunStepWrite(
            step_id="owner-clarification-step",
            run_id=owner_result.run_id,
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
            input_summary_json={},
            output_summary_json={},
            error_json={},
        )
    )
    recorder.record_clarification_request(
        ClarificationRequestWrite(
            clarification_id="owner-clarification",
            run_id=owner_result.run_id,
            step_id="owner-clarification-step",
            basis=ClarificationBasis.MULTIPLE_MATCHING_ENTITIES,
            question_text="Which store do you mean?",
        )
    )
    other_result = bundle.questions.ask(
        AskRequest(
            question="How many payments came in today?",
            principal=_principal("other", "other"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="other-conversation",
        )
    )
    SQLWorkItemQueue(bundle.engine).mark_work_item_terminal(
        run_id=other_result.run_id,
        status="FAILED",
        error="setup_terminal",
    )

    with pytest.raises(
        LineageRecorderConflict,
        match="clarification trigger run must belong to the same question",
    ):
        bundle.questions.continue_question(
            ContinueQuestionRequest(
                question_id=other_result.question_id,
                question="ABC Mall",
                principal=_principal("other", "other"),
                trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
                execution_mode=ExecutionMode.QUEUED,
                trigger_clarification_response_run_id=owner_result.run_id,
                trigger_clarification_response_id="owner-clarification",
            )
        )

    assert _clarification_response_count(root) == 0


def test_sql_storage_records_model_call_audit_with_empty_subcalls(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    ask = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    recorder = LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    recorder.record_step(
        RunStepWrite(
            step_id="step_model_turn",
            run_id=ask.run_id,
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
            input_summary_json={},
            output_summary_json={},
            error_json={},
        )
    )

    recorder.record_model_call_audit(
        ModelCallAuditWrite(
            model_call=ModelCallWrite(
                model_call_id="model_call_1",
                run_id=ask.run_id,
                step_id="step_model_turn",
                call_index=1,
                provider="openai",
                model_key="openai:gpt-5.4-mini",
                status=ModelCallStatus.FAILED,
                duration_ms=1,
                prompt_chars=10,
                schema_chars=20,
                tool_spec_chars=30,
                submitted_payload_chars=0,
            )
        )
    )

    with bundle.engine.connect() as connection:
        row_count = connection.execute(
            sa.text("select count(*) from fervis_model_call")
        ).scalar_one()
    assert row_count == 1


def test_sql_lineage_recorder_treats_json_normalization_as_idempotent(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    ask = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    recorder = LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    recorder.record_step(
        RunStepWrite(
            step_id="step_contract",
            run_id=ask.run_id,
            sequence=1,
            step_key=RunStepKey.QUESTION_CONTRACT,
            kind=RunStepKind.MODEL_TURN,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_compile",
            run_id=ask.run_id,
            sequence=2,
            step_key=RunStepKey.COMPILE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_step(
        RunStepWrite(
            step_id="step_execute",
            run_id=ask.run_id,
            sequence=3,
            step_key=RunStepKey.EXECUTE,
            kind=RunStepKind.DETERMINISTIC,
        )
    )
    recorder.record_requested_fact(
        RequestedFactWrite(
            requested_fact_id="fact_1",
            run_id=ask.run_id,
            produced_by_step_id="step_contract",
            fact_key="fact_1",
            description="order count",
            answer_expression_family="scalar_aggregate",
            requested_fact_json={"description": "order count"},
            answer_requests_json={"outputs": ("answer_1",)},
        )
    )
    recorder.record_fact_result(
        FactResultWrite(
            fact_result_id="fact_result_1",
            run_id=ask.run_id,
            requested_fact_id="fact_1",
            produced_by_step_id="step_execute",
            result_kind=FactResultKind.ANSWERED,
        )
    )
    proof_graph = ExecutionProofGraphWrite(
        proof_graph_id="proof_1",
        run_id=ask.run_id,
        fact_result_id="fact_result_1",
        compile_step_id="step_compile",
        execute_step_id="step_execute",
        payload_schema="fervis.execution_proof_graph",
        payload_schema_rev=1,
        payload_json={
            "nodes": [
                {
                    "id": "answer_1",
                    "kind": "answer_output",
                    "proof_refs": [],
                    "debug_terms": ("orders", "today"),
                }
            ],
            "edges": [],
            "contributions": [],
        },
    )

    recorder.record_execution_proof_graph(proof_graph)
    recorder.record_execution_proof_graph(proof_graph)


def test_sql_storage_idempotency_and_active_conflict(tmp_path: Path) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    first = AskRequest(
        question="How many orders came in today?",
        principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
        execution_mode=ExecutionMode.QUEUED,
        conversation_id="c1",
        idempotency_key="orders-today",
    )

    first_result = bundle.questions.ask(first)
    second_result = bundle.questions.ask(first)
    conflict = bundle.questions.ask(
        AskRequest(
            question="How many payments came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )

    assert second_result.run_id == first_result.run_id
    assert conflict.status == "ACTIVE_RUN_CONFLICT"
    assert conflict.active_run_id == first_result.run_id
    assert _count_rows(root, "fervis_run_work_item") == 1


def test_sql_storage_worker_execute_terminalizes_leased_run(tmp_path: Path) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root, lookup=_FailingLookup())
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    [item] = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )

    executed = bundle.run_work.process_queued_run(
        QueuedRunRequest(
            run_id=result.run_id,
            worker_id="worker-1",
            active_attempt=item.active_attempt,
        )
    )

    assert executed.status == "FAILED"
    assert executed.error == "lookup_failed"
    assert _work_item_status(root, result.run_id) == "FAILED"
    assert _count_rows(root, "fervis_run_result") == 1
    assert _count_rows(root, "fervis_runtime_error_detail") == 1


def test_sql_storage_worker_execute_returns_lookup_payload(tmp_path: Path) -> None:
    root = _migrated_fastapi_project(tmp_path)
    lookup = _CompletedLookup()
    bundle = _storage_bundle(root, lookup=lookup)
    lookup.terminal_answer_writer = make_terminal_answer_writer(
        LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    )
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    [item] = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )

    executed = bundle.run_work.process_queued_run(
        QueuedRunRequest(
            run_id=result.run_id,
            worker_id="worker-1",
            active_attempt=item.active_attempt,
        )
    )

    assert executed.status == "COMPLETED"
    assert executed.answer == "42 orders"
    assert executed.result_data == {"value": 42}
    terminal = terminal_result_for_run(bundle.engine, result.run_id)
    assert terminal is not None
    assert terminal.status == "COMPLETED"
    assert terminal.answer == "42 orders"


def test_sql_storage_worker_load_reconciles_existing_terminal_lineage(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root, lookup=_FailingLookup())
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    [item] = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )
    record_runtime_error_result(
        engine=bundle.engine,
        run_id=result.run_id,
        error_code="already_terminal",
    )

    executed = bundle.run_work.process_queued_run(
        QueuedRunRequest(
            run_id=result.run_id,
            worker_id="worker-1",
            active_attempt=item.active_attempt,
        )
    )

    assert executed.status == "FAILED"
    assert executed.error == "already_terminal"
    assert _work_item_status(root, result.run_id) == "FAILED"
    assert _work_item_lease_owner(root, result.run_id) == "None"
    assert _work_item_last_error(root, result.run_id) == "already_terminal"


def test_sql_storage_claim_reconciles_terminal_lineage_before_retrying(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    [item] = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )
    assert item.active_attempt == 1
    record_runtime_error_result(
        engine=bundle.engine,
        run_id=result.run_id,
        error_code="already_terminal",
    )
    _expire_run_below_max_attempts(root, result.run_id)

    claimed = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-2",
        batch_size=1,
        lease_seconds=60,
    )

    assert claimed == []
    assert _work_item_status(root, result.run_id) == "FAILED"
    assert _work_item_lease_owner(root, result.run_id) == "None"
    assert _work_item_last_error(root, result.run_id) == "already_terminal"


def test_sql_storage_expired_run_at_max_attempts_fails_queue_and_lineage(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    _expire_run_at_max_attempts(root, result.run_id)

    claimed = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )

    assert claimed == []
    assert _work_item_status(root, result.run_id) == "FAILED"
    assert _count_rows(root, "fervis_run_result") == 1
    assert _runtime_error_message(root, result.run_id) == (
        "run_max_attempts_exceeded"
    )


def test_sql_storage_claim_predicate_rejects_stale_candidate_snapshot(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    queue = SQLWorkItemQueue(bundle.engine)
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    table = metadata.tables["fervis_run_work_item"]
    stale_snapshot_filter = queue._claim_filter_for_candidate(
        run_id=result.run_id,
        attempt_count=0,
        now=now_utc(),
    )

    with bundle.engine.begin() as connection:
        connection.execute(
            sa.update(table)
            .where(table.c.run_id == result.run_id)
            .values(
                status="RUNNING",
                attempt_count=1,
                active_attempt=1,
                lease_owner="worker-a",
                lease_expires_at=now_utc() + timedelta(minutes=5),
            )
        )
        claim = connection.execute(
            sa.update(table).where(stale_snapshot_filter).values(lease_owner="worker-b")
        )

    assert claim.rowcount == 0
    assert _work_item_lease_owner(root, result.run_id) == "worker-a"


def test_sql_storage_max_attempt_cleanup_preserves_existing_terminal_lineage(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    record_runtime_error_result(
        engine=bundle.engine,
        run_id=result.run_id,
        error_code="existing_terminal_error",
    )
    _expire_run_at_max_attempts(root, result.run_id)

    claimed = SQLWorkItemQueue(bundle.engine).claim_run_work_items(
        worker_id="worker-1",
        batch_size=1,
        lease_seconds=60,
    )

    assert claimed == []
    assert _work_item_status(root, result.run_id) == "FAILED"
    assert _work_item_last_error(root, result.run_id) == "existing_terminal_error"
    assert _runtime_error_message(root, result.run_id) == "existing_terminal_error"


def test_sql_storage_active_conflict_ignores_rows_with_terminal_lineage(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    first = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    record_runtime_error_result(
        engine=bundle.engine,
        run_id=first.run_id,
        error_code="already_terminal",
    )

    second = bundle.questions.ask(
        AskRequest(
            question="How many payments came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )

    assert second.status == "QUEUED"
    assert second.run_id != first.run_id
    assert _work_item_status(root, first.run_id) == "FAILED"
    assert _work_item_last_error(root, first.run_id) == "already_terminal"


def test_sql_storage_terminal_result_reads_active_transaction(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )

    with rollback_sql_transaction(bundle.engine):
        record_runtime_error_result(
            engine=bundle.engine,
            run_id=result.run_id,
            error_code="transaction_visible_error",
        )
        terminal = terminal_result_for_run(bundle.engine, result.run_id)

    assert terminal is not None
    assert terminal.status == "FAILED"
    assert terminal.error == "transaction_visible_error"
    assert _count_rows(root, "fervis_run_result") == 0


def test_fervis_runtime_ask_queued_uses_sqlite_storage_and_explain_reads_it(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)

    ask = _run_fervis(
        root,
        "runtime",
        "ask",
        "How many orders came in today?",
        "--tenant-id",
        "t1",
        "--principal-id",
        "u1",
        "--conversation-id",
        "c1",
    )
    ask_events = [json.loads(line) for line in ask.stdout.splitlines() if line.strip()]
    accepted = ask_events[0]
    queued = ask_events[-1]
    run_id = str(accepted["run_id"])
    question_id = str(accepted["question_id"])
    explain = _run_fervis(root, "explain", "--run-id", run_id)

    assert ask.returncode == 0
    assert accepted["event"] == "run.accepted"
    assert accepted["status"] == "QUEUED"
    assert queued == {
        "event": "run.queued",
        "conversation_id": "c1",
        "next_actions": [inspect_question_action(question_id)],
        "question_id": question_id,
        "run_id": run_id,
        "status": "QUEUED",
    }
    assert _count_rows(root, "fervis_run_work_item") == 1
    assert explain.returncode == 0
    explain_payload = json.loads(explain.stdout)
    assert explain_payload["command"] == "explain"
    assert explain_payload["payload"]["root"] == {"kind": "run", "id": run_id}


def test_fervis_runtime_ask_without_wait_does_not_execute_configured_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    _write_order_count_fastapi_app(root)
    lookup_calls = []

    def fake_run_lookup_question(request, ports):
        del request, ports
        lookup_calls.append("called")
        raise AssertionError("non-wait runtime ask must not execute lookup")

    monkeypatch.setattr(
        "fervis.lookup.orchestration.pipeline.run_lookup_question",
        fake_run_lookup_question,
    )
    exit_code, events = _run_default_runtime_ask(root)
    run_id = str(events[0]["run_id"])
    question_id = str(events[0]["question_id"])

    assert exit_code == 0
    assert lookup_calls == []
    assert events[-1] == {
        "event": "run.queued",
        "conversation_id": "c1",
        "next_actions": [inspect_question_action(question_id)],
        "question_id": question_id,
        "run_id": run_id,
        "status": "QUEUED",
    }


def test_fervis_runtime_ask_wait_executes_sqlite_queued_run_to_terminal(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    lookup = _CompletedLookup()
    bundle = _storage_bundle(root, lookup=lookup)
    lookup.terminal_answer_writer = make_terminal_answer_writer(
        LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    )
    stdout = StringIO()

    exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "t1",
            "--principal-id",
            "u1",
            "--conversation-id",
            "c1",
            "--wait",
            "60",
        ),
        ports=FervisCliPorts(
            lineage_query=bundle.lineage_query,
            observability_query=bundle.observability_query,
            prompt_capture_query=bundle.prompt_capture_query,
            questions=bundle.questions,
            question_run_follower=LocalQueuedRunFollower(
                run_work=bundle.run_work,
                work_queue=SQLWorkItemQueue(bundle.engine),
                worker_id="cli-test-worker",
                lease_seconds=60,
            ),
            project=discover_project(root),
            model_policy=_model_policy(root),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    run_id = str(events[0]["run_id"])

    assert exit_code == 0
    assert events == [
        {
            "event": "run.accepted",
            "conversation_id": "c1",
            "question_id": events[0]["question_id"],
            "run_id": run_id,
            "status": "QUEUED",
        },
        {
            "event": "run.progress",
            "message": "starting lookup",
            "run_id": run_id,
            "stage": "lookup",
        },
        {
            "answer": "42 orders",
            "conversation_id": "c1",
            "event": "run.completed",
            "next_actions": [inspect_question_action(events[0]["question_id"])],
            "question_id": events[0]["question_id"],
            "result_data": {"value": 42},
            "run_id": run_id,
            "status": "COMPLETED",
        },
    ]
    assert _work_item_status(root, run_id) == "COMPLETED"
    terminal = terminal_result_for_run(bundle.engine, run_id)
    assert terminal is not None
    assert terminal.answer == "42 orders"


def test_fervis_worker_once_executes_sqlite_queued_runtime_ask_to_terminal(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    lookup = _CompletedLookup()
    bundle = _storage_bundle(root, lookup=lookup)
    lookup.terminal_answer_writer = make_terminal_answer_writer(
        LineageRecorder(SQLLineageRecorderStore(bundle.engine))
    )
    ask_stdout = StringIO()
    worker_stdout = StringIO()

    ask_exit_code = run_fervis(
        (
            "runtime",
            "ask",
            "How many orders came in today?",
            "--tenant-id",
            "t1",
            "--principal-id",
            "u1",
            "--conversation-id",
            "c1",
        ),
        ports=FervisCliPorts(
            lineage_query=bundle.lineage_query,
            observability_query=bundle.observability_query,
            prompt_capture_query=bundle.prompt_capture_query,
            questions=bundle.questions,
            project=discover_project(root),
            model_policy=_model_policy(root),
        ),
        stdout=ask_stdout,
        stderr=StringIO(),
    )
    ask_events = [
        json.loads(line) for line in ask_stdout.getvalue().splitlines() if line.strip()
    ]
    run_id = str(ask_events[0]["run_id"])

    worker_exit_code = run_fervis(
        (
            "worker",
            "--once",
            "--worker-id",
            "worker-1",
            "--lease-seconds",
            "60",
        ),
        ports=FervisCliPorts(
            lineage_query=bundle.lineage_query,
            observability_query=bundle.observability_query,
            prompt_capture_query=bundle.prompt_capture_query,
            questions=bundle.questions,
            project=discover_project(root),
            run_worker=RunWorkBatchProcessor(
                worker=RunWorkServiceWorker(bundle.run_work),
                work_queue=SQLWorkItemQueue(bundle.engine),
            ),
            model_policy=_model_policy(root),
        ),
        stdout=worker_stdout,
        stderr=StringIO(),
    )

    worker_envelope = json.loads(worker_stdout.getvalue())
    assert ask_exit_code == 0
    assert worker_exit_code == 0
    assert worker_envelope["command"] == "worker"
    assert worker_envelope["payload_schema"] == "fervis-worker-cycle.v0.1"
    assert worker_envelope["payload"]["claimed_count"] == 1
    assert worker_envelope["payload"]["completed_count"] == 1
    assert worker_envelope["payload"]["failed_count"] == 0
    assert worker_envelope["payload"]["runs"] == [
        {
            "run_id": run_id,
            "active_attempt": 1,
            "status": "COMPLETED",
            "error": None,
        }
    ]
    assert _work_item_status(root, run_id) == "COMPLETED"
    terminal = terminal_result_for_run(bundle.engine, run_id)
    assert terminal is not None
    assert terminal.answer == "42 orders"


def test_fervis_runtime_ask_wait_executes_configured_fastapi_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    _write_order_count_fastapi_app(root)
    _configure_fastapi_auth(root)
    calls: list[dict[str, object]] = []

    def fake_run_lookup_question(request, ports):
        catalog = ports.relation_catalog_port.build_relation_catalog()
        result = ports.data_access_port.read(
            endpoint_name="get_order_count",
            args={"get_order_count.query.store": "ABC"},
        )
        calls.append(
            {
                "host": request.host.organization_name,
                "catalog": [read.endpoint_name for read in catalog.reads],
                "endpoint": result["endpointName"],
                "body": result["responseBody"],
                "provider": request.provider_preferences["provider"],
                "model_key": request.provider_preferences["modelKey"],
            }
        )
        _write_terminal_answer_from_lookup_request(
            root,
            request,
            "42 orders",
            {"value": 42},
        )
        return LookupResult(
            status="COMPLETED",
            answer="42 orders",
            result_data={"value": 42},
        )

    monkeypatch.setattr(
        "fervis.lookup.orchestration.pipeline.run_lookup_question",
        fake_run_lookup_question,
    )
    exit_code, events = _run_default_runtime_ask_wait(root)

    run_id = str(events[0]["run_id"])
    assert events[-1]["event"] == "run.completed"
    assert calls == [
        {
            "host": "Test Shop",
            "catalog": ["get_order_count"],
            "endpoint": "get_order_count",
            "body": {"count": 42},
            "provider": "openai",
            "model_key": "openai:gpt-5.4-mini",
        }
    ]
    assert exit_code == 0
    assert events[-1] == {
        "answer": "42 orders",
        "conversation_id": "c1",
        "event": "run.completed",
        "next_actions": [inspect_question_action(events[0]["question_id"])],
        "question_id": events[0]["question_id"],
        "result_data": {"value": 42},
        "run_id": run_id,
        "status": "COMPLETED",
    }
    assert _work_item_status(root, run_id) == "COMPLETED"


def test_fervis_runtime_ask_wait_does_not_emit_queued_only_lookup_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    _write_order_count_fastapi_app(root)
    _configure_fastapi_auth(root)

    def fake_run_lookup_question(request, ports):
        del ports
        _write_terminal_answer_from_lookup_request(
            root,
            request,
            "42 orders",
            {"value": 42},
        )
        return LookupResult(
            status="COMPLETED",
            answer="42 orders",
            result_data={"value": 42},
        )

    monkeypatch.setattr(
        "fervis.lookup.orchestration.pipeline.run_lookup_question",
        fake_run_lookup_question,
    )
    exit_code, events = _run_default_runtime_ask_wait(root)

    assert all(event.get("event") != "run.failed" for event in events)
    assert "Inline lookup execution is not configured" not in json.dumps(events)
    assert exit_code == 0


def test_local_sql_follower_emits_existing_terminal_lineage(tmp_path: Path) -> None:
    root = _migrated_fastapi_project(tmp_path)
    bundle = _storage_bundle(root)
    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )
    record_runtime_error_result(
        engine=bundle.engine,
        run_id=result.run_id,
        error_code="already_terminal",
    )
    sink = CollectingQuestionRunEventSink()

    followed = LocalQueuedRunFollower(
        run_work=bundle.run_work,
        work_queue=SQLWorkItemQueue(bundle.engine),
        worker_id="cli-test-worker",
        lease_seconds=60,
    ).follow(result, event_sink=sink, wait_seconds=1)

    assert followed.status == "FAILED"
    assert followed.error == "already_terminal"
    assert sink.events == [
        {
            "event": "run.failed",
            "run_id": result.run_id,
            "question_id": result.question_id,
            "conversation_id": result.conversation_id,
            "status": "FAILED",
            "error": {
                "code": "already_terminal",
                "message": "already_terminal",
                "retryable": False,
            },
        }
    ]


def test_fervis_doctor_adds_storage_writability_checks_after_migrate(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    _configure_fastapi_auth(root)

    doctor = _run_fervis(root, "doctor", "--probe-read-context-key", "user_7")

    assert doctor.returncode == 0
    checks = {
        check["id"]: check for check in json.loads(doctor.stdout)["payload"]["checks"]
    }
    assert checks["persistence.lineage_writable"]["status"] == "passed"
    assert checks["persistence.queue_writable"]["status"] == "passed"
    assert checks["persistence.question_run_dry_run"]["status"] == "passed"
    assert _count_rows(root, "fervis_run_work_item") == 0


def test_database_url_persistence_uses_sql_storage_adapters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    database_path = root / "runtime.sqlite3"
    _replace_sqlite_config_with_database_url(root)
    monkeypatch.setenv("FERVIS_DATABASE_URL", f"sqlite:///{database_path}")
    run_migrate_command(("migrate",), project=discover_project(root), stdout=StringIO())
    bundle = _storage_bundle(root)

    result = bundle.questions.ask(
        AskRequest(
            question="How many orders came in today?",
            principal=QuestionPrincipal(principal_id="u1", tenant_id="t1"),
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="c1",
        )
    )

    assert result.status == "QUEUED"
    assert database_path.is_file()
    assert bundle.kind == "database_url"


def test_sql_read_context_migration_adds_read_context_column(
    tmp_path: Path,
) -> None:
    import importlib

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from fervis.project.persistence.schema_snapshots import v0001

    database_path = tmp_path / "fervis.sqlite3"
    engine = create_sqlite_engine(f"sqlite:///{database_path}")
    v0001.metadata.create_all(engine)
    revision = importlib.import_module(
        "fervis.project.persistence_migrations.versions.0002_read_context_ref"
    )
    with engine.begin() as connection:
        original_op = revision.op
        revision.op = Operations(MigrationContext.configure(connection))
        try:
            revision.upgrade()
        finally:
            revision.op = original_op
        columns = {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(fervis_run_work_item)"
            )
        }

    assert "read_context_ref" in columns
    assert "subject_ref" not in columns


class _FailingLookup(QuestionLookupPort):
    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del request, progress_sink
        return LookupExecutionResult(
            status="FAILED",
            error="lookup_failed",
            terminal_lineage_recorded=False,
        )


class _CapturingLookup(QuestionLookupPort):
    def __init__(self) -> None:
        self.requests: list[LookupExecutionRequest] = []

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del progress_sink
        self.requests.append(request)
        return LookupExecutionResult(
            status="FAILED",
            error="lookup_failed",
            terminal_lineage_recorded=False,
        )


class _CompletedLookup(QuestionLookupPort):
    terminal_answer_writer: TerminalAnswerWriter | None = None

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del progress_sink
        if self.terminal_answer_writer is None:
            raise RuntimeError("terminal answer writer is not configured")
        self.terminal_answer_writer(request, "42 orders", {"value": 42})
        return LookupExecutionResult(
            status="COMPLETED",
            answer="42 orders",
            result_data={"value": 42},
            terminal_lineage_recorded=True,
        )


def _storage_bundle(root: Path, lookup: QuestionLookupPort | None = None):
    loaded = load_fervis_project_config(discover_project(root))
    return sql_storage_bundle(
        project=discover_project(root),
        loaded_config=loaded,
        lookup=lookup,
    )


def _principal(principal_id: str, read_context_key: str) -> QuestionPrincipal:
    return QuestionPrincipal(
        principal_id=principal_id,
        tenant_id="t1",
        read_context_ref=ReadContextRef(
            scheme="fastapi_principal",
            key=read_context_key,
        ),
    )


def _model_policy(root: Path) -> ConfiguredModelPolicy:
    loaded = load_fervis_project_config(discover_project(root))
    assert hasattr(loaded, "config")
    return ConfiguredModelPolicy.from_config(loaded.config.model)


def _migrated_fastapi_project(tmp_path: Path) -> Path:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(("migrate",), project=discover_project(root), stdout=StringIO())
    return root


def _configure_fastapi_auth(root: Path) -> None:
    _write_fastapi_auth_helpers(root)
    run_auth_command(
        (
            "auth",
            "configure",
            "--principal-dependency",
            "app.api.deps:get_current_user",
            "--principal-id-attr",
            "id",
            "--principal-resolver",
            "app.users:get_user_by_id",
            "--transport-mode",
            "in_process",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )


def _write_fastapi_auth_helpers(root: Path) -> None:
    app_dir = root / "app"
    api_dir = app_dir / "api"
    api_dir.mkdir(exist_ok=True)
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "deps.py").write_text(
        "from app.users import User\n\n"
        "def get_current_user():\n"
        "    return User('anonymous')\n",
        encoding="utf-8",
    )
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    del tenant_id\n"
        "    return User(user_id)\n",
        encoding="utf-8",
    )


def _fastapi_project(tmp_path: Path) -> Path:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    app_dir = root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n\n"
        "class HealthResponse(BaseModel):\n"
        "    status: str\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/', response_model=HealthResponse)\n"
        "def get_health() -> HealthResponse:\n"
        "    return HealthResponse(status='ok')\n",
        encoding="utf-8",
    )
    return root


def _write_order_count_fastapi_app(root: Path) -> None:
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n\n"
        "from fervis import configured_fervis\n\n\n"
        "class OrderCountResponse(BaseModel):\n"
        "    count: int\n\n\n"
        "app = FastAPI()\n\n\n"
        "@app.get(\n"
        "    '/orders/count/',\n"
        "    operation_id='get_order_count',\n"
        "    tags=['orders'],\n"
        "    response_model=OrderCountResponse,\n"
        ")\n"
        "def get_order_count(store: str) -> OrderCountResponse:\n"
        "    return OrderCountResponse(count=42)\n\n\n"
        "@app.get(\n"
        "    '/internal/order-count/',\n"
        "    operation_id='internal_order_count',\n"
        "    tags=['orders'],\n"
        "    response_model=OrderCountResponse,\n"
        ")\n"
        "def internal_order_count() -> OrderCountResponse:\n"
        "    return OrderCountResponse(count=999)\n\n\n"
        "configured_fervis().mount(app)\n",
        encoding="utf-8",
    )
    schema = _read_project_schema(root)
    schema["host"] = {
        "organization_name": "Test Shop",
        "about_api": "The Test Shop API exposes order records.",
    }
    schema["models"] = {
        "providers": [
            {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
        ],
    }
    schema["environments"][schema["default_environment"]]["models"] = {
        "default": {"provider": "openai", "model_key": "gpt-5.4-mini"}
    }
    schema["sources"][0]["path_prefixes"] = ["/orders/count/"]
    _write_project_schema(root, schema)


def _write_terminal_answer_from_lookup_request(
    root: Path,
    request: LookupRequest,
    answer: str,
    result_data: dict[str, object],
) -> None:
    engine = create_sqlite_engine(f"sqlite:///{root / '.fervis' / 'fervis.sqlite3'}")
    writer = make_terminal_answer_writer(
        LineageRecorder(SQLLineageRecorderStore(engine))
    )
    writer(
        LookupExecutionRequest(
            run_id=request.run_id,
            conversation_id=str(request.user_context.get("conversationId") or ""),
            tenant_id=request.tenant_id,
            question=request.question,
            read_context_ref=ReadContextRef(scheme="anonymous"),
            principal=None,
            provider=str(request.provider_preferences.get("provider") or ""),
            model_key=str(request.provider_preferences.get("modelKey") or ""),
            conversation_context=dict(request.conversation_context),
            runtime_context=dict(request.user_context),
            max_budget_usd=None,
            max_thinking_tokens=request.max_thinking_tokens,
            active_attempt=request.active_attempt,
        ),
        answer,
        result_data,
    )


def _run_default_runtime_ask_wait(root: Path) -> tuple[int, list[dict[str, object]]]:
    return _run_default_runtime_ask(root, wait=True)


def _run_default_runtime_ask(
    root: Path,
    *,
    wait: bool = False,
) -> tuple[int, list[dict[str, object]]]:
    bundle = _storage_bundle(root)
    stdout = StringIO()
    args = [
        "runtime",
        "ask",
        "How many orders came in today?",
        "--tenant-id",
        "t1",
        "--principal-id",
        "u1",
        "--conversation-id",
        "c1",
    ]
    if wait:
        args.extend(("--wait", "60"))
    exit_code = run_fervis(
        tuple(args),
        ports=FervisCliPorts(
            lineage_query=bundle.lineage_query,
            observability_query=bundle.observability_query,
            prompt_capture_query=bundle.prompt_capture_query,
            questions=bundle.questions,
            question_run_follower=LocalQueuedRunFollower(
                run_work=bundle.run_work,
                work_queue=SQLWorkItemQueue(bundle.engine),
                worker_id="cli-test-worker",
                lease_seconds=60,
            ),
            project=discover_project(root),
            model_policy=_model_policy(root),
        ),
        stdout=stdout,
        stderr=StringIO(),
    )
    return exit_code, [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]


def _replace_sqlite_config_with_database_url(root: Path) -> None:
    schema = _read_project_schema(root)
    schema["environments"][schema["default_environment"]]["persistence"] = {
        "kind": "database_url",
        "url_env": "FERVIS_DATABASE_URL",
    }
    _write_project_schema(root, schema)


def _read_project_schema(root: Path) -> dict[str, object]:
    return json.loads((root / "config" / "fervis.json").read_text(encoding="utf-8"))


def _write_project_schema(root: Path, schema: dict[str, object]) -> None:
    (root / "config" / "fervis.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_fervis(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fervis.interfaces.cli.main", *args],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(API_DIR),
            "FERVIS_INVOCATION_CWD": str(root),
        },
        check=False,
        capture_output=True,
        text=True,
    )


def _count_rows(root: Path, table: str) -> int:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _work_item_status(root: Path, run_id: str) -> str:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT status FROM fervis_run_work_item WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return str(row[0])


def _work_item_read_context_ref(root: Path, run_id: str) -> dict[str, object]:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT read_context_ref FROM fervis_run_work_item WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return json.loads(str(row[0]))


def _work_item_runtime_context(root: Path, run_id: str) -> dict[str, object]:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT runtime_context FROM fervis_run_work_item WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return json.loads(str(row[0]))


def _conversation_read_context_ref(
    root: Path,
    conversation_id: str,
) -> dict[str, object]:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT read_context_ref FROM fervis_conversation WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return json.loads(str(row[0]))


def _set_work_item_read_context_ref(
    root: Path,
    run_id: str,
    read_context_ref: ReadContextRef,
) -> None:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        connection.execute(
            "UPDATE fervis_run_work_item SET read_context_ref = ? WHERE run_id = ?",
            (json.dumps(read_context_ref.to_storage_dict()), run_id),
        )


def _set_run_created_at(root: Path, run_id: str, created_at: str) -> None:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        connection.execute(
            "UPDATE fervis_question_run SET created_at = ? WHERE run_id = ?",
            (created_at, run_id),
        )


def _clarification_response_count(root: Path) -> int:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM fervis_clarification_response"
        ).fetchone()
    return int(row[0])


def _json_value(value) -> object:
    if isinstance(value, dict):
        return value
    text = str(value)
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_invalid_json": text}


def _work_item_last_error(root: Path, run_id: str) -> str:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT last_error FROM fervis_run_work_item WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return str(row[0])


def _work_item_lease_owner(root: Path, run_id: str) -> str:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT lease_owner FROM fervis_run_work_item WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return str(row[0])


def _expire_run_at_max_attempts(root: Path, run_id: str) -> None:
    expired = datetime.now(timezone.utc) - timedelta(minutes=1)
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        connection.execute(
            "UPDATE fervis_run_work_item "
            "SET status = 'RUNNING', lease_owner = 'worker-old', "
            "lease_expires_at = ?, attempt_count = 2, active_attempt = 2, "
            "max_attempts = 2 WHERE run_id = ?",
            (expired.strftime("%Y-%m-%d %H:%M:%S.%f"), run_id),
        )


def _expire_run_below_max_attempts(root: Path, run_id: str) -> None:
    expired = datetime.now(timezone.utc) - timedelta(minutes=1)
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        connection.execute(
            "UPDATE fervis_run_work_item "
            "SET status = 'RUNNING', lease_owner = 'worker-old', "
            "lease_expires_at = ?, attempt_count = 1, active_attempt = 1, "
            "max_attempts = 2 WHERE run_id = ?",
            (expired.strftime("%Y-%m-%d %H:%M:%S.%f"), run_id),
        )


def _runtime_error_message(root: Path, run_id: str) -> str:
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        row = connection.execute(
            "SELECT message FROM fervis_runtime_error_detail WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return str(row[0])
