from __future__ import annotations

import pytest

from fervis.questions import (
    AskRequest,
    ContinueQuestionRequest,
    ExecutionMode,
    QuestionPrincipal,
)
from fervis.lineage.enums import RunTriggerKind
from fervis.run_work import QueuedRunRequest

pytestmark = pytest.mark.django_db


def test_queued_ask_records_run_without_lookup(adapter):
    result = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.QUEUED)
    )

    assert result.status == "QUEUED"
    assert adapter.lookup.call_count() == 0
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 1
    assert adapter.probe.work_item_status(result.run_id) == "QUEUED"


def test_inline_lookup_exception_records_failed_terminal(adapter):
    adapter.lookup.raise_error(error="provider timeout")

    result = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.INLINE)
    )

    assert result.status == "FAILED"
    assert result.error == "provider timeout"
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "FAILED"
    assert adapter.probe.terminal_result(result.run_id)["error"] == "provider timeout"


def test_inline_terminal_lookup_returns_answer(adapter):
    adapter.lookup.complete_with_terminal(answer="42")

    result = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.INLINE)
    )

    assert result.status == "COMPLETED", result.error
    assert result.answer == "42"
    assert result.result_data == {"value": 42}
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "COMPLETED"
    assert adapter.probe.terminal_result(result.run_id) == {
        "status": "COMPLETED",
        "answer": "42",
        "result_data": {"value": 42},
        "error": None,
    }


def test_inline_failed_lookup_without_terminal_lineage_records_fallback(adapter):
    adapter.lookup.fail_result_without_terminal(error="fervis_failed")

    result = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.INLINE)
    )

    assert result.status == "FAILED"
    assert result.error == "fervis_failed"
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "FAILED"
    assert adapter.probe.terminal_result(result.run_id)["error"] == "fervis_failed"


def test_inline_missing_success_terminal_lineage_fails_closed(adapter):
    adapter.lookup.complete_without_terminal(answer="42")

    result = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.INLINE)
    )

    assert result.status == "FAILED"
    assert result.answer is None
    assert result.error == (
        f"lookup completed without terminal lineage for {result.run_id}: COMPLETED"
    )
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "FAILED"


def test_idempotent_inline_retry_returns_existing_terminal_without_duplicate_lineage(
    adapter,
):
    adapter.lookup.fail_result_without_terminal(error="fervis_failed")
    request = _ask(
        adapter,
        execution_mode=ExecutionMode.INLINE,
        idempotency_key="same-key",
    )

    first = adapter.questions.ask(request)
    second = adapter.questions.ask(request)

    assert first.status == "FAILED"
    assert second.status == "FAILED"
    assert second.run_id == first.run_id
    assert second.error == "fervis_failed"
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 1


def test_idempotent_inline_retry_returns_persisted_terminal_answer(adapter):
    adapter.lookup.complete_with_terminal(answer="42")
    request = _ask(
        adapter,
        execution_mode=ExecutionMode.INLINE,
        idempotency_key="same-key",
    )

    first = adapter.questions.ask(request)
    second = adapter.questions.ask(request)

    assert first.status == "COMPLETED", first.error
    assert second.status == "COMPLETED"
    assert second.run_id == first.run_id
    assert second.answer == "42"
    assert adapter.probe.terminal_result(second.run_id) == {
        "status": "COMPLETED",
        "answer": "42",
        "result_data": {"value": 42},
        "error": None,
    }
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 1


def test_idempotent_ask_replays_when_conversation_id_is_generated(adapter):
    first = adapter.questions.ask(
        _ask(
            adapter,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="",
            idempotency_key="generated-conversation-replay",
        )
    )
    repeated = adapter.questions.ask(
        _ask(
            adapter,
            question="Retry same submitted command.",
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="",
            idempotency_key="generated-conversation-replay",
        )
    )

    assert repeated.run_id == first.run_id
    assert repeated.question_id == first.question_id
    assert repeated.conversation_id == first.conversation_id
    assert adapter.probe.question_count(first.conversation_id) == 1
    assert adapter.probe.run_count(first.conversation_id) == 1


def test_active_run_conflict_does_not_duplicate_lineage(adapter):
    first = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.QUEUED)
    )
    conflict = adapter.questions.ask(
        _ask(
            adapter,
            question="How many payments came in today?",
            execution_mode=ExecutionMode.QUEUED,
        )
    )

    assert first.status == "QUEUED"
    assert conflict.status == "ACTIVE_RUN_CONFLICT"
    assert conflict.active_run_id == first.run_id
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 1


def test_active_conflict_ignores_terminal_lineage(adapter):
    first = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.QUEUED)
    )
    adapter.probe.record_terminal_error(first.run_id, error="already_terminal")

    second = adapter.questions.ask(
        _ask(
            adapter,
            question="How many payments came in today?",
            execution_mode=ExecutionMode.QUEUED,
        )
    )

    assert second.status == "QUEUED"
    assert second.run_id != first.run_id
    assert adapter.probe.work_item_status(first.run_id) == "FAILED"
    assert adapter.probe.work_item_status(second.run_id) == "QUEUED"
    assert adapter.probe.run_count("conversation_1") == 2


def test_question_continuation_creates_next_run_under_same_question(adapter):
    adapter.lookup.fail_result_without_terminal(error="first_failed")
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

    continued = adapter.questions.continue_question(
        ContinueQuestionRequest(
            question_id=first.question_id,
            question="Try again with the same question.",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            trigger_kind=RunTriggerKind.RETRY,
            execution_mode=ExecutionMode.QUEUED,
            previous_run_id=first.run_id,
            provider=adapter.provider,
            model_key=adapter.model_key,
        )
    )

    assert continued.status == "QUEUED"
    assert continued.question_id == first.question_id
    assert continued.run_id != first.run_id
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 2


def test_terminal_queued_run_short_circuits_without_lookup(adapter):
    queued = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.QUEUED)
    )
    adapter.probe.record_terminal_error(queued.run_id, error="already_terminal")

    result = adapter.run_work.process_queued_run(
        QueuedRunRequest(run_id=queued.run_id)
    )

    assert result.status == "FAILED"
    assert result.error == "already_terminal"
    assert adapter.lookup.call_count() == 0
    assert adapter.probe.work_item_status(queued.run_id) == "FAILED"


def test_terminal_lineage_reconciles_stale_active_work_item(adapter):
    queued = adapter.questions.ask(
        _ask(adapter, execution_mode=ExecutionMode.QUEUED)
    )
    claimed = adapter.work_items.claim_one(worker_id="worker_1")
    adapter.probe.record_terminal_error(queued.run_id, error="already_terminal")
    adapter.probe.expire_running_lease(queued.run_id)

    result = adapter.run_work.process_queued_run(
        QueuedRunRequest(
            run_id=queued.run_id,
            worker_id="worker_1",
            active_attempt=claimed.active_attempt,
        )
    )

    assert result.status == "FAILED"
    assert result.error == "already_terminal"
    assert adapter.lookup.call_count() == 0
    assert adapter.probe.work_item_status(queued.run_id) == "FAILED"


def _ask(
    adapter,
    *,
    question: str = "How many orders came in today?",
    execution_mode: ExecutionMode,
    conversation_id: str = "conversation_1",
    idempotency_key: str | None = None,
) -> AskRequest:
    return AskRequest(
        question=question,
        principal=QuestionPrincipal(
            principal_id=adapter.principal_id,
            tenant_id=adapter.tenant_id,
        ),
        execution_mode=execution_mode,
        conversation_id=conversation_id,
        provider=adapter.provider,
        model_key=adapter.model_key,
        idempotency_key=idempotency_key,
    )
