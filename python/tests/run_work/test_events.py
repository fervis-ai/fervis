import pytest

from fervis.run_work.events import run_terminal_event, run_waiting_for_clarification_event


def test_run_terminal_event_preserves_question_and_conversation_handles():
    assert run_terminal_event(
        status="COMPLETED",
        run_id="run_1",
        question_id="question_1",
        conversation_id="conversation_1",
        answer="42",
        result_data={"value": 42},
    ) == {
        "event": "run.completed",
        "run_id": "run_1",
        "question_id": "question_1",
        "conversation_id": "conversation_1",
        "status": "COMPLETED",
        "answer": "42",
        "result_data": {"value": 42},
    }

    assert run_terminal_event(
        status="QUEUED",
        run_id="run_1",
        question_id="question_1",
        conversation_id="conversation_1",
    ) == {
        "event": "run.queued",
        "run_id": "run_1",
        "question_id": "question_1",
        "conversation_id": "conversation_1",
        "status": "QUEUED",
    }

    assert run_terminal_event(
        status="FAILED",
        run_id="run_1",
        question_id="question_1",
        conversation_id="conversation_1",
        error="provider_failed",
    ) == {
        "event": "run.failed",
        "run_id": "run_1",
        "question_id": "question_1",
        "conversation_id": "conversation_1",
        "status": "FAILED",
        "error": {
            "code": "provider_failed",
            "message": "provider_failed",
            "retryable": False,
        },
    }


def test_run_waiting_event_requires_actionable_clarifications():
    with pytest.raises(
        ValueError,
        match="clarification wait requires clarifications",
    ):
        run_waiting_for_clarification_event(
            run_id="run_1",
            question_id="question_1",
            conversation_id="conversation_1",
            result_data={},
        )

    with pytest.raises(
        ValueError,
        match="clarification wait requires clarification question",
    ):
        run_waiting_for_clarification_event(
            run_id="run_1",
            question_id="question_1",
            conversation_id="conversation_1",
            result_data={"details": {"clarifications": [{"id": "clarification_1"}]}},
        )

    assert run_waiting_for_clarification_event(
        run_id="run_1",
        question_id="question_1",
        conversation_id="conversation_1",
        result_data={
            "kind": "needs_clarification",
            "details": {
                "clarifications": [
                    {
                        "id": "clarification_1",
                        "question": "Which store should I use?",
                    }
                ]
            },
        },
    ) == {
        "event": "run.waiting_for_clarification",
        "conversation_id": "conversation_1",
        "question_id": "question_1",
        "run_id": "run_1",
        "status": "WAITING_FOR_CLARIFICATION",
        "clarifications": [
            {
                "id": "clarification_1",
                "question": "Which store should I use?",
            }
        ],
    }


@pytest.mark.parametrize("code,retryable", [
    ("provider_rate_limited", True), ("provider_timeout", True),
    ("provider_bad_request", False), ("provider_authentication_failed", False),
])
def test_persisted_provider_failure_retains_its_code_and_retry_policy(code, retryable):
    from types import SimpleNamespace
    from fervis.questions.run_views import _run_error
    from fervis.questions.contracts import AskResult
    from fervis.evaluation.goldsets.runner import _is_retryable_provider_failure
    run = SimpleNamespace(runtime_errors=(SimpleNamespace(error_kind="infrastructure_failed",
        message=code + ': {"provider_metadata": {"statusCode": "429"}}'),))
    work = SimpleNamespace(last_error=code)
    projected = _run_error(run, work)
    event = run_terminal_event(status="FAILED", run_id="run", error=projected)
    assert event["error"]["code"] == code
    assert event["error"]["retryable"] is retryable
    result = AskResult(status="FAILED", conversation_id="conversation", question_id="question", run_id="run", error=projected)
    assert _is_retryable_provider_failure(result) is retryable
