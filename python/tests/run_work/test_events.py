import pytest

from fervis.run_work.events import run_terminal_event


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


def test_run_terminal_event_requires_actionable_clarifications():
    with pytest.raises(
        ValueError,
        match="NEEDS_CLARIFICATION terminal event requires clarifications",
    ):
        run_terminal_event(
            status="NEEDS_CLARIFICATION",
            run_id="run_1",
            question_id="question_1",
            conversation_id="conversation_1",
            result_data={},
        )

    with pytest.raises(
        ValueError,
        match="NEEDS_CLARIFICATION terminal event requires clarification question",
    ):
        run_terminal_event(
            status="NEEDS_CLARIFICATION",
            run_id="run_1",
            question_id="question_1",
            conversation_id="conversation_1",
            result_data={"details": {"clarifications": [{"id": "clarification_1"}]}},
        )

    assert run_terminal_event(
        status="NEEDS_CLARIFICATION",
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
        "event": "run.needs_clarification",
        "conversation_id": "conversation_1",
        "question_id": "question_1",
        "run_id": "run_1",
        "status": "NEEDS_CLARIFICATION",
        "clarifications": [
            {
                "id": "clarification_1",
                "question": "Which store should I use?",
            }
        ],
    }
