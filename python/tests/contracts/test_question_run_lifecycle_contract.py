from __future__ import annotations

import pytest

from fervis.host_api.contracts.authority import ReadAuthority
from fervis.interfaces.django.question_run_ports import DjangoQuestionStateReaderPort
from fervis.questions import (
    AskRequest,
    ClarificationResponseRequest,
    ExecutionMode,
    QuestionPrincipal,
    RetryQuestionRequest,
)
from fervis.run_work import QueuedRunRequest
from fervis.run_work.events import CollectingQuestionRunEventSink
from fervis.storage.sql.question_run_ports import SQLQuestionStateReaderPort

pytestmark = pytest.mark.django_db

_CANONICAL_ANSWER_RESULT = {
    "kind": "answer",
    "outputs": [
        {
            "key": "answer_1",
            "valueKind": "number",
            "value": {"value": "42"},
            "displayValue": "42",
        }
    ],
}

_DESKTOP_RUN_FIELDS = frozenset(
    {
        "runId",
        "questionId",
        "conversationId",
        "runNumber",
        "kind",
        "triggerKind",
        "baseRunId",
        "programId",
        "invocationId",
        "patchId",
        "revisionId",
        "status",
        "answer",
        "resultData",
        "explanation",
        "steps",
        "error",
        "worker",
        "usage",
    }
)


def test_queued_ask_records_run_without_lookup(adapter):
    result = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.QUEUED))

    assert result.status == "QUEUED"
    assert adapter.lookup.call_count() == 0
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 1
    assert adapter.probe.work_item_status(result.run_id) == "QUEUED"


def test_inline_lookup_exception_records_failed_terminal(adapter):
    adapter.lookup.raise_error(error="provider timeout")

    result = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

    assert result.status == "FAILED"
    assert result.error == "provider timeout"
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "FAILED"
    assert adapter.probe.terminal_result(result.run_id)["error"] == "provider timeout"


def test_inline_terminal_lookup_returns_answer(adapter):
    adapter.lookup.complete_with_terminal(answer="42")

    result = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

    assert result.status == "COMPLETED", result.error
    assert result.answer == "42"
    assert result.result_data == _CANONICAL_ANSWER_RESULT
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "COMPLETED"


def test_terminal_answer_projection_survives_reload(adapter):
    adapter.lookup.complete_with_terminal(answer="42")
    result = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

    persisted = _persisted_run(adapter, result.question_id, result.run_id)

    assert _DESKTOP_RUN_FIELDS <= persisted.keys()
    assert persisted["status"] == "COMPLETED"
    assert persisted["answer"] == "42"
    assert persisted["resultData"] == _CANONICAL_ANSWER_RESULT


def test_inline_failed_lookup_without_terminal_lineage_records_fallback(adapter):
    adapter.lookup.fail_result_without_terminal(error="fervis_failed")

    result = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

    assert result.status == "FAILED"
    assert result.error == "fervis_failed"
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.work_item_status(result.run_id) == "FAILED"
    assert adapter.probe.terminal_result(result.run_id)["error"] == "fervis_failed"


def test_inline_missing_success_terminal_lineage_fails_closed(adapter):
    adapter.lookup.complete_without_terminal(answer="42")

    result = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

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
    first_persisted = _persisted_run(adapter, first.question_id, first.run_id)
    second = adapter.questions.ask(request)
    second_persisted = _persisted_run(adapter, second.question_id, second.run_id)

    assert first.status == "COMPLETED", first.error
    assert second.status == "COMPLETED"
    assert second.run_id == first.run_id
    assert second.answer == "42"
    assert second.result_data == _CANONICAL_ANSWER_RESULT
    assert first.result_data == second.result_data
    assert _DESKTOP_RUN_FIELDS <= first_persisted.keys()
    assert _DESKTOP_RUN_FIELDS <= second_persisted.keys()
    assert _stable_run_fields(second_persisted) == _stable_run_fields(first_persisted)
    assert adapter.lookup.call_count() == 1
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 1


def test_idempotent_ask_returns_existing_when_conversation_id_is_generated(adapter):
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


def test_new_conversation_idempotency_rejects_different_payload(adapter):
    adapter.questions.ask(
        _ask(
            adapter,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="",
            idempotency_key="payload-bound-replay",
        )
    )

    with pytest.raises(
        ValueError,
        match="idempotency key was already used for a different request",
    ):
        adapter.questions.ask(
            _ask(
                adapter,
                question="How many payments came in today?",
                execution_mode=ExecutionMode.QUEUED,
                conversation_id="",
                idempotency_key="payload-bound-replay",
            )
        )


def test_new_conversation_idempotency_is_principal_scoped(adapter):
    first = adapter.questions.ask(
        _ask(
            adapter,
            execution_mode=ExecutionMode.QUEUED,
            conversation_id="",
            idempotency_key="principal-scoped-replay",
        )
    )
    second_request = _ask(
        adapter,
        execution_mode=ExecutionMode.QUEUED,
        conversation_id="",
        idempotency_key="principal-scoped-replay",
        principal_id="other-principal",
    )
    second = adapter.questions.ask(second_request)

    assert second.run_id != first.run_id
    replay = adapter.questions.ask(second_request)
    assert replay.run_id == second.run_id


def test_active_run_conflict_does_not_duplicate_lineage(adapter):
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.QUEUED))
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
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.QUEUED))
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

    continued = adapter.questions.retry_question(
        RetryQuestionRequest(
            question_id=first.question_id,
            question="Try again with the same question.",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.QUEUED,
            base_run_id=first.run_id,
            provider=adapter.provider,
            model_key=adapter.model_key,
        )
    )

    assert continued.status == "QUEUED"
    assert continued.question_id == first.question_id
    assert continued.run_id != first.run_id
    assert adapter.probe.question_count("conversation_1") == 1
    assert adapter.probe.run_count("conversation_1") == 2


def test_clarification_response_resumes_the_waiting_run(adapter):
    adapter.lookup.needs_clarification()
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))
    waiting = _persisted_run(adapter, first.question_id, first.run_id)

    adapter.lookup.complete_with_terminal(answer="42")
    resumed = adapter.questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=first.run_id,
            clarification_id="clarification_area",
            response_text="Nairobi",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.INLINE,
            selected_option_id="area_nairobi",
        )
    )

    assert first.status == "WAITING_FOR_CLARIFICATION"
    assert waiting["status"] == "WAITING_FOR_CLARIFICATION"
    assert waiting["resultData"]["details"]["clarifications"][0]["id"] == (
        "clarification_area"
    )
    assert resumed.status == "COMPLETED", resumed.error
    assert resumed.run_id == first.run_id
    assert resumed.question_id == first.question_id
    assert adapter.probe.run_count("conversation_1") == 1
    completed = _persisted_run(adapter, first.question_id, first.run_id)
    assert completed["status"] == "COMPLETED"
    assert completed["resultData"] == _CANONICAL_ANSWER_RESULT
    request = adapter.lookup.last_request()
    assert request is not None
    assert request.question == "How many orders came in today?"
    assert len(request.clarification_responses) == 1
    assert request.clarification_responses[0].option.id == "area_nairobi"


def test_grounding_prose_starts_a_cr_annotated_successor_run(adapter):
    clarification = {
        "id": "clarification_area",
        "need": "target_reference",
        "reason": "unresolved_reference",
        "owner": "grounding",
        "continuation": {
            "kind": "grounding",
            "knownInputId": "area_input",
            "acceptsFreeText": True,
        },
        "requestedFactId": "fact_1",
        "question": "Which matching location or area should I use?",
        "subjects": [
            {
                "kind": "question_input",
                "id": "area_input",
                "label": "area",
                "sourceText": "Nairobi",
                "options": [],
            }
        ],
        "evidence": [],
    }
    adapter.lookup.needs_clarification(payload=clarification)
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))

    adapter.lookup.complete_with_terminal(answer="28")
    continued = adapter.questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=first.run_id,
            clarification_id="clarification_area",
            response_text="Area: Nairobi",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.INLINE,
        )
    )

    assert continued.status == "COMPLETED", continued.error
    assert continued.run_id != first.run_id
    assert adapter.probe.work_item_status(first.run_id) == "SUPERSEDED"
    assert adapter.probe.run_count("conversation_1") == 2
    persisted = _persisted_run(adapter, first.question_id, continued.run_id)
    assert persisted["triggerKind"] == "clarification_response"
    assert persisted["baseRunId"] == first.run_id
    request = adapter.lookup.last_request()
    assert request is not None
    assert request.question == "Area: Nairobi"
    assert len(request.clarification_responses) == 1
    annotation = request.clarification_responses[0].annotation
    assert annotation is not None
    assert annotation.suspended_question_text == "How many orders came in today?"
    assert annotation.clarification_question_text == (
        'I could not find area "Nairobi". Which area should I use?'
    )


def test_consecutive_grounding_prose_preserves_the_clarification_chain(adapter):
    first_clarification = {
        "id": "clarification_area",
        "need": "target_reference",
        "reason": "unresolved_reference",
        "owner": "grounding",
        "continuation": {
            "kind": "grounding",
            "knownInputId": "area_input",
            "acceptsFreeText": True,
        },
        "requestedFactId": "fact_1",
        "question": "Which matching location or area should I use?",
        "subjects": [
            {
                "kind": "question_input",
                "id": "area_input",
                "label": "area",
                "sourceText": "Nairobi",
                "options": [],
            }
        ],
        "evidence": [],
    }
    second_clarification = {
        **first_clarification,
        "id": "clarification_location_type",
        "question": "Which kind of place should I use?",
    }
    adapter.lookup.needs_clarification(payload=first_clarification)
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))
    adapter.lookup.needs_clarification(payload=second_clarification)

    second = adapter.questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=first.run_id,
            clarification_id="clarification_area",
            response_text="Nairobi",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.INLINE,
        )
    )

    assert second.status == "WAITING_FOR_CLARIFICATION"
    adapter.lookup.complete_with_terminal(answer="28")
    third = adapter.questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=second.run_id,
            clarification_id="clarification_location_type",
            response_text="Area",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.INLINE,
        )
    )

    assert third.status == "COMPLETED", third.error
    assert adapter.probe.run_count("conversation_1") == 3
    request = adapter.lookup.last_request()
    assert request is not None
    assert request.question == "Area"
    annotations = tuple(
        response
        for response in request.clarification_responses
        if response.annotation is not None
    )
    assert tuple(
        (
            response.source.clarification_id,
            response.annotation.suspended_question_text,
            response.annotation.clarification_question_text,
        )
        for response in annotations
    ) == (
        (
            "clarification_area",
            "How many orders came in today?",
            'I could not find area "Nairobi". Which area should I use?',
        ),
        (
            "clarification_location_type",
            "Nairobi",
            'I could not find area "Nairobi". Which area should I use?',
        ),
    )


def test_clarification_cycles_expose_only_the_current_request(adapter):
    adapter.lookup.needs_clarification()
    first = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.INLINE))
    second_payload = {
        **adapter.lookup.clarification_payload,
        "id": "clarification_period",
        "question": "Which period should I use?",
    }
    adapter.lookup.needs_clarification(payload=second_payload)

    second_wait = adapter.questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=first.run_id,
            clarification_id="clarification_area",
            response_text="Nairobi",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.INLINE,
            selected_option_id="area_nairobi",
        )
    )

    persisted = _persisted_run(adapter, first.question_id, first.run_id)
    clarifications = persisted["resultData"]["details"]["clarifications"]
    assert second_wait.status == "WAITING_FOR_CLARIFICATION"
    assert [item["id"] for item in clarifications] == ["clarification_period"]

    adapter.lookup.complete_with_terminal(answer="42")
    completed = adapter.questions.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=first.run_id,
            clarification_id="clarification_period",
            response_text="This month",
            principal=QuestionPrincipal(
                principal_id=adapter.principal_id,
                tenant_id=adapter.tenant_id,
            ),
            execution_mode=ExecutionMode.INLINE,
            selected_option_id="area_nairobi",
        )
    )

    assert completed.status == "COMPLETED", completed.error
    request = adapter.lookup.last_request()
    assert request is not None
    assert tuple(
        response.clarification_id for response in request.clarification_responses
    ) == ("clarification_area", "clarification_period")


def test_idempotent_waiting_run_emits_actionable_clarification(adapter):
    adapter.lookup.needs_clarification()
    request = _ask(
        adapter,
        execution_mode=ExecutionMode.INLINE,
        idempotency_key="clarification-replay",
    )
    first = adapter.questions.ask(request)
    events = CollectingQuestionRunEventSink()

    replay = adapter.questions.ask(request, event_sink=events)

    assert replay.run_id == first.run_id
    assert replay.status == "WAITING_FOR_CLARIFICATION"
    assert events.events[-1]["event"] == "run.waiting_for_clarification"
    assert events.events[-1]["clarifications"][0]["id"] == "clarification_area"


def test_terminal_queued_run_short_circuits_without_lookup(adapter):
    queued = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.QUEUED))
    adapter.probe.record_terminal_error(queued.run_id, error="already_terminal")

    result = adapter.run_work.process_queued_run(QueuedRunRequest(run_id=queued.run_id))

    assert result.status == "FAILED"
    assert result.error == "already_terminal"
    assert adapter.lookup.call_count() == 0
    assert adapter.probe.work_item_status(queued.run_id) == "FAILED"


def test_terminal_lineage_reconciles_stale_active_work_item(adapter):
    queued = adapter.questions.ask(_ask(adapter, execution_mode=ExecutionMode.QUEUED))
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
    principal_id: str | None = None,
) -> AskRequest:
    return AskRequest(
        question=question,
        principal=QuestionPrincipal(
            principal_id=principal_id or adapter.principal_id,
            tenant_id=adapter.tenant_id,
        ),
        execution_mode=execution_mode,
        conversation_id=conversation_id,
        provider=adapter.provider,
        model_key=adapter.model_key,
        idempotency_key=idempotency_key,
    )


def _persisted_run(adapter, question_id: str, run_id: str) -> dict[str, object]:
    principal = QuestionPrincipal(
        principal_id=adapter.principal_id,
        tenant_id=adapter.tenant_id,
    )
    access = adapter.questions.runs.get_question(
        question_id=question_id,
        authority=ReadAuthority.from_principal(principal),
    )
    assert access is not None
    if adapter.name == "django":
        reader = DjangoQuestionStateReaderPort()
    elif adapter.name == "sql":
        reader = SQLQuestionStateReaderPort(adapter.probe.engine)
    else:
        raise AssertionError(f"unsupported contract adapter {adapter.name}")
    persisted = reader.get_question_run(run_id, access=access)
    assert persisted is not None
    return persisted


def _stable_run_fields(run: dict[str, object]) -> dict[str, object]:
    return {key: run.get(key) for key in _DESKTOP_RUN_FIELDS}
