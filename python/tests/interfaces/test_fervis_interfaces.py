from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from fervis.interfaces.agent.actions import (
    inspect_question_action,
    provide_clarification_action,
)


def test_fervis_django_urls_expose_question_lifecycle_routes() -> None:
    urls = importlib.import_module("fervis.integrations.django.urls")

    names = {pattern.name for pattern in urls.urlpatterns}
    routes = {str(pattern.pattern) for pattern in urls.urlpatterns}

    assert routes == {
        "",
        "conversations/",
        "questions/",
        "questions/<str:question_id>/",
        "questions/<str:question_id>/runs/",
        "questions/<str:question_id>/runs/<uuid:run_id>/",
    }
    assert names == {
        "fervis-runtime-status",
        "fervis-conversation-list",
        "fervis-question-create",
        "fervis-question-detail",
        "fervis-question-run-list",
        "fervis-question-run-detail",
    }


def test_fervis_django_views_use_fervis_throttle_scope() -> None:
    from fervis.interfaces.django.throttles import FervisQuestionThrottle
    from fervis.interfaces.django.views import (
        ConversationListView,
        FervisRuntimeStatusView,
        QuestionCreateView,
        QuestionDetailView,
        QuestionRunDetailView,
        QuestionRunListView,
    )

    assert {
        tuple(view.throttle_classes)
        for view in (
            FervisRuntimeStatusView,
            ConversationListView,
            QuestionCreateView,
            QuestionDetailView,
            QuestionRunListView,
            QuestionRunDetailView,
        )
    } == {(FervisQuestionThrottle,)}


def test_common_question_interface_creates_question_with_canonical_payload() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
    )
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="RUNNING",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-1",
        ),
        question={
            "questionId": "question-1",
            "latestRunId": "run-1",
            "conversationId": "conv-1",
            "status": "RUNNING",
            "answer": None,
            "resultData": None,
        },
    )
    interface = _question_interface(questions)

    response = interface.create_question(
        {
            "question": "How many orders were placed today?",
            "conversationId": "conv-1",
            "maxBudgetUsd": "0.25",
            "maxThinkingTokens": 128,
        },
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
        idempotency_key="idem-1",
    )

    assert response.status_code == 202
    assert response.payload == {
        "questionId": "question-1",
        "latestRunId": "run-1",
        "conversationId": "conv-1",
        "status": "RUNNING",
        "answer": None,
        "resultData": None,
    }
    assert questions.requests[0].question == "How many orders were placed today?"
    assert questions.requests[0].principal.principal_id == "user-1"
    assert questions.requests[0].principal.tenant_id == "tenant-1"
    assert questions.requests[0].provider == "openai"
    assert questions.requests[0].model_key == "openai:gpt-5.4-mini"
    assert questions.requests[0].idempotency_key == "idem-1"


def test_common_question_interface_requires_persisted_state_after_admission() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="QUEUED",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-1",
        ),
        question=None,
    )

    with pytest.raises(
        RuntimeError,
        match="admitted question is missing its persisted projection",
    ):
        _question_interface(questions).create_question(
            {"question": "How many orders were placed today?"},
            principal=InterfacePrincipal(
                principal_id="user-1",
                tenant_id="tenant-1",
            ),
        )


def test_common_question_interface_parses_explicit_context_run_id() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="RUNNING",
            conversation_id="conv-1",
            question_id="question-2",
            run_id="run-3",
        ),
        question={"questionId": "question-2", "status": "RUNNING"},
    )

    _question_interface(questions).create_question(
        {
            "question": "Break that down by store.",
            "conversationId": "conv-1",
            "contextRunId": "run-variant",
        },
        principal=InterfacePrincipal(
            principal_id="user-1",
            tenant_id="tenant-1",
        ),
    )

    assert questions.requests[0].context_run_id == "run-variant"


def test_common_question_interface_rejects_non_string_context_run_id() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
        QuestionInterfaceValidationError,
    )

    with pytest.raises(QuestionInterfaceValidationError) as error:
        _question_interface(_FakeQuestions()).create_question(
            {
                "question": "Break that down by store.",
                "conversationId": "conv-1",
                "contextRunId": 42,
            },
            principal=InterfacePrincipal(
                principal_id="user-1",
                tenant_id="tenant-1",
            ),
        )

    assert (error.value.field, error.value.message) == (
        "contextRunId",
        "contextRunId must be a non-empty string.",
    )


def test_common_question_interface_maps_unavailable_context_run_to_its_field() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
        QuestionInterfaceValidationError,
    )

    class _UnavailableContextQuestions(_FakeQuestions):
        def ask(self, request, *, event_sink=None):
            del request, event_sink
            raise PermissionError("context run is not an authorized answered run")

    with pytest.raises(QuestionInterfaceValidationError) as error:
        _question_interface(_UnavailableContextQuestions()).create_question(
            {
                "question": "Break that down by store.",
                "conversationId": "conv-1",
                "contextRunId": "run-unavailable",
            },
            principal=InterfacePrincipal(
                principal_id="user-1",
                tenant_id="tenant-1",
            ),
        )

    assert (error.value.field, error.value.message) == (
        "contextRunId",
        "context run is not an authorized answered run",
    )


def test_common_question_interface_lists_subject_conversations() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal

    questions = _FakeQuestions(
        conversations=[
            {
                "conversationId": "conv-2",
                "firstQuestion": "How many orders today?",
                "latestQuestionId": "question-2",
                "latestRunId": "run-3",
                "status": "RUNNING",
                "runCount": 1,
                "updatedAt": "2026-06-27T10:15:00Z",
            },
            {
                "conversationId": "conv-1",
                "firstQuestion": "How many sales yesterday?",
                "latestQuestionId": "question-1",
                "latestRunId": "run-2",
                "status": "COMPLETED",
                "runCount": 2,
                "updatedAt": "2026-06-26T09:00:00Z",
            },
        ]
    )
    interface = _question_interface(questions)

    response = interface.list_conversations(
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1")
    )

    assert response.status_code == 200
    assert response.payload == {
        "conversations": [
            {
                "conversationId": "conv-2",
                "firstQuestion": "How many orders today?",
                "latestQuestionId": "question-2",
                "latestRunId": "run-3",
                "status": "RUNNING",
                "runCount": 1,
                "updatedAt": "2026-06-27T10:15:00Z",
            },
            {
                "conversationId": "conv-1",
                "firstQuestion": "How many sales yesterday?",
                "latestQuestionId": "question-1",
                "latestRunId": "run-2",
                "status": "COMPLETED",
                "runCount": 2,
                "updatedAt": "2026-06-26T09:00:00Z",
            },
        ]
    }
    assert questions.conversation_requests == [
        {
            "tenant_id": "tenant-1",
            "read_context_ref": {
                "scheme": "anonymous",
                "key": None,
                "tenant_key": None,
            },
        }
    ]


def test_common_question_interface_rejects_public_model_outside_configured_policy() -> (
    None
):
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
        QuestionInterfaceValidationError,
    )

    interface = _question_interface(_FakeQuestions())

    try:
        interface.create_question(
            {
                "question": "How many orders?",
                "modelKey": "opencode:deepseek-v4-pro",
            },
            principal=InterfacePrincipal(
                principal_id="user-1",
                tenant_id="tenant-1",
            ),
        )
    except QuestionInterfaceValidationError as exc:
        assert (exc.field, exc.message) == (
            "modelKey",
            "modelKey is not allowed by the configured Fervis model policy.",
        )
    else:
        raise AssertionError("expected configured model policy to reject modelKey")


def test_common_question_interface_reports_provider_field_for_model_mismatch() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
        QuestionInterfaceValidationError,
    )

    interface = _question_interface(_FakeQuestions())

    try:
        interface.create_question(
            {
                "question": "How many orders?",
                "provider": "opencode",
                "modelKey": "openai:gpt-5.4-mini",
            },
            principal=InterfacePrincipal(
                principal_id="user-1",
                tenant_id="tenant-1",
            ),
        )
    except QuestionInterfaceValidationError as exc:
        assert (exc.field, exc.message) == (
            "provider",
            "provider must match the configured Fervis model policy.",
        )
    else:
        raise AssertionError("expected provider/model mismatch to be rejected")


def test_common_question_interface_uses_configured_default_model() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
    )
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="RUNNING",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-1",
        ),
        question={"questionId": "question-1", "status": "RUNNING"},
    )
    interface = _question_interface(questions)

    interface.create_question(
        {"question": "How many orders?"},
        principal=InterfacePrincipal(
            principal_id="user-1",
            tenant_id="tenant-1",
        ),
    )

    assert questions.requests[0].provider == "openai"
    assert questions.requests[0].model_key == "openai:gpt-5.4-mini"


def test_common_question_interface_allows_explicitly_configured_model_override() -> (
    None
):
    from fervis.interfaces.common.admission import ConfiguredModelPolicy
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
    )
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="RUNNING",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-1",
        ),
        question={"questionId": "question-1", "status": "RUNNING"},
    )
    interface = _question_interface(
        questions,
        model_policy=ConfiguredModelPolicy(
            default_provider="openai",
            default_model_key="gpt-5.4-mini",
            allowed_model_keys_by_provider={
                "openai": frozenset({"gpt-5.4-mini"}),
                "opencode": frozenset({"deepseek-v4-pro"}),
            },
        ),
    )

    interface.create_question(
        {
            "question": "How many orders?",
            "modelKey": "opencode:deepseek-v4-pro",
        },
        principal=InterfacePrincipal(
            principal_id="user-1",
            tenant_id="tenant-1",
        ),
    )

    assert questions.requests[0].provider == "opencode"
    assert questions.requests[0].model_key == "opencode:deepseek-v4-pro"


def test_common_question_interface_lists_runs_for_question() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
    )

    questions = _FakeQuestions(
        question={"questionId": "question-1", "status": "COMPLETED"},
        runs=[
            {"runId": "run-1", "questionId": "question-1", "status": "FAILED"},
            {"runId": "run-2", "questionId": "question-1", "status": "COMPLETED"},
        ],
    )
    interface = _question_interface(questions)

    response = interface.list_question_runs(
        "question-1",
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
    )

    assert response.status_code == 200
    assert response.payload == {
        "questionId": "question-1",
        "runs": [
            {"runId": "run-1", "questionId": "question-1", "status": "FAILED"},
            {"runId": "run-2", "questionId": "question-1", "status": "COMPLETED"},
        ],
    }
    assert questions.state_requests == [
        {
            "question_id": "question-1",
            "tenant_id": "tenant-1",
            "read_context_ref": {
                "scheme": "anonymous",
                "key": None,
                "tenant_key": None,
            },
        }
    ]
    assert questions.list_requests == [
        {
            "question_id": "question-1",
            "tenant_id": "tenant-1",
            "read_context_ref": {
                "scheme": "anonymous",
                "key": None,
                "tenant_key": None,
            },
        }
    ]


def test_common_question_interface_continues_question_for_clarification_response() -> (
    None
):
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
    )
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="RUNNING",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-2",
        ),
        question={
            "questionId": "question-1",
            "latestRunId": "run-2",
            "conversationId": "conv-1",
            "status": "RUNNING",
        },
    )
    interface = _question_interface(questions)

    response = interface.create_question_run(
        "question-1",
        {
            "responseText": "ABC Mall",
            "runId": "run-1",
            "clarificationId": "clar-1",
            "selectedOptionId": "store:abc",
        },
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
    )

    request = questions.continue_requests[0]
    assert response.status_code == 202
    assert response.payload["latestRunId"] == "run-2"
    assert request.question_id == "question-1"
    assert request.run_id == "run-1"
    assert request.clarification_id == "clar-1"
    assert request.selected_option_id == "store:abc"


def test_common_question_interface_accepts_selected_clarification_option() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="RUNNING",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-1",
        ),
        question={
            "questionId": "question-1",
            "latestRunId": "run-1",
            "conversationId": "conv-1",
            "status": "RUNNING",
        },
    )
    interface = _question_interface(questions)

    response = interface.create_question_run(
        "question-1",
        {
            "runId": "run-1",
            "clarificationId": "clar-1",
            "selectedOptionId": "store:abc",
        },
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
    )

    request = questions.continue_requests[0]
    assert response.status_code == 202
    assert request.response_text == ""
    assert request.selected_option_id == "store:abc"


def test_common_question_interface_parses_typed_deterministic_rerun() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal
    from fervis.lookup.answer_program.values import StringSetValuePayload
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="QUEUED",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-2",
        ),
        question={
            "questionId": "question-1",
            "primaryRunId": "run-1",
            "latestRunId": "run-2",
            "activeRunId": "run-2",
            "conversationId": "conv-1",
            "status": "COMPLETED",
        },
    )

    response = _question_interface(questions).create_question_run(
        "question-1",
        {
            "triggerKind": "rerun",
            "baseRunId": "run-1",
            "patch": {
                "operations": [
                    {
                        "kind": "set",
                        "parameterId": "population.sale_states",
                        "value": {
                            "kind": "string_set",
                            "values": ["COMPLETED", "PLACED"],
                        },
                    }
                ]
            },
        },
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
        idempotency_key="rerun-1",
    )

    request = questions.rerun_requests[0]
    operation = request.patch.operations[0]
    assert response.status_code == 202
    assert request.question_id == "question-1"
    assert request.base_run_id == "run-1"
    assert request.idempotency_key == "rerun-1"
    assert operation.parameter_id == "population.sale_states"
    assert isinstance(operation.value.payload, StringSetValuePayload)
    assert operation.value.payload.values == ("COMPLETED", "PLACED")


def test_common_question_interface_accepts_same_binding_rerun_without_patch() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="QUEUED",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-2",
        ),
        question={
            "questionId": "question-1",
            "primaryRunId": "run-1",
            "latestRunId": "run-2",
            "activeRunId": "run-2",
            "conversationId": "conv-1",
            "status": "COMPLETED",
        },
    )

    response = _question_interface(questions).create_question_run(
        "question-1",
        {"triggerKind": "rerun", "baseRunId": "run-1"},
        principal=InterfacePrincipal(
            principal_id="user-1",
            tenant_id="tenant-1",
        ),
    )

    assert response.status_code == 202
    assert questions.rerun_requests[0].patch is None


def test_common_question_interface_parses_declared_capability_application() -> None:
    from fervis.interfaces.common.questions import InterfacePrincipal
    from fervis.lookup.answer_program import canonical_fact_value
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="QUEUED",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-2",
        ),
        question={
            "questionId": "question-1",
            "primaryRunId": "run-1",
            "latestRunId": "run-2",
            "activeRunId": "run-2",
            "conversationId": "conv-1",
            "status": "COMPLETED",
        },
    )

    response = _question_interface(questions).create_question_run(
        "question-1",
        {
            "triggerKind": "rerun",
            "baseRunId": "run-1",
            "capabilityApplication": {
                "capabilityId": "filter_by_sale_channel",
                "binding": {
                    "parameterId": "semantic.sale_channels",
                    "value": {"kind": "string_set", "values": ["STORE"]},
                },
            },
        },
        principal=InterfacePrincipal(
            principal_id="user-1",
            tenant_id="tenant-1",
        ),
    )

    application = questions.rerun_requests[0].capability_application
    assert response.status_code == 202
    assert application is not None
    assert application.capability_id == "filter_by_sale_channel"
    assert application.binding.parameter_id == "semantic.sale_channels"
    assert canonical_fact_value(application.binding.value) == ["STORE"]
    assert application.binding.provenance.refs == ("capability:filter_by_sale_channel",)


def test_common_question_interface_adds_transport_neutral_clarification_follow_up_actions() -> (
    None
):
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
    )
    from fervis.questions import AskResult

    questions = _FakeQuestions(
        AskResult(
            status="WAITING_FOR_CLARIFICATION",
            conversation_id="conv-1",
            question_id="question-1",
            run_id="run-1",
        ),
        question={
            "questionId": "question-1",
            "latestRunId": "run-1",
            "conversationId": "conv-1",
            "status": "WAITING_FOR_CLARIFICATION",
            "resultData": {
                "kind": "needs_clarification",
                "details": {
                    "clarifications": [
                        {"id": "clar-1", "question": "Which store do you mean?"}
                    ]
                },
            },
        },
    )
    interface = _question_interface(questions)

    response = interface.create_question(
        {"question": "How many sales?", "conversationId": "conv-1"},
        principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
    )

    assert response.payload["nextActions"] == [
        {
            "kind": "provide_clarification",
            "questionId": "question-1",
            "conversationId": "conv-1",
            "runId": "run-1",
            "clarificationId": "clar-1",
            "request": {
                "method": "POST",
                "path": "/questions/question-1/runs/",
                "body": {
                    "responseText": "<clarification-answer>",
                    "runId": "run-1",
                    "clarificationId": "clar-1",
                    "selectedOptionId": "<selected-option-id>",
                },
            },
        }
    ]


def test_common_question_interface_rejects_unknown_create_fields() -> None:
    from fervis.interfaces.common.questions import (
        InterfacePrincipal,
        QuestionInterfaceValidationError,
    )

    interface = _question_interface(_FakeQuestions())

    try:
        interface.create_question(
            {"question": "hello", "surprise": True},
            principal=InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1"),
        )
    except QuestionInterfaceValidationError as exc:
        assert (exc.field, exc.message) == (
            "surprise",
            "surprise is not a supported field.",
        )
    else:
        raise AssertionError("expected validation error")


def test_fervis_fastapi_router_exposes_question_lifecycle_routes() -> None:
    from fastapi import FastAPI
    from fervis import (
        FervisConfig,
        HostConfig,
        ModelConfig,
        ProviderConfig,
        RuntimeRoutes,
    )
    from fervis.integrations.fastapi import FastAPIIntegration

    router = FastAPIIntegration(
        config=FervisConfig(
            host=HostConfig(timezone="UTC"),
            routes=RuntimeRoutes(prefix="/fervis/"),
            model=ModelConfig(
                default_provider="openai",
                default_model_key="gpt-5.4-mini",
                providers=[
                    ProviderConfig(
                        name="openai",
                        allowed_model_keys=["gpt-5.4-mini"],
                    )
                ],
            ),
            sources=[],
        )
    ).router(question_interface=_FakeQuestionInterface())

    app = FastAPI()
    app.include_router(router, prefix="/fervis")

    paths = app.openapi()["paths"]
    assert {
        (method.upper(), path)
        for path, operations in paths.items()
        for method in operations
    } >= {
        ("GET", "/fervis/"),
        ("GET", "/fervis/conversations/"),
        ("POST", "/fervis/questions/"),
        ("GET", "/fervis/questions/{question_id}/"),
        ("GET", "/fervis/questions/{question_id}/runs/"),
        ("GET", "/fervis/questions/{question_id}/runs/{run_id}/"),
    }


def test_fastapi_integration_closes_question_interface_with_host_lifespan() -> None:
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fervis import FervisConfig, HostConfig, ModelConfig, RuntimeRoutes
    from fervis.integrations.fastapi import FastAPIIntegration

    events: list[str] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        events.append("host_started")
        yield
        events.append("host_stopped")

    class CloseableQuestionInterface(_FakeQuestionInterface):
        def close(self) -> None:
            events.append("fervis_closed")

    app = FastAPI(lifespan=lifespan)
    integration = FastAPIIntegration(
        config=FervisConfig(
            host=HostConfig(timezone="UTC"),
            routes=RuntimeRoutes(prefix="/fervis/"),
            model=ModelConfig(
                default_provider="openai",
                default_model_key="gpt-5.4-mini",
            ),
            sources=[],
        )
    )
    integration.mount(app, question_interface=CloseableQuestionInterface())

    with TestClient(app) as client:
        response = client.get("/fervis/")
        assert response.status_code == 200
        assert events == ["host_started"]

    assert events == ["host_started", "fervis_closed", "host_stopped"]


def test_fervis_fastapi_router_requires_question_interface() -> None:
    from fervis.integrations.fastapi import fervis_fastapi_router

    try:
        fervis_fastapi_router()
    except TypeError as exc:
        assert "question_interface" in str(exc)
    else:
        raise AssertionError("expected question_interface to be required")


def test_fervis_fastapi_router_denies_anonymous_questions_by_default() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fervis.integrations.fastapi import fervis_fastapi_router

    interface = _FakeQuestionInterface()
    app = FastAPI()
    app.include_router(
        fervis_fastapi_router(question_interface=interface),
        prefix="/fervis",
    )

    response = TestClient(app).post(
        "/fervis/questions/",
        json={"question": "hello"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == {
        "error": {
            "type": "authorization",
            "code": "read_context_required",
            "message": "Fervis could not capture an authenticated read context.",
            "retryable": False,
        }
    }
    assert interface.created == []


def test_fervis_fastapi_router_uses_configured_common_question_interface() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fervis.host_api.contracts.authority import ReadContextRef
    from fervis.integrations.fastapi import fervis_fastapi_router

    interface = _FakeQuestionInterface()
    app = FastAPI()
    app.include_router(
        fervis_fastapi_router(
            question_interface=interface,
            read_context_capture=lambda request: ReadContextRef(
                scheme="fastapi_principal",
                key="user-1",
            ),
        ),
        prefix="/fervis",
    )

    client = TestClient(app)
    response = client.post(
        "/fervis/questions/",
        json={"question": "hello"},
        headers={"Idempotency-Key": "idem-1"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "questionId": "question-1",
        "latestRunId": "run-1",
        "status": "RUNNING",
    }
    assert interface.created == [
        {
            "payload": {"question": "hello"},
            "principal_id": "user-1",
            "tenant_id": "default",
            "read_context_ref": {
                "scheme": "fastapi_principal",
                "key": "user-1",
                "tenant_key": None,
            },
            "idempotency_key": "idem-1",
        }
    ]

    response = client.get("/fervis/conversations/")
    assert response.status_code == 200
    assert response.json() == {
        "conversations": [
            {
                "conversationId": "conv-1",
                "firstQuestion": "How many orders?",
                "latestQuestionId": "question-1",
                "latestRunId": "run-1",
                "status": "RUNNING",
                "runCount": 1,
                "updatedAt": "2026-06-27T10:15:00Z",
            }
        ]
    }

    response = client.get("/fervis/questions/question-1/")
    assert response.status_code == 200
    assert response.json() == {"questionId": "question-1", "status": "RUNNING"}

    response = client.get("/fervis/questions/question-1/runs/")
    assert response.status_code == 200
    assert response.json() == {"questionId": "question-1", "runs": []}

    response = client.post(
        "/fervis/questions/question-1/runs/",
        json={
            "question": "ABC Mall",
            "triggerKind": "clarification_response",
            "baseRunId": "run-1",
            "clarificationId": "clar-1",
        },
        headers={"Idempotency-Key": "idem-2"},
    )
    assert response.status_code == 202
    assert response.json() == {
        "questionId": "question-1",
        "latestRunId": "run-2",
        "status": "RUNNING",
    }
    assert interface.continued == [
        {
            "question_id": "question-1",
            "payload": {
                "question": "ABC Mall",
                "triggerKind": "clarification_response",
                "baseRunId": "run-1",
                "clarificationId": "clar-1",
            },
            "principal_id": "user-1",
            "tenant_id": "default",
            "read_context_ref": {
                "scheme": "fastapi_principal",
                "key": "user-1",
                "tenant_key": None,
            },
            "idempotency_key": "idem-2",
        }
    ]

    rerun_payload = {
        "triggerKind": "rerun",
        "baseRunId": "run-1",
        "patch": {
            "operations": [
                {
                    "kind": "set",
                    "parameterId": "population.sale_states",
                    "value": {
                        "kind": "string_set",
                        "values": ["COMPLETED", "PLACED"],
                    },
                }
            ]
        },
    }
    rerun_response = client.post(
        "/fervis/questions/question-1/runs/",
        json=rerun_payload,
        headers={"Idempotency-Key": "idem-rerun"},
    )
    assert rerun_response.status_code == 202
    assert interface.continued[1] == {
        "question_id": "question-1",
        "payload": rerun_payload,
        "principal_id": "user-1",
        "tenant_id": "default",
        "read_context_ref": {
            "scheme": "fastapi_principal",
            "key": "user-1",
            "tenant_key": None,
        },
        "idempotency_key": "idem-rerun",
    }

    response = client.get("/fervis/questions/question-1/runs/run-1/")
    assert response.status_code == 200
    assert response.json() == {
        "questionId": "question-1",
        "runId": "run-1",
        "status": "RUNNING",
    }


def test_fervis_fastapi_router_captures_configured_dependency_principal() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fervis.integrations.fastapi import fervis_fastapi_router

    def current_user() -> dict[str, str]:
        return {"id": "dep-user-1"}

    interface = _FakeQuestionInterface()
    app = FastAPI()
    app.include_router(
        fervis_fastapi_router(
            question_interface=interface,
            principal_dependency=current_user,
            principal_id_attr="id",
            require_read_context=True,
        ),
        prefix="/fervis",
    )

    client = TestClient(app)
    create_response = client.post(
        "/fervis/questions/",
        json={"question": "hello"},
    )
    continue_response = client.post(
        "/fervis/questions/question-1/runs/",
        json={
            "question": "ABC Mall",
            "triggerKind": "clarification_response",
            "baseRunId": "run-1",
            "clarificationId": "clar-1",
        },
    )

    assert create_response.status_code == 202
    assert continue_response.status_code == 202
    assert interface.created[0]["principal_id"] == "dep-user-1"
    assert interface.continued[0]["principal_id"] == "dep-user-1"
    assert interface.created[0]["read_context_ref"] == {
        "scheme": "fastapi_principal",
        "key": "dep-user-1",
        "tenant_key": None,
    }
    assert interface.continued[0]["read_context_ref"] == {
        "scheme": "fastapi_principal",
        "key": "dep-user-1",
        "tenant_key": None,
    }


def test_fervis_fastapi_router_fails_closed_when_subject_is_required() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fervis.integrations.fastapi import fervis_fastapi_router

    interface = _FakeQuestionInterface()
    app = FastAPI()
    app.include_router(
        fervis_fastapi_router(
            question_interface=interface,
            require_read_context=True,
        ),
        prefix="/fervis",
    )

    response = TestClient(app).post(
        "/fervis/questions/",
        json={"question": "hello"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == {
        "error": {
            "type": "authorization",
            "code": "read_context_required",
            "message": "Fervis could not capture an authenticated read context.",
            "retryable": False,
        }
    }
    assert interface.created == []


def test_fervis_flask_blueprint_exposes_question_lifecycle_routes() -> None:
    from flask import Flask
    from fervis.interfaces.flask import fervis_flask_blueprint

    app = Flask(__name__)
    app.register_blueprint(
        fervis_flask_blueprint(question_interface=_FakeQuestionInterface()),
        url_prefix="/fervis",
    )

    rules = {
        (next(iter(rule.methods & {"GET", "POST"})), rule.rule)
        for rule in app.url_map.iter_rules()
        if rule.endpoint != "static"
    }
    assert rules == {
        ("GET", "/fervis/"),
        ("GET", "/fervis/conversations/"),
        ("POST", "/fervis/questions/"),
        ("GET", "/fervis/questions/<question_id>/"),
        ("GET", "/fervis/questions/<question_id>/runs/"),
        ("POST", "/fervis/questions/<question_id>/runs/"),
        ("GET", "/fervis/questions/<question_id>/runs/<run_id>/"),
    }


def test_fervis_flask_blueprint_uses_configured_common_question_interface() -> None:
    from flask import Flask
    from fervis.host_api.contracts.authority import ReadContextRef
    from fervis.interfaces.flask import fervis_flask_blueprint

    interface = _FakeQuestionInterface()
    app = Flask(__name__)
    app.register_blueprint(
        fervis_flask_blueprint(
            question_interface=interface,
            read_context_capture=lambda request: ReadContextRef(
                scheme="flask_principal",
                key="user-1",
            ),
        ),
        url_prefix="/fervis",
    )

    client = app.test_client()
    response = client.post(
        "/fervis/questions/",
        json={"question": "hello"},
        headers={"Idempotency-Key": "idem-1"},
    )

    assert response.status_code == 202
    assert response.get_json() == {
        "questionId": "question-1",
        "latestRunId": "run-1",
        "status": "RUNNING",
    }
    assert interface.created == [
        {
            "payload": {"question": "hello"},
            "principal_id": "user-1",
            "tenant_id": "default",
            "read_context_ref": {
                "scheme": "flask_principal",
                "key": "user-1",
                "tenant_key": None,
            },
            "idempotency_key": "idem-1",
        }
    ]

    response = client.get("/fervis/conversations/")
    assert response.status_code == 200
    assert response.get_json() == {
        "conversations": [
            {
                "conversationId": "conv-1",
                "firstQuestion": "How many orders?",
                "latestQuestionId": "question-1",
                "latestRunId": "run-1",
                "status": "RUNNING",
                "runCount": 1,
                "updatedAt": "2026-06-27T10:15:00Z",
            }
        ]
    }

    assert client.get("/fervis/questions/question-1/").get_json() == {
        "questionId": "question-1",
        "status": "RUNNING",
    }
    assert client.get("/fervis/questions/question-1/runs/").get_json() == {
        "questionId": "question-1",
        "runs": [],
    }
    continue_response = client.post(
        "/fervis/questions/question-1/runs/",
        json={
            "question": "ABC Mall",
            "triggerKind": "clarification_response",
            "baseRunId": "run-1",
            "clarificationId": "clar-1",
        },
        headers={"Idempotency-Key": "idem-2"},
    )
    assert continue_response.status_code == 202
    assert interface.continued == [
        {
            "question_id": "question-1",
            "payload": {
                "question": "ABC Mall",
                "triggerKind": "clarification_response",
                "baseRunId": "run-1",
                "clarificationId": "clar-1",
            },
            "principal_id": "user-1",
            "tenant_id": "default",
            "read_context_ref": {
                "scheme": "flask_principal",
                "key": "user-1",
                "tenant_key": None,
            },
            "idempotency_key": "idem-2",
        }
    ]
    assert client.get("/fervis/questions/question-1/runs/run-1/").get_json() == {
        "questionId": "question-1",
        "runId": "run-1",
        "status": "RUNNING",
    }


def test_fervis_flask_blueprint_fails_closed_when_subject_is_required() -> None:
    from flask import Flask
    from fervis.interfaces.flask import fervis_flask_blueprint

    interface = _FakeQuestionInterface()
    app = Flask(__name__)
    app.register_blueprint(
        fervis_flask_blueprint(
            question_interface=interface,
            require_read_context=True,
        ),
        url_prefix="/fervis",
    )

    response = app.test_client().post(
        "/fervis/questions/",
        json={"question": "hello"},
    )

    assert response.status_code == 401
    assert response.get_json() == {
        "error": {
            "type": "authorization",
            "code": "read_context_required",
            "message": "Fervis could not capture an authenticated read context.",
            "retryable": False,
        }
    }
    assert interface.created == []


def test_agent_run_event_projection_adds_follow_up_actions_once() -> None:
    from fervis.interfaces.common.events import agent_run_event

    assert agent_run_event(
        {
            "event": "run.waiting_for_clarification",
            "conversation_id": "conv-1",
            "question_id": "question-1",
            "run_id": "run-1",
            "status": "WAITING_FOR_CLARIFICATION",
            "clarifications": [
                {
                    "id": "clar-1",
                    "question": "Which store should I use?",
                }
            ],
        },
        tenant_id="tenant-1",
        principal_id="user-1",
    )["next_actions"] == [
        provide_clarification_action(
            "conv-1",
            question_id="question-1",
            run_id="run-1",
            clarification_id="clar-1",
            tenant_id="tenant-1",
            principal_id="user-1",
        )
    ]

    assert agent_run_event(
        {
            "event": "run.completed",
            "question_id": "question-2",
            "run_id": "run-2",
            "status": "COMPLETED",
            "answer": "42",
            "result_data": {},
        },
    )["next_actions"] == [inspect_question_action("question-2")]

    failed_next_actions = agent_run_event(
        {
            "event": "run.failed",
            "question_id": "question-3",
            "run_id": "run-3",
            "status": "FAILED",
            "error": {"code": "provider_runtime_failed"},
        },
    )["next_actions"]

    assert failed_next_actions == [inspect_question_action("question-3", debug=True)]


def test_agent_run_event_projection_requires_actionable_clarification() -> None:
    from fervis.interfaces.common.events import agent_run_event

    with pytest.raises(
        ValueError,
        match="run.waiting_for_clarification event requires clarifications",
    ):
        agent_run_event(
            {
                "event": "run.waiting_for_clarification",
                "conversation_id": "conv-1",
                "question_id": "question-1",
                "run_id": "run-1",
                "status": "WAITING_FOR_CLARIFICATION",
                "clarifications": [],
            },
        )

    with pytest.raises(
        ValueError,
        match="run.waiting_for_clarification event requires clarification question",
    ):
        agent_run_event(
            {
                "event": "run.waiting_for_clarification",
                "conversation_id": "conv-1",
                "question_id": "question-1",
                "run_id": "run-1",
                "status": "WAITING_FOR_CLARIFICATION",
                "clarifications": [{"id": "clar-1"}],
            },
        )


def test_common_question_interface_has_no_framework_or_runtime_internal_imports() -> (
    None
):
    from pathlib import Path

    source = Path("src/fervis/interfaces/common/questions.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "django",
        "fastapi",
        "rest_framework",
        "fervis.lookup",
        "fervis.lineage",
        "fervis.storage",
        "fervis.run_work.queue",
    ]

    assert [token for token in forbidden if token in source] == []


def test_question_and_run_execution_services_import_independently() -> None:
    import importlib

    assert importlib.import_module("fervis.run_work.service")
    assert importlib.import_module("fervis.questions.service")


@dataclass
class _FakeQuestions:
    result: object | None = None
    question: dict | None = None
    conversations: list[dict] | None = None
    runs: list[dict] | None = None
    run: dict | None = None

    def __post_init__(self) -> None:
        self.requests = []
        self.continue_requests = []
        self.rerun_requests = []
        self.conversation_requests = []
        self.state_requests = []
        self.list_requests = []
        self.run_requests = []

    def ask(self, request, *, event_sink=None):
        del event_sink
        self.requests.append(request)
        if self.result is None:
            raise AssertionError("questions.ask should not be called")
        return self.result

    def respond_to_clarification(self, request, *, event_sink=None):
        del event_sink
        self.continue_requests.append(request)
        if self.result is None:
            raise AssertionError(
                "questions.respond_to_clarification should not be called"
            )
        return self.result

    def rerun_question(self, request, *, event_sink=None):
        del event_sink
        self.rerun_requests.append(request)
        if self.result is None:
            raise AssertionError("questions.rerun_question should not be called")
        return self.result

    def list_conversations(self, *, principal):
        self.conversation_requests.append(
            {
                "tenant_id": principal.tenant_id,
                "read_context_ref": principal.read_context_ref.to_storage_dict(),
            }
        )
        return list(self.conversations or [])

    def get_question_state(
        self,
        question_id: str,
        *,
        principal,
    ):
        self.state_requests.append(
            {
                "question_id": question_id,
                "tenant_id": principal.tenant_id,
                "read_context_ref": principal.read_context_ref.to_storage_dict(),
            }
        )
        return self.question

    def list_question_runs(
        self,
        question_id: str,
        *,
        principal,
    ):
        self.list_requests.append(
            {
                "question_id": question_id,
                "tenant_id": principal.tenant_id,
                "read_context_ref": principal.read_context_ref.to_storage_dict(),
            }
        )
        return list(self.runs or [])

    def get_question_run(
        self,
        question_id: str,
        run_id: str,
        *,
        principal,
    ):
        self.run_requests.append(
            {
                "question_id": question_id,
                "run_id": run_id,
                "tenant_id": principal.tenant_id,
                "read_context_ref": principal.read_context_ref.to_storage_dict(),
            }
        )
        return self.run


def _question_interface(
    questions: _FakeQuestions,
    *,
    model_policy=None,
):
    from fervis.interfaces.common.admission import ConfiguredModelPolicy
    from fervis.interfaces.common.questions import QuestionInterface

    return QuestionInterface(
        questions=questions,
        model_policy=model_policy
        or ConfiguredModelPolicy(
            default_provider="openai",
            default_model_key="gpt-5.4-mini",
            allowed_model_keys_by_provider={"openai": frozenset({"gpt-5.4-mini"})},
        ),
    )


class _FakeQuestionInterface:
    def __init__(self):
        self.created = []
        self.continued = []

    def list_conversations(self, *, principal):
        del principal
        return _FakeInterfaceResponse(
            200,
            {
                "conversations": [
                    {
                        "conversationId": "conv-1",
                        "firstQuestion": "How many orders?",
                        "latestQuestionId": "question-1",
                        "latestRunId": "run-1",
                        "status": "RUNNING",
                        "runCount": 1,
                        "updatedAt": "2026-06-27T10:15:00Z",
                    }
                ]
            },
        )

    def create_question(self, payload, *, principal, idempotency_key=None):
        self.created.append(
            {
                "payload": dict(payload),
                "principal_id": principal.principal_id,
                "tenant_id": principal.tenant_id,
                "read_context_ref": principal.read_context_ref.to_storage_dict(),
                "idempotency_key": idempotency_key,
            }
        )
        return _FakeInterfaceResponse(
            202,
            {
                "questionId": "question-1",
                "latestRunId": "run-1",
                "status": "RUNNING",
            },
        )

    def create_question_run(
        self,
        question_id,
        payload,
        *,
        principal,
        idempotency_key=None,
    ):
        self.continued.append(
            {
                "question_id": question_id,
                "payload": dict(payload),
                "principal_id": principal.principal_id,
                "tenant_id": principal.tenant_id,
                "read_context_ref": principal.read_context_ref.to_storage_dict(),
                "idempotency_key": idempotency_key,
            }
        )
        return _FakeInterfaceResponse(
            202,
            {
                "questionId": question_id,
                "latestRunId": "run-2",
                "status": "RUNNING",
            },
        )

    def get_question(self, question_id, *, principal):
        del principal
        return _FakeInterfaceResponse(
            200, {"questionId": question_id, "status": "RUNNING"}
        )

    def list_question_runs(self, question_id, *, principal):
        del principal
        return _FakeInterfaceResponse(200, {"questionId": question_id, "runs": []})

    def get_question_run(self, question_id, run_id, *, principal):
        del principal
        return _FakeInterfaceResponse(
            200,
            {"questionId": question_id, "runId": run_id, "status": "RUNNING"},
        )


@dataclass(frozen=True)
class _FakeInterfaceResponse:
    status_code: int
    payload: object
