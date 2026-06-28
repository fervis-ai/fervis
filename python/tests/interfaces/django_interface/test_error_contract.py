import pytest

from tests.interfaces.django_interface.helpers import (
    question_run_detail_url,
    questions_url,
)


ERROR_FIELDS = {
    "type",
    "code",
    "message",
    "developer_message",
    "retryable",
    "retry_after",
    "request_id",
    "details",
    "context",
}


def _error_summary(response):
    body = response.json()
    error = body["error"]
    return (
        response.status_code,
        set(body.keys()),
        set(error.keys()),
        error["type"],
        error["code"],
        bool(error["request_id"]),
        error["details"],
        error["context"],
    )


@pytest.mark.django_db
def test_fervis_run_rejects_non_object_payload_with_error_envelope(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.post(
        questions_url(),
        ["not", "an", "object"],
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "__all__",
                "code": "invalid",
                "message": "Payload must be an object.",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_requires_question_with_error_envelope(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.post(
        questions_url(),
        {},
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "question",
                "code": "invalid",
                "message": "question is required.",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_rejects_empty_model_key_with_error_envelope(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.post(
        questions_url(),
        {"question": "show sales", "modelKey": ""},
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "modelKey",
                "code": "invalid",
                "message": "modelKey must not be empty.",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_rejects_unknown_request_fields_with_error_envelope(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.post(
        questions_url(),
        {"question": "show sales", "model": "GPT_5_4_MINI"},
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "model",
                "code": "unknown",
                "message": "model is not a supported field.",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_rejects_public_evaluation_context(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.post(
        questions_url(),
        {
            "question": "show sales",
            "evaluationContext": {"caseId": "forged-case"},
        },
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "evaluationContext",
                "code": "unknown",
                "message": "evaluationContext is not a supported field.",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_budget_maximum_is_server_configured(
    api_client,
    fervis_foundation_reset,
    settings,
):
    settings.FERVIS_MAX_REQUEST_BUDGET_USD = 0.25
    response = api_client.post(
        questions_url(),
        {
            "question": "show sales",
            "maxBudgetUsd": 0.5,
        },
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "maxBudgetUsd",
                "code": "invalid",
                "message": "max_budget_usd must be at most 0.25",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_budget_lower_bound_uses_error_envelope(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.post(
        questions_url(),
        {
            "question": "show sales",
            "maxBudgetUsd": 0,
        },
        format="json",
        HTTP_X_REQUESTER_SCOPES="fervis:write fervis:read",
    )

    assert _error_summary(response) == (
        400,
        {"error"},
        ERROR_FIELDS,
        "validation",
        "validation_error",
        True,
        [
            {
                "field": "maxBudgetUsd",
                "code": "invalid",
                "message": "max_budget_usd must be greater than 0",
            }
        ],
        {},
    )


@pytest.mark.django_db
def test_fervis_run_detail_not_found_uses_resource_error_envelope(
    api_client,
    fervis_foundation_reset,
):
    response = api_client.get(
        question_run_detail_url(
            "missing-question",
            "00000000-0000-0000-0000-000000000000",
        ),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )

    assert _error_summary(response) == (
        404,
        {"error"},
        ERROR_FIELDS,
        "not_found",
        "fervis_run_not_found",
        True,
        [],
        {
            "resource_type": "fervis_run",
            "resource_id": "00000000-0000-0000-0000-000000000000",
        },
    )
