import pytest

from .helpers import (
    current_run_detail,
    post_fervis_question,
    question_detail_url,
    question_run_detail_url,
    question_runs_url,
    run_worker_until_terminal,
)


@pytest.mark.django_db
def test_question_detail_reads_current_run_from_lineage(
    api_client,
    fervis_foundation_reset,
):
    question = post_fervis_question(api_client, "best product").json()

    response = api_client.get(
        question_detail_url(question["questionId"]),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )

    assert (
        response.status_code,
        response.json()["questionId"],
        response.json()["currentRunId"],
        response.json()["conversationId"],
    ) == (
        200,
        question["questionId"],
        question["currentRunId"],
        question["conversationId"],
    )


@pytest.mark.django_db
def test_question_run_detail_includes_structured_answer_explanation(
    api_client,
    fervis_foundation_reset,
):
    response = post_fervis_question(
        api_client,
        "How many in-person sales happened this month?",
    )
    question = run_worker_until_terminal(api_client, response).json()

    response = current_run_detail(api_client, question)

    explanation = response.json()["explanation"]
    assert (
        response.status_code,
        response.json()["questionId"],
        set(explanation),
        set(explanation["lineage"]),
        explanation["inputs"]["rootKind"],
        explanation["lineage"]["verbose"]["rootId"],
    ) == (
        200,
        question["questionId"],
        {"inputs", "lineage"},
        {"compact", "verbose"},
        "run",
        question["currentRunId"],
    )


@pytest.mark.django_db
def test_question_runs_lists_attempts_for_question(
    api_client,
    fervis_foundation_reset,
):
    question = post_fervis_question(api_client, "show sales").json()

    response = api_client.get(
        question_runs_url(question["questionId"]),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )

    assert response.status_code == 200
    assert response.json()["questionId"] == question["questionId"]
    assert [run["runId"] for run in response.json()["runs"]] == [
        question["currentRunId"]
    ]


@pytest.mark.django_db
def test_question_run_detail_is_tenant_scoped(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    from fervis.interfaces.django import principal

    monkeypatch.setattr(principal, "tenant_from_request", lambda request: "tenant-a")
    question = post_fervis_question(api_client, "show sales").json()

    monkeypatch.setattr(principal, "tenant_from_request", lambda request: "tenant-b")
    response = api_client.get(
        question_run_detail_url(question["questionId"], question["currentRunId"]),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )

    assert response.status_code == 404


@pytest.mark.django_db
def test_question_create_rejects_second_active_run_for_same_tenant_conversation(
    api_client,
    fervis_foundation_reset,
):
    first = post_fervis_question(api_client, "show sales").json()

    second = post_fervis_question(
        api_client,
        "show sales again",
        conversation_id=first["conversationId"],
    )

    assert (second.status_code, second.json()["error"]["context"]["activeRunId"]) == (
        409,
        first["currentRunId"],
    )
