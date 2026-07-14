from datetime import timedelta

import pytest
from django.utils import timezone

from fervis.host_api.context import get_host_api_context
from fervis.host_api.contracts.authority import ReadContextRef
from fervis.run_work.queue.django.models import RunWorkItem

from .helpers import (
    primary_run_detail,
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
        response.json()["latestRunId"],
        response.json()["conversationId"],
    ) == (
        200,
        question["questionId"],
        question["latestRunId"],
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

    response = primary_run_detail(api_client, question)

    explanation = response.json()["explanation"]
    duration_ms = response.json()["durationMs"]
    assert (
        response.status_code,
        response.json()["questionId"],
        set(explanation),
        set(explanation["lineage"]),
        explanation["inputs"]["rootKind"],
        explanation["lineage"]["verbose"]["rootId"],
        isinstance(duration_ms, int),
    ) == (
        200,
        question["questionId"],
        {"inputs", "lineage"},
        {"compact", "verbose"},
        "run",
        question["latestRunId"],
        True,
    )


@pytest.mark.django_db
def test_question_run_duration_spans_submission_to_terminal_result(
    api_client,
    fervis_foundation_reset,
):
    submitted = post_fervis_question(api_client, "best product")
    question = run_worker_until_terminal(api_client, submitted).json()
    completed_at = timezone.now()
    RunWorkItem.objects.filter(run_id=question["latestRunId"]).update(
        created_at=completed_at - timedelta(seconds=10),
        started_at=completed_at - timedelta(seconds=3),
        completed_at=completed_at,
    )

    response = primary_run_detail(api_client, question)

    assert response.json()["durationMs"] == 10_000


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
        question["latestRunId"]
    ]


@pytest.mark.django_db
def test_question_run_detail_is_tenant_scoped(
    api_client,
    fervis_foundation_reset,
    monkeypatch,
):
    adapter_type = type(get_host_api_context().adapter)
    tenant = {"key": "tenant-a"}
    monkeypatch.setattr(
        adapter_type,
        "capture_read_context",
        lambda _adapter, request: ReadContextRef(
            scheme="django_principal",
            key=str(request.user.pk),
            tenant_key=tenant["key"],
        ),
    )
    question = post_fervis_question(api_client, "show sales").json()

    tenant["key"] = "tenant-b"
    response = api_client.get(
        question_run_detail_url(question["questionId"], question["latestRunId"]),
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
        first["latestRunId"],
    )
