import json
import time
import uuid

from fervis.interfaces.django.composition import reset_runtime_for_tests
from fervis.interfaces.django.worker import process_run_batch
from fervis.model_io.backbone.factory import build_test_provider_backbone

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "NEEDS_CLARIFICATION"}
FERVIS_SCOPES = "fervis:write fervis:read"


def questions_url() -> str:
    return "/v1/questions/"


def question_detail_url(question_id: str) -> str:
    return f"/v1/questions/{question_id}/"


def question_runs_url(question_id: str) -> str:
    return f"/v1/questions/{question_id}/runs/"


def question_run_detail_url(question_id: str, run_id: str) -> str:
    return f"/v1/questions/{question_id}/runs/{run_id}/"


def post_fervis_question(
    api_client,
    question: str = "list stores",
    *,
    conversation_id: str | None = None,
    payload: dict[str, object] | None = None,
    scopes: str = FERVIS_SCOPES,
):
    request_payload = {
        "question": question,
        **(payload or {}),
    }
    if conversation_id is not None:
        request_payload["conversationId"] = conversation_id
    return api_client.post(
        questions_url(),
        request_payload,
        format="json",
        HTTP_X_REQUESTER_SCOPES=scopes,
    )


def run_worker_until_terminal(api_client, response, *, timeout_seconds=30.0):
    if response.status_code != 202:
        return response
    question_id = response.json().get("questionId")
    if not question_id:
        return response
    worker_id = f"test:{uuid.uuid4()}"
    deadline = time.monotonic() + timeout_seconds
    latest_response = response
    while time.monotonic() < deadline:
        body = latest_response.json()
        if body.get("status") in TERMINAL_STATUSES:
            latest_response.status_code = 202
            return latest_response
        process_run_batch(
            worker_id=worker_id,
            batch_size=1,
            lease_seconds=300,
        )
        latest_response = api_client.get(
            question_detail_url(question_id),
            HTTP_X_REQUESTER_SCOPES="fervis:read",
        )
        if latest_response.status_code != 200:
            return latest_response
    return latest_response


def primary_run_detail(api_client, question: dict):
    return api_client.get(
        question_run_detail_url(question["questionId"], question["latestRunId"]),
        HTTP_X_REQUESTER_SCOPES="fervis:read",
    )


def install_test_model_adapter(adapter, *, provider: str = "anthropic"):
    reset_runtime_for_tests(
        provider_backbone=build_test_provider_backbone(
            provider_name=provider,
            adapters={
                provider: adapter,
                "anthropic": adapter,
                "openai": adapter,
            },
        )
    )
    from fervis.interfaces.django.composition import get_runtime

    return get_runtime()


def first_row_source_evidence_ref(prompt: str) -> str:
    return first_api_row_source_evidence_ref(prompt)


def first_api_row_source_evidence_ref(prompt: str) -> str:
    payload = _prompt_json_section(
        prompt,
        start_label="Relation catalog:",
        end_label="Catalog selection:",
    )
    for source in payload.get("row_sources", ()):
        if source.get("kind") != "api_read":
            continue
        evidence_ref = str(source.get("evidence_ref") or "")
        if evidence_ref:
            return evidence_ref
    raise AssertionError("fact-plan prompt did not expose API row-source evidence")


def _prompt_json_section(
    prompt: str,
    *,
    start_label: str,
    end_label: str,
) -> dict:
    try:
        start = prompt.index(start_label) + len(start_label)
        end = prompt.index(end_label, start)
    except ValueError as exc:
        raise AssertionError("fact-plan prompt missing expected section") from exc
    return json.loads(prompt[start:end].strip())
