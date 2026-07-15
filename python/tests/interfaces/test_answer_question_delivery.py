from __future__ import annotations

import uuid
import io
import wave

from fervis.delivery.answer_question import (
    AnswerQuestionGenerationError,
    GeneratedAnswerAudio,
)
from fervis.interfaces.common.questions import (
    InterfacePrincipal,
    QuestionInterface,
    QuestionInterfaceResponse,
)


def test_completed_owned_run_returns_disposable_audio() -> None:
    questions = _RunQuestions({"runId": "run-1", "status": "COMPLETED"})
    delivery = _AudioDelivery()
    interface = QuestionInterface(questions=questions, answer_questions=delivery)

    response = interface.answer_computation_question(
        "question-1",
        "run-1",
        principal=_principal(),
        audio_data=_question_wav(),
        content_type="audio/wav",
    )

    assert response.status_code == 200
    assert response.payload == GeneratedAnswerAudio(b"RIFFaudio")
    assert delivery.run_ids == ["run-1"]
    assert questions.accesses == [
        {
            "question_id": "question-1",
            "run_id": "run-1",
            "principal_id": "user-1",
            "tenant_id": "tenant-1",
        }
    ]


def test_audio_is_not_generated_for_an_inaccessible_or_unfinished_run() -> None:
    delivery = _AudioDelivery()
    inaccessible = QuestionInterface(
        questions=_RunQuestions(None),
        answer_questions=delivery,
    ).answer_computation_question(
        "question-1",
        "run-hidden",
        principal=_principal(),
        audio_data=b"",
        content_type="",
    )
    unfinished = QuestionInterface(
        questions=_RunQuestions({"runId": "run-2", "status": "RUNNING"}),
        answer_questions=delivery,
    ).answer_computation_question(
        "question-1",
        "run-2",
        principal=_principal(),
        audio_data=b"",
        content_type="",
    )

    assert inaccessible.status_code == 404
    assert inaccessible.payload["error"]["code"] == "fervis_run_not_found"
    assert unfinished.status_code == 409
    assert unfinished.payload["error"]["code"] == "fervis_run_not_askable"
    assert delivery.run_ids == []


def test_missing_or_failed_provider_does_not_affect_existing_run_access() -> None:
    questions = _RunQuestions({"runId": "run-1", "status": "COMPLETED"})
    missing = QuestionInterface(questions=questions).answer_computation_question(
        "question-1",
        "run-1",
        principal=_principal(),
        audio_data=_question_wav(),
        content_type="audio/wav",
    )
    failed = QuestionInterface(
        questions=questions,
        answer_questions=_FailingAudioDelivery(),
    ).answer_computation_question(
        "question-1",
        "run-1",
        principal=_principal(),
        audio_data=_question_wav(),
        content_type="audio/wav",
    )

    assert missing.status_code == 503
    assert missing.payload["error"]["code"] == ("fervis_answer_question_unavailable")
    assert failed.status_code == 502
    assert failed.payload["error"]["code"] == (
        "fervis_answer_question_generation_failed"
    )
    assert failed.payload["error"]["message"] == (
        "The answer explanation could not be generated."
    )


def test_provider_action_message_is_preserved_for_the_desktop() -> None:
    response = QuestionInterface(
        questions=_RunQuestions({"runId": "run-1", "status": "COMPLETED"}),
        answer_questions=_QuotaAudioDelivery(),
    ).answer_computation_question(
        "question-1",
        "run-1",
        principal=_principal(),
        audio_data=_question_wav(),
        content_type="audio/wav",
    )

    assert response.status_code == 502
    assert response.payload["error"]["message"] == (
        "OpenAI quota is exhausted. Add credits or increase the project budget, "
        "then try again."
    )


def test_invalid_recording_is_rejected_before_the_provider_is_called() -> None:
    delivery = _AudioDelivery()
    response = QuestionInterface(
        questions=_RunQuestions({"runId": "run-1", "status": "COMPLETED"}),
        answer_questions=delivery,
    ).answer_computation_question(
        "question-1",
        "run-1",
        principal=_principal(),
        audio_data=b"not a wav",
        content_type="audio/webm",
    )

    assert response.status_code == 400
    assert response.payload["error"]["code"] == "fervis_invalid_answer_question"
    assert delivery.run_ids == []


def test_truncated_recording_is_rejected_before_the_provider_is_called() -> None:
    delivery = _AudioDelivery()
    response = QuestionInterface(
        questions=_RunQuestions({"runId": "run-1", "status": "COMPLETED"}),
        answer_questions=delivery,
    ).answer_computation_question(
        "question-1",
        "run-1",
        principal=_principal(),
        audio_data=_question_wav()[:-100],
        content_type="audio/wav",
    )

    assert response.status_code == 400
    assert response.payload["error"]["code"] == "fervis_invalid_answer_question"
    assert response.payload["error"]["message"] == "The recording is truncated."
    assert delivery.run_ids == []


def test_fastapi_serves_audio_as_a_non_cacheable_binary_response() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fervis.interfaces.fastapi import fervis_fastapi_router

    app = FastAPI()
    app.include_router(
        fervis_fastapi_router(
            question_interface=_BinaryQuestionInterface(),
            require_read_context=False,
        ),
        prefix="/fervis",
    )

    response = TestClient(app).post(
        "/fervis/questions/question-1/runs/run-1/ask/",
        content=_question_wav(),
        headers={"Content-Type": "audio/wav"},
    )

    _assert_disposable_audio_response(response)


def test_flask_serves_audio_as_a_non_cacheable_binary_response() -> None:
    from flask import Flask
    from fervis.interfaces.flask import fervis_flask_blueprint

    app = Flask(__name__)
    app.register_blueprint(
        fervis_flask_blueprint(
            question_interface=_BinaryQuestionInterface(),
            require_read_context=False,
        ),
        url_prefix="/fervis",
    )

    response = app.test_client().post(
        "/fervis/questions/question-1/runs/run-1/ask/",
        data=_question_wav(),
        content_type="audio/wav",
    )

    assert response.status_code == 200
    assert response.data == b"RIFFaudio"
    assert response.content_type == "audio/wav"
    assert response.headers["Cache-Control"] == "no-store"


def test_django_serves_audio_as_a_non_cacheable_binary_response(monkeypatch) -> None:
    from rest_framework.test import APIRequestFactory
    from fervis.interfaces.django import views

    monkeypatch.setattr(views, "_require_authenticated_subject", lambda request: None)
    monkeypatch.setattr(views, "_principal_from_request", lambda request: _principal())
    monkeypatch.setattr(
        views,
        "django_question_interface",
        lambda: _BinaryQuestionInterface(),
    )
    run_id = uuid.uuid4()

    response = views.QuestionRunAnswerQuestionView.as_view()(
        APIRequestFactory().post(
            f"/fervis/questions/question-1/runs/{run_id}/ask/",
            data=_question_wav(),
            content_type="audio/wav",
            HTTP_ACCEPT="audio/wav",
        ),
        question_id="question-1",
        run_id=run_id,
    )

    assert response.status_code == 200
    assert response.content == b"RIFFaudio"
    assert response["Content-Type"] == "audio/wav"
    assert response["Cache-Control"] == "no-store"


def _assert_disposable_audio_response(response) -> None:
    assert response.status_code == 200
    assert response.content == b"RIFFaudio"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"


def _principal() -> InterfacePrincipal:
    return InterfacePrincipal(principal_id="user-1", tenant_id="tenant-1")


class _RunQuestions:
    def __init__(self, run) -> None:
        self.run = run
        self.accesses: list[dict[str, str]] = []

    def get_question_run(self, question_id, run_id, *, principal):
        self.accesses.append(
            {
                "question_id": question_id,
                "run_id": run_id,
                "principal_id": principal.principal_id,
                "tenant_id": principal.tenant_id,
            }
        )
        return self.run


class _AudioDelivery:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    def answer(self, run_id: str, question) -> GeneratedAnswerAudio:
        assert question.duration_ms == 250
        self.run_ids.append(run_id)
        return GeneratedAnswerAudio(b"RIFFaudio")


class _FailingAudioDelivery:
    def answer(self, run_id: str, question) -> GeneratedAnswerAudio:
        del question
        del run_id
        raise AnswerQuestionGenerationError("provider failed")


class _QuotaAudioDelivery:
    def answer(self, run_id: str, question) -> GeneratedAnswerAudio:
        del question, run_id
        raise AnswerQuestionGenerationError(
            "insufficient_quota.insufficient_quota",
            public_message=(
                "OpenAI quota is exhausted. Add credits or increase the project budget, "
                "then try again."
            ),
        )


class _BinaryQuestionInterface:
    def answer_computation_question(
        self,
        question_id,
        run_id,
        *,
        principal,
        audio_data,
        content_type,
    ):
        del question_id, run_id, principal
        assert audio_data == _question_wav()
        assert content_type == "audio/wav"
        return QuestionInterfaceResponse(
            status_code=200,
            payload=GeneratedAnswerAudio(b"RIFFaudio"),
        )


def _question_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(24_000)
        recording.writeframes(b"\x00\x00" * 6_000)
    return output.getvalue()
