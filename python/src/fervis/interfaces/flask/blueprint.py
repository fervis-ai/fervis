"""Flask Blueprint for the Fervis question lifecycle."""

from __future__ import annotations

from typing import Any

from fervis.interfaces.common.questions import (
    QuestionInterface,
    QuestionInterfaceValidationError,
)
from fervis.interfaces.common.read_contexts import ReadContextCaptureError

from .principal import principal_from_request


def fervis_flask_blueprint(
    *,
    question_interface: QuestionInterface | None = None,
    question_interface_factory=None,
    read_context_capture=None,
    delegated_credential_capture=None,
    require_read_context: bool = True,
):
    from flask import Blueprint, jsonify, request

    blueprint = Blueprint("fervis", __name__)

    def request_principal():
        return _principal_from_request(
            request,
            read_context_capture=read_context_capture,
            delegated_credential_capture=delegated_credential_capture,
            require_read_context=require_read_context,
        )

    def json_response(status_code: int, payload: Any):
        response = jsonify(payload)
        response.status_code = status_code
        return response

    def questions() -> QuestionInterface:
        if question_interface is not None:
            return question_interface
        if question_interface_factory is None:
            raise RuntimeError("Flask Fervis blueprint requires a QuestionInterface.")
        return question_interface_factory()

    @blueprint.route("/", methods=["GET"])
    def runtime_status():
        return {"runtime": "fervis", "status": "ok"}

    @blueprint.route("/conversations/", methods=["GET"])
    def list_conversations():
        response = questions().list_conversations(
            principal=request_principal(),
        )
        return json_response(response.status_code, response.payload)

    @blueprint.route("/questions/", methods=["POST"])
    def create_question():
        try:
            response = questions().create_question(
                _json_payload(request),
                principal=request_principal(),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except QuestionInterfaceValidationError as exc:
            return _invalid_request_response(exc)
        return json_response(response.status_code, response.payload)

    @blueprint.route("/questions/<question_id>/", methods=["GET"])
    def get_question(question_id: str):
        response = questions().get_question(
            question_id,
            principal=request_principal(),
        )
        return json_response(response.status_code, response.payload)

    @blueprint.route("/questions/<question_id>/runs/", methods=["GET"])
    def list_question_runs(question_id: str):
        response = questions().list_question_runs(
            question_id,
            principal=request_principal(),
        )
        return json_response(response.status_code, response.payload)

    @blueprint.route("/questions/<question_id>/runs/", methods=["POST"])
    def create_question_run(question_id: str):
        try:
            response = questions().create_question_run(
                question_id,
                _json_payload(request),
                principal=request_principal(),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except QuestionInterfaceValidationError as exc:
            return _invalid_request_response(exc)
        return json_response(response.status_code, response.payload)

    @blueprint.route("/questions/<question_id>/runs/<run_id>/", methods=["GET"])
    def get_question_run(question_id: str, run_id: str):
        response = questions().get_question_run(
            question_id,
            run_id,
            principal=request_principal(),
        )
        return json_response(response.status_code, response.payload)

    return blueprint


def _json_payload(request) -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _principal_from_request(
    request,
    *,
    read_context_capture,
    delegated_credential_capture,
    require_read_context: bool,
):
    try:
        return principal_from_request(
            request,
            read_context_capture=read_context_capture,
            delegated_credential_capture=delegated_credential_capture,
            require_read_context=require_read_context,
        )
    except ReadContextCaptureError as exc:
        raise _read_context_required_error(exc) from exc


def _read_context_required_error(exc: ReadContextCaptureError):
    from flask import abort

    abort(
        _error_response(
            401,
            {
                "error": {
                    "type": "authorization",
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": False,
                }
            },
        )
    )


def _invalid_request_response(exc: QuestionInterfaceValidationError):
    return _error_response(
        400,
        {
            "error": {
                "type": "validation",
                "code": "invalid_request",
                "message": exc.message,
                "retryable": False,
                "details": [
                    {
                        "field": exc.field,
                        "code": exc.code,
                        "message": exc.message,
                    }
                ],
            }
        },
    )


def _error_response(status_code: int, payload: dict[str, object]):
    from flask import jsonify

    response = jsonify(payload)
    response.status_code = status_code
    return response
