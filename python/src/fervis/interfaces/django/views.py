"""HTTP views for the Fervis question API."""

from __future__ import annotations

import uuid

from rest_framework.response import Response
from rest_framework.views import APIView

from fervis import errors as api_errors
from fervis.interfaces.common.questions import (
    QuestionInterfaceResponse,
    QuestionInterfaceValidationError,
)
from fervis.interfaces.common.read_contexts import ReadContextCaptureError

from .principal import principal_from_request
from .question_interface import django_question_interface
from .security import MissingAuthenticationError, require_fervis_access
from .throttles import FervisQuestionThrottle


class FervisAPIView(APIView):
    permission_classes = []
    throttle_classes = [FervisQuestionThrottle]

    def handle_exception(self, exc):
        if isinstance(exc, api_errors.APIError):
            return _api_error_response(exc, self.request)
        return super().handle_exception(exc)


def _require_authenticated_subject(request) -> None:
    try:
        require_fervis_access(request)
    except MissingAuthenticationError as exc:
        raise api_errors.Authentication.read_context_required() from exc


def _principal_from_request(request):
    try:
        return principal_from_request(request)
    except ReadContextCaptureError as exc:
        raise api_errors.Authentication.read_context_required(exc.message) from exc


class FervisRuntimeStatusView(FervisAPIView):
    def get(self, request):
        _require_authenticated_subject(request)
        return Response({"runtime": "fervis", "status": "ok"})


class ConversationListView(FervisAPIView):
    def get(self, request):
        _require_authenticated_subject(request)

        response = django_question_interface().list_conversations(
            principal=_principal_from_request(request),
        )
        return _question_interface_response(response, request)


class QuestionCreateView(FervisAPIView):
    def post(self, request):
        _require_authenticated_subject(request)

        try:
            response = django_question_interface().create_question(
                request.data,
                principal=_principal_from_request(request),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except QuestionInterfaceValidationError as exc:
            raise api_errors.Validation.request_validation_failed(
                details=[
                    {
                        "field": exc.field,
                        "code": exc.code,
                        "message": exc.message,
                    }
                ]
            ) from exc

        return _question_interface_response(response, request)


class QuestionDetailView(FervisAPIView):
    def get(self, request, question_id: str):
        _require_authenticated_subject(request)

        response = django_question_interface().get_question(
            str(question_id),
            principal=_principal_from_request(request),
        )
        if response.status_code == 404:
            raise api_errors.NotFound.for_resource("fervis_question", question_id)
        return _question_interface_response(response, request)


class QuestionRunListView(FervisAPIView):
    def get(self, request, question_id: str):
        _require_authenticated_subject(request)

        response = django_question_interface().list_question_runs(
            str(question_id),
            principal=_principal_from_request(request),
        )
        if response.status_code == 404:
            raise api_errors.NotFound.for_resource("fervis_question", question_id)
        return _question_interface_response(response, request)

    def post(self, request, question_id: str):
        _require_authenticated_subject(request)

        try:
            response = django_question_interface().create_question_run(
                str(question_id),
                request.data,
                principal=_principal_from_request(request),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except QuestionInterfaceValidationError as exc:
            raise api_errors.Validation.request_validation_failed(
                details=[
                    {
                        "field": exc.field,
                        "code": exc.code,
                        "message": exc.message,
                    }
                ]
            ) from exc
        return _question_interface_response(response, request)


class QuestionRunDetailView(FervisAPIView):
    def get(self, request, question_id: str, run_id: str):
        _require_authenticated_subject(request)

        response = django_question_interface().get_question_run(
            str(question_id),
            str(run_id),
            principal=_principal_from_request(request),
        )
        if response.status_code == 404:
            raise api_errors.NotFound.for_resource("fervis_run", run_id)
        return _question_interface_response(response, request)


def _question_interface_response(
    response: QuestionInterfaceResponse,
    request,
) -> Response:
    payload = response.payload
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error = dict(payload["error"])
        error.setdefault("developer_message", error.get("message", ""))
        error.setdefault("retry_after", None)
        error.setdefault("request_id", _request_id(request))
        payload = {**payload, "error": error}
    return Response(payload, status=response.status_code)


def _api_error_response(error: api_errors.APIError, request) -> Response:
    payload = {
        "error": {
            "type": error.error_type,
            "code": error.code,
            "message": error.message,
            "developer_message": error.developer_message,
            "retryable": error.retryable,
            "retry_after": error.retry_after,
            "request_id": _request_id(request),
            "details": error.details,
            "context": error.context,
        }
    }
    return Response(payload, status=error.status_code or 500)


def _request_id(request) -> str:
    value = str(getattr(request, "request_id", "") or "").strip()
    if value:
        return value
    header_value = str(request.headers.get("X-Request-ID", "") or "").strip()
    if header_value:
        return header_value
    generated = str(uuid.uuid4())
    setattr(request, "request_id", generated)
    return generated
