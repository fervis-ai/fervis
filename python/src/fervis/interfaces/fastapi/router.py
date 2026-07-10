"""FastAPI routes for the Fervis question lifecycle."""

from typing import Any

from fervis.interfaces.common.questions import (
    QuestionInterface,
    QuestionInterfaceValidationError,
)
from fervis.interfaces.common.read_contexts import ReadContextCaptureError

from .principal import principal_from_request


def fervis_fastapi_router(
    *,
    question_interface: QuestionInterface,
    read_context_capture=None,
    delegated_credential_capture=None,
    principal_dependency=None,
    principal_id_attr: str = "id",
    require_read_context: bool = True,
):
    from fastapi import APIRouter, Depends, Request
    from fastapi.responses import JSONResponse

    router = APIRouter()
    principal_dependency = principal_dependency or _anonymous_principal

    def request_principal(request: Request, dependency_principal):
        return _principal_from_request(
            request,
            read_context_capture=read_context_capture,
            delegated_credential_capture=delegated_credential_capture,
            dependency_principal=dependency_principal,
            principal_id_attr=principal_id_attr,
            require_read_context=require_read_context,
        )

    @router.get("/")
    def runtime_status() -> dict[str, str]:
        return {"runtime": "fervis", "status": "ok"}

    @router.get("/conversations/")
    def list_conversations(
        request: Request,
        dependency_principal=Depends(principal_dependency),
    ):
        response = question_interface.list_conversations(
            principal=request_principal(request, dependency_principal),
        )
        return JSONResponse(status_code=response.status_code, content=response.payload)

    @router.post("/questions/")
    def create_question(
        payload: dict[str, Any],
        request: Request,
        dependency_principal=Depends(principal_dependency),
    ):
        try:
            response = question_interface.create_question(
                payload,
                principal=request_principal(request, dependency_principal),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except QuestionInterfaceValidationError as exc:
            raise _invalid_request_error(exc) from exc
        return JSONResponse(status_code=response.status_code, content=response.payload)

    @router.get("/questions/{question_id}/")
    def get_question(
        question_id: str,
        request: Request,
        dependency_principal=Depends(principal_dependency),
    ):
        response = question_interface.get_question(
            question_id,
            principal=request_principal(request, dependency_principal),
        )
        return JSONResponse(status_code=response.status_code, content=response.payload)

    @router.get("/questions/{question_id}/runs/")
    def list_question_runs(
        question_id: str,
        request: Request,
        dependency_principal=Depends(principal_dependency),
    ):
        response = question_interface.list_question_runs(
            question_id,
            principal=request_principal(request, dependency_principal),
        )
        return JSONResponse(status_code=response.status_code, content=response.payload)

    @router.post("/questions/{question_id}/runs/")
    def create_question_run(
        question_id: str,
        payload: dict[str, Any],
        request: Request,
        dependency_principal=Depends(principal_dependency),
    ):
        try:
            response = question_interface.create_question_run(
                question_id,
                payload,
                principal=request_principal(request, dependency_principal),
                idempotency_key=request.headers.get("Idempotency-Key"),
            )
        except QuestionInterfaceValidationError as exc:
            raise _invalid_request_error(exc) from exc
        return JSONResponse(status_code=response.status_code, content=response.payload)

    @router.get("/questions/{question_id}/runs/{run_id}/")
    def get_question_run(
        question_id: str,
        run_id: str,
        request: Request,
        dependency_principal=Depends(principal_dependency),
    ):
        response = question_interface.get_question_run(
            question_id,
            run_id,
            principal=request_principal(request, dependency_principal),
        )
        return JSONResponse(status_code=response.status_code, content=response.payload)

    return router


def _anonymous_principal() -> None:
    return None


def _principal_from_request(
    request,
    *,
    read_context_capture,
    delegated_credential_capture,
    dependency_principal,
    principal_id_attr: str,
    require_read_context: bool,
):
    try:
        return principal_from_request(
            request,
            read_context_capture=read_context_capture,
            delegated_credential_capture=delegated_credential_capture,
            dependency_principal=dependency_principal,
            principal_id_attr=principal_id_attr,
            require_read_context=require_read_context,
        )
    except ReadContextCaptureError as exc:
        raise _read_context_required_error(exc) from exc


def _read_context_required_error(exc: ReadContextCaptureError):
    from fastapi import HTTPException

    return HTTPException(
        status_code=401,
        detail={
            "error": {
                "type": "authorization",
                "code": exc.code,
                "message": exc.message,
                "retryable": False,
            }
        },
    )


def _invalid_request_error(exc: QuestionInterfaceValidationError):
    from fastapi import HTTPException

    return HTTPException(
        status_code=400,
        detail={
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
