"""Structured-output error mapping."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fervis import error_codes as common_error_codes

from fervis.model_io.backbone.dto import ToolSpec

from .parsing import raw_tool_output_text


class ModelIoErrorCode(StrEnum):
    PROVIDER_RUNTIME_FAILED = "provider_runtime_failed"
    PROVIDER_CONNECTION_FAILED = "provider_connection_failed"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_BAD_REQUEST = "provider_bad_request"
    PROVIDER_AUTHENTICATION_FAILED = "provider_authentication_failed"
    PROVIDER_PERMISSION_DENIED = "provider_permission_denied"
    PROVIDER_NOT_FOUND = "provider_not_found"
    PROVIDER_CONFLICT = "provider_conflict"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_INTERNAL_ERROR = "provider_internal_error"
    PROVIDER_RESPONSE_INVALID = "provider_response_invalid"
    PROVIDER_CONFIGURATION_FAILED = "provider_configuration_failed"


class RequiredToolOutputError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        output: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        tool_specs: tuple[ToolSpec, ...] = (),
        error_code: str = "",
        error_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.output = dict(output or {})
        self.arguments = dict(arguments or {})
        self.tool_specs = tuple(tool_specs)
        self.raw_output = raw_tool_output_text(self.output)
        self.error_code = str(error_code or "")
        self.error_context = dict(error_context or {})


def provider_error_code(exc: BaseException) -> str:
    code = str(getattr(exc, "code", "") or "")
    return _PROVIDER_ERROR_CODES.get(
        code,
        ModelIoErrorCode.PROVIDER_RUNTIME_FAILED,
    )


def provider_error_context(exc: BaseException) -> dict[str, Any]:
    context = getattr(exc, "context", None)
    return dict(context) if isinstance(context, dict) else {}


_PROVIDER_ERROR_CODES: dict[str, str] = {
    common_error_codes.LLM_API_ERROR: ModelIoErrorCode.PROVIDER_RUNTIME_FAILED,
    common_error_codes.LLM_API_CONNECTION_ERROR: (
        ModelIoErrorCode.PROVIDER_CONNECTION_FAILED
    ),
    common_error_codes.LLM_API_TIMEOUT: ModelIoErrorCode.PROVIDER_TIMEOUT,
    common_error_codes.LLM_API_BAD_REQUEST: ModelIoErrorCode.PROVIDER_BAD_REQUEST,
    common_error_codes.LLM_API_AUTHENTICATION_ERROR: (
        ModelIoErrorCode.PROVIDER_AUTHENTICATION_FAILED
    ),
    common_error_codes.LLM_API_PERMISSION_ERROR: (
        ModelIoErrorCode.PROVIDER_PERMISSION_DENIED
    ),
    common_error_codes.LLM_API_NOT_FOUND: ModelIoErrorCode.PROVIDER_NOT_FOUND,
    common_error_codes.LLM_API_CONFLICT: ModelIoErrorCode.PROVIDER_CONFLICT,
    common_error_codes.LLM_API_RATE_LIMITED: ModelIoErrorCode.PROVIDER_RATE_LIMITED,
    common_error_codes.LLM_API_INTERNAL_ERROR: (
        ModelIoErrorCode.PROVIDER_INTERNAL_ERROR
    ),
    common_error_codes.LLM_API_RESPONSE_INVALID: (
        ModelIoErrorCode.PROVIDER_RESPONSE_INVALID
    ),
    common_error_codes.CONFIGURATION_ERROR: (
        ModelIoErrorCode.PROVIDER_CONFIGURATION_FAILED
    ),
}
