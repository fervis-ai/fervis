"""Fervis-owned API error hierarchy.

This module intentionally has no Django/FastAPI/Flask dependency. Framework
adapters may translate these errors into host responses, but provider and core
runtime code should not import host application error modules.
"""

from __future__ import annotations

from typing import Any

from . import error_codes as codes


class APIError(Exception):
    """Base exception for structured Fervis errors."""

    error_type = "system"
    status_code: int | None = None
    retryable = False

    def __init__(
        self,
        code: str,
        message: str,
        *,
        developer_message: str | None = None,
        context: dict[str, Any] | None = None,
        details: list[dict[str, Any]] | None = None,
        retry_after: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.developer_message = developer_message or message
        self.context = dict(context or {})
        self.details = list(details or [])
        self.retry_after = retry_after
        super().__init__(message)

    @classmethod
    def configuration_error(
        cls, message: str = "Service configuration is invalid."
    ) -> "APIError":
        return build_error(cls, codes.CONFIGURATION_ERROR, message)

    @classmethod
    def internal_error(cls) -> "APIError":
        return build_error(cls, codes.INTERNAL_ERROR, "An unexpected error occurred.")


class Validation(APIError):
    error_type = "validation"
    status_code = 400

    @classmethod
    def request_validation_failed(
        cls,
        *,
        message: str = "Request validation failed.",
        details: list[dict[str, Any]] | None = None,
    ) -> "Validation":
        return build_error(
            cls,
            codes.VALIDATION_ERROR,
            message,
            details=details or [],
        )

    @classmethod
    def invalid_model_key(
        cls, *, allowed_models: list[str], message: str = "Invalid model key."
    ) -> "Validation":
        return build_error(
            cls,
            codes.INVALID_MODEL_KEY,
            message,
            context={"allowed_models": list(allowed_models)},
        )


class Authorization(APIError):
    error_type = "authorization"
    status_code = 403

    @classmethod
    def insufficient_permissions(cls) -> "Authorization":
        return build_error(
            cls,
            codes.INSUFFICIENT_PERMISSIONS,
            "You do not have permission to perform this action.",
        )


class Authentication(APIError):
    error_type = "authorization"
    status_code = 401

    @classmethod
    def read_context_required(
        cls,
        message: str = "Fervis could not capture an authenticated read context.",
    ) -> "Authentication":
        return build_error(cls, codes.READ_CONTEXT_REQUIRED, message)


class NotFound(APIError):
    error_type = "not_found"
    status_code = 404

    @classmethod
    def for_resource(
        cls,
        resource_type: str,
        resource_id: str | int | None = "unknown",
        *,
        message: str | None = None,
    ) -> "NotFound":
        return build_error(
            cls,
            f"{resource_type}_not_found",
            message or f"{resource_type.replace('_', ' ').title()} not found.",
            developer_message=f"{resource_type} {resource_id} does not exist",
            context={"resource_type": resource_type, "resource_id": str(resource_id)},
        )


class RateLimit(APIError):
    error_type = "rate_limit"
    status_code = 429
    retryable = True

    @classmethod
    def llm_api_rate_limited(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ) -> "RateLimit":
        return build_error(
            cls,
            codes.LLM_API_RATE_LIMITED,
            "LLM provider rate limit exceeded.",
            context=_provider_context(provider, reason, error_class, context),
            retry_after=retry_after,
        )


class Unavailable(APIError):
    error_type = "system"
    status_code = 502
    retryable = True

    @classmethod
    def llm_api_error(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_ERROR,
            message="LLM provider is currently unavailable.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_connection_error(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_CONNECTION_ERROR,
            message="Could not connect to the LLM provider.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_timeout(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_TIMEOUT,
            message="LLM provider timed out.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_bad_request(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_BAD_REQUEST,
            message="LLM request was rejected by the provider.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_authentication_error(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_AUTHENTICATION_ERROR,
            message="LLM provider authentication failed.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_permission_error(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_PERMISSION_ERROR,
            message="LLM provider denied access to the requested resource.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_not_found(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_NOT_FOUND,
            message="LLM provider resource was not found.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_conflict(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_CONFLICT,
            message="LLM provider reported a request conflict.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_internal_error(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_INTERNAL_ERROR,
            message="LLM provider encountered an internal error.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def llm_api_response_invalid(
        cls,
        *,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return cls._llm_error(
            code=codes.LLM_API_RESPONSE_INVALID,
            message="LLM provider returned an invalid response payload.",
            provider=provider,
            reason=reason,
            error_class=error_class,
            context=context,
        )

    @classmethod
    def _llm_error(
        cls,
        *,
        code: str,
        message: str,
        provider: str,
        reason: str | None = None,
        error_class: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Unavailable":
        return build_error(
            cls,
            code,
            message,
            context=_provider_context(provider, reason, error_class, context),
        )


def build_error(
    exception_cls: type[APIError],
    code: str,
    message: str,
    *,
    context: dict[str, Any] | None = None,
    developer_message: str | None = None,
    details: list[dict[str, Any]] | None = None,
    retry_after: int | None = None,
) -> APIError:
    return exception_cls(
        code=code,
        message=message,
        developer_message=developer_message,
        context=context,
        details=details,
        retry_after=retry_after,
    )


def _provider_context(
    provider: str,
    reason: str | None,
    error_class: str | None,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "provider": provider,
        "reason": reason,
        "error_class": error_class,
    }
    if context:
        payload.update(context)
    return payload
