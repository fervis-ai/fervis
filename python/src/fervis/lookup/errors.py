"""Runtime error-code constants."""

from __future__ import annotations


class ErrorCode:
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
    INVALID_MODEL_TOOL_CALL = "invalid_model_tool_call"
    PLANNING_FAILED = "planning_failed"
    PLAN_VALIDATION_FAILED = "plan_validation_failed"
    FACT_PLAN_EXECUTION_FAILED = "fact_plan_execution_failed"
    FRAMEWORK_ADAPTER_FAILED = "framework_adapter_failed"
    LINEAGE_PERSISTENCE_FAILED = "lineage_persistence_failed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    MAX_BUDGET_EXCEEDED = "max_budget_exceeded"
