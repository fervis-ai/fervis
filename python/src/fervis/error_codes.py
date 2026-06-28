"""Fervis error codes used by framework and provider adapters."""

VALIDATION_ERROR = "validation_error"
MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
INVALID_MODEL_KEY = "invalid_model_key"

INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
READ_CONTEXT_REQUIRED = "read_context_required"

RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

INTERNAL_ERROR = "internal_error"
CONFIGURATION_ERROR = "configuration_error"
LLM_API_ERROR = "llm_api_error"
LLM_API_CONNECTION_ERROR = "llm_api_connection_error"
LLM_API_TIMEOUT = "llm_api_timeout"
LLM_API_BAD_REQUEST = "llm_api_bad_request"
LLM_API_AUTHENTICATION_ERROR = "llm_api_authentication_error"
LLM_API_PERMISSION_ERROR = "llm_api_permission_error"
LLM_API_NOT_FOUND = "llm_api_not_found"
LLM_API_CONFLICT = "llm_api_conflict"
LLM_API_RATE_LIMITED = "llm_api_rate_limited"
LLM_API_INTERNAL_ERROR = "llm_api_internal_error"
LLM_API_RESPONSE_INVALID = "llm_api_response_invalid"
