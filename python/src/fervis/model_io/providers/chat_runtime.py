"""Shared runtime primitives for chat-completion model providers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import os
import queue
from typing import Any, Callable

from fervis.observability.usage_types import CostSource
from fervis.observability.usage_types import UsageKey
from fervis.model_io.backbone.dto import ProviderRunRequest, ProviderRunResult
from fervis.model_io.backbone.dto import ToolSpec, ProviderOutputMode
from fervis.model_io.pricing import ModelPricing, resolve_model_pricing
from fervis import errors as api_errors


@dataclass(frozen=True)
class ChatProviderConfig:
    provider_name: str
    model_name: str
    api_key_env_var: str
    sdk_name: str
    base_url_env_var: str | None = None
    default_base_url: str | None = None
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0
    thinking_cost_per_million_tokens: float = 0.0
    pricing_version: str = ""
    temperature: float = 0.0
    max_output_tokens_parameter: str = "max_tokens"

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env_var)

    @property
    def base_url(self) -> str | None:
        if not self.base_url_env_var:
            return self.default_base_url
        return os.getenv(self.base_url_env_var, self.default_base_url or "")

    @property
    def has_configured_pricing(self) -> bool:
        rates = (
            Decimal(str(self.input_cost_per_million_tokens)),
            Decimal(str(self.output_cost_per_million_tokens)),
            Decimal(str(self.thinking_cost_per_million_tokens)),
        )
        return bool(self.pricing_version.strip()) and any(rate > 0 for rate in rates)


class ConfiguredChatModelAdapter:
    def __init__(self, *, loop_runtime: Any, config: ChatProviderConfig):
        self.loop_runtime = loop_runtime
        self.config = config
        self.provider_name = config.provider_name

    def generate(
        self,
        *,
        model_id: str | None = None,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str,
        output_mode: ProviderOutputMode = ProviderOutputMode.TEXT,
        tool_specs: tuple[ToolSpec, ...] = (),
    ) -> dict[str, Any]:
        result = self.loop_runtime.run(
            ProviderRunRequest(
                provider=self.config.provider_name,
                model_id=model_id,
                prompt=prompt,
                max_thinking_tokens=max_thinking_tokens,
                system_prompt=system_prompt,
                output_mode=output_mode,
                tool_specs=tool_specs,
            )
        )
        return {
            "provider": result.provider,
            "answer": result.answer,
            "toolRequests": [],
            "usage": dict(result.usage),
            "raw": dict(result.raw_payload),
        }


class ConfiguredChatLoopRuntime:
    def __init__(self, *, config: ChatProviderConfig):
        self.config = config

    def sdk_available(self) -> bool:
        raise NotImplementedError

    def worker(self) -> ProviderWorker:
        raise NotImplementedError

    def request_payload(self, request: ProviderRunRequest) -> Any:
        raise NotImplementedError

    def _sdk_status(self) -> str:
        return provider_sdk_status(self.config, sdk_available=self.sdk_available())

    def run(self, request: ProviderRunRequest) -> ProviderRunResult:
        provider_status = self._sdk_status()
        if provider_status != "enabled":
            raise provider_unavailable_error(
                self.config,
                reason=provider_status,
                error_class="ProviderConfigurationError",
            )
        try:
            payload = run_provider_worker(
                self.worker(),
                self.request_payload(request),
            )
        except ProviderExecutionError as exc:
            raise provider_unavailable_error(
                self.config,
                reason=exc.reason,
                error_class=exc.error_class,
                context=exc.context,
            ) from exc
        try:
            return build_provider_run_result(
                self.config,
                model_name=request.model_id or self.config.model_name,
                answer=str(payload.get("answer") or ""),
                input_tokens=int(payload.get("inputTokens") or 0),
                output_tokens=int(payload.get("outputTokens") or 0),
                thinking_tokens=int(payload.get("thinkingTokens") or 0),
                usage_details=_dict_or_empty(payload.get("usageDetails")),
            )
        except ProviderExecutionError as exc:
            raise provider_unavailable_error(
                self.config,
                reason=exc.reason,
                error_class=exc.error_class,
                context=exc.context,
            ) from exc


class ProviderExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        error_class: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.error_class = error_class
        self.reason = reason
        self.context = context or {}
        super().__init__(f"{error_class}: {reason}" if reason else error_class)


ProviderWorker = Callable[[Any, Any], None]


def provider_sdk_status(config: ChatProviderConfig, *, sdk_available: bool) -> str:
    if not sdk_available:
        return "missing_sdk"
    if not config.api_key:
        return f"missing_{config.api_key_env_var.lower()}"
    return "enabled"


def provider_unavailable_error(
    config: ChatProviderConfig,
    *,
    reason: str,
    error_class: str,
    context: dict[str, Any] | None = None,
) -> api_errors.APIError:
    normalized = str(error_class or "")
    provider_metadata = (
        dict((context or {}).get("provider_metadata") or {})
        if isinstance(context, dict)
        else {}
    )
    status_code = _int_or_none(provider_metadata.get("statusCode"))
    error_type = str(provider_metadata.get("errorType") or "").strip().lower()

    if normalized == "ProviderConfigurationError":
        return api_errors.APIError.configuration_error(
            f"{config.provider_name} provider configuration is invalid."
        )
    if normalized in {"APIResponseValidationError"}:
        return api_errors.Unavailable.llm_api_response_invalid(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if normalized in {"APIConnectionError"}:
        return api_errors.Unavailable.llm_api_connection_error(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized
        in {
            "TimeoutError",
            "APITimeoutError",
            "TimeoutException",
        }
        or status_code == 504
        or error_type == "timeout_error"
    ):
        return api_errors.Unavailable.llm_api_timeout(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized in {"BadRequestError", "UnprocessableEntityError"}
        or status_code
        in {
            400,
            413,
            422,
        }
        or error_type in {"invalid_request_error", "request_too_large"}
    ):
        return api_errors.Unavailable.llm_api_bad_request(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized == "AuthenticationError"
        or status_code == 401
        or error_type
        in {
            "authentication_error",
            "billing_error",
        }
    ):
        return api_errors.Unavailable.llm_api_authentication_error(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized == "PermissionDeniedError"
        or status_code == 403
        or error_type == "permission_error"
    ):
        return api_errors.Unavailable.llm_api_permission_error(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized == "NotFoundError"
        or status_code == 404
        or error_type == "not_found_error"
    ):
        return api_errors.Unavailable.llm_api_not_found(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if normalized == "ConflictError" or status_code == 409:
        return api_errors.Unavailable.llm_api_conflict(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized == "RateLimitError"
        or status_code in {429, 529}
        or error_type
        in {
            "rate_limit_error",
            "overloaded_error",
        }
    ):
        return api_errors.RateLimit.llm_api_rate_limited(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    if (
        normalized == "InternalServerError"
        or (status_code is not None and status_code >= 500)
        or error_type == "api_error"
    ):
        return api_errors.Unavailable.llm_api_internal_error(
            provider=config.provider_name,
            reason=reason,
            error_class=error_class,
            context=context,
        )
    return api_errors.Unavailable.llm_api_error(
        provider=config.provider_name,
        reason=reason,
        error_class=error_class,
        context=context,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def build_provider_run_result(
    config: ChatProviderConfig,
    *,
    model_name: str | None = None,
    answer: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    usage_details: dict[str, Any] | None = None,
) -> ProviderRunResult:
    if input_tokens < 1:
        raise ProviderExecutionError(
            error_class="APIResponseValidationError",
            reason="provider response did not include input token usage",
        )
    if output_tokens < 0 or thinking_tokens < 0:
        raise ProviderExecutionError(
            error_class="APIResponseValidationError",
            reason="provider response included invalid token usage",
        )
    actual_model_name = model_name or config.model_name
    pricing = _pricing_for_model(config, actual_model_name)
    if not pricing.priced:
        return ProviderRunResult(
            provider=config.provider_name,
            answer=answer,
            usage={
                UsageKey.INPUT_TOKENS: input_tokens,
                UsageKey.OUTPUT_TOKENS: output_tokens,
                UsageKey.THINKING_TOKENS: thinking_tokens,
                UsageKey.COST_USD: 0,
                UsageKey.COST_SOURCE: pricing.cost_source,
                UsageKey.PRICING_VERSION: pricing.pricing_version,
            },
            raw_payload={
                "provider": config.provider_name,
                "sdk": config.sdk_name,
                "providerStatus": "enabled",
                "model": actual_model_name,
            },
        )
    input_cost = _token_cost(input_tokens, pricing.input_cost_per_million_tokens)
    output_cost = _token_cost(output_tokens, pricing.output_cost_per_million_tokens)
    thinking_cost = _token_cost(
        thinking_tokens,
        pricing.thinking_cost_per_million_tokens,
    )
    total_cost = input_cost + output_cost + thinking_cost
    usage = {
        UsageKey.INPUT_TOKENS: input_tokens,
        UsageKey.OUTPUT_TOKENS: output_tokens,
        UsageKey.THINKING_TOKENS: thinking_tokens,
        UsageKey.COST_USD: float(total_cost),
        UsageKey.INPUT_COST_USD: float(input_cost),
        UsageKey.OUTPUT_COST_USD: float(output_cost),
        UsageKey.THINKING_COST_USD: float(thinking_cost),
        UsageKey.COST_SOURCE: pricing.cost_source,
        UsageKey.PRICING_VERSION: pricing.pricing_version,
    }
    model_subcalls = _priced_model_subcalls(
        pricing,
        _dict_or_empty(usage_details).get(UsageKey.MODEL_SUBCALLS),
    )
    if model_subcalls:
        usage[UsageKey.MODEL_SUBCALLS] = model_subcalls
    return ProviderRunResult(
        provider=config.provider_name,
        answer=answer,
        usage=usage,
        raw_payload={
            "provider": config.provider_name,
            "sdk": config.sdk_name,
            "providerStatus": "enabled",
            "model": actual_model_name,
        },
    )


def _pricing_for_model(config: ChatProviderConfig, model_name: str) -> ModelPricing:
    if config.has_configured_pricing:
        return ModelPricing(
            input_cost_per_million_tokens=config.input_cost_per_million_tokens,
            output_cost_per_million_tokens=config.output_cost_per_million_tokens,
            thinking_cost_per_million_tokens=config.thinking_cost_per_million_tokens,
            pricing_version=config.pricing_version,
            cost_source=CostSource.CONFIGURED_PROVIDER_PRICING,
        )
    return resolve_model_pricing(provider=config.provider_name, model_key=model_name)


def _token_cost(tokens: int, rate_per_million_tokens: float) -> Decimal:
    return (
        Decimal(int(tokens))
        * Decimal(str(rate_per_million_tokens))
        / Decimal(1_000_000)
    ).quantize(Decimal("0.000001"))


def _priced_model_subcalls(
    pricing: ModelPricing,
    raw_subcalls: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_subcalls, list):
        return []
    output: list[dict[str, Any]] = []
    for raw in raw_subcalls:
        if not isinstance(raw, dict):
            continue
        input_tokens = _nonnegative_int(raw.get(UsageKey.INPUT_TOKENS))
        output_tokens = _nonnegative_int(raw.get(UsageKey.OUTPUT_TOKENS))
        thinking_tokens = _nonnegative_int(raw.get(UsageKey.THINKING_TOKENS))
        input_cost = _token_cost(input_tokens, pricing.input_cost_per_million_tokens)
        output_cost = _token_cost(output_tokens, pricing.output_cost_per_million_tokens)
        thinking_cost = _token_cost(
            thinking_tokens,
            pricing.thinking_cost_per_million_tokens,
        )
        item = dict(raw)
        item.update(
            {
                UsageKey.INPUT_TOKENS: input_tokens,
                UsageKey.OUTPUT_TOKENS: output_tokens,
                UsageKey.THINKING_TOKENS: thinking_tokens,
                UsageKey.COST_USD: float(input_cost + output_cost + thinking_cost),
                UsageKey.INPUT_COST_USD: float(input_cost),
                UsageKey.OUTPUT_COST_USD: float(output_cost),
                UsageKey.THINKING_COST_USD: float(thinking_cost),
            }
        )
        output.append(item)
    return output


def _nonnegative_int(value: Any) -> int:
    try:
        integer = int(str(value))
    except (TypeError, ValueError):
        return 0
    return max(0, integer)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def chat_json_system_prompt(runtime_system_prompt: str = "") -> str:
    return (
        _prefixed_runtime_system_prompt(runtime_system_prompt)
        + "Return exactly one raw JSON object and nothing else. "
        f"{chat_runtime_base_system_prompt()} "
        "If the user prompt is a JSON array of Fervis runtime messages, follow "
        "the inner system prompt and return exactly one "
        'tool call object: {"tool":"tool_name","arguments":{...}}. '
    )


def chat_tool_system_prompt(runtime_system_prompt: str = "") -> str:
    return (
        _prefixed_runtime_system_prompt(runtime_system_prompt)
        + "Use the required provider-native tool call exactly once. "
        "Return no prose or markdown outside the tool call. "
        "Do not answer the business question directly during tool-call turns."
    )


def _prefixed_runtime_system_prompt(runtime_system_prompt: str) -> str:
    text = runtime_system_prompt.strip()
    return f"{text} " if text else ""


def chat_runtime_base_system_prompt() -> str:
    return (
        "No markdown fences, no commentary, no prose outside the requested output shape. "
        "Do not answer the business question directly during tool-call turns. "
        "If the user prompt contains a responseFormat object, return that requested response shape."
    )


def provider_timeout_seconds() -> float:
    return _environment_float(
        "FERVIS_PROVIDER_TIMEOUT_SECONDS",
        default=120.0,
        minimum=1.0,
    )


def provider_max_retries() -> int:
    return _environment_int(
        "FERVIS_PROVIDER_MAX_RETRIES",
        default=0,
        minimum=0,
    )


def provider_max_output_tokens() -> int:
    return _environment_int(
        "FERVIS_PROVIDER_MAX_OUTPUT_TOKENS",
        default=4096,
        minimum=1024,
        maximum=8192,
    )


def _environment_float(name: str, *, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise _invalid_provider_setting(name) from exc
    if value < minimum:
        raise _invalid_provider_setting(name)
    return value


def _environment_int(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise _invalid_provider_setting(name) from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise _invalid_provider_setting(name)
    return value


def _invalid_provider_setting(name: str) -> ProviderExecutionError:
    return ProviderExecutionError(
        error_class="ProviderConfigurationError",
        reason=f"{name} is invalid.",
    )


def run_provider_worker(
    worker: ProviderWorker,
    payload: Any,
) -> dict[str, Any]:
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
    _run_provider_worker(worker, payload, result_queue)

    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise ProviderExecutionError(
            error_class="ProviderNoResultError",
            reason="Provider returned no result.",
            context=_provider_worker_no_result_context(),
        ) from exc
    if not result.get("ok"):
        raise ProviderExecutionError(
            error_class=str(result.get("errorClass") or "ProviderExecutionError"),
            reason=str(result.get("error") or "Provider request failed."),
            context=_provider_error_context(result),
        )
    return dict(result)


def _run_provider_worker(
    worker: ProviderWorker,
    payload: Any,
    result_queue: queue.Queue[dict[str, Any]],
) -> None:
    try:
        worker(payload, result_queue)
    except BaseException as exc:
        _put_provider_worker_result(result_queue, provider_error_payload(exc))


def _put_provider_worker_result(
    result_queue: queue.Queue[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    try:
        result_queue.put_nowait(result)
    except queue.Full:
        return


def provider_error_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ProviderExecutionError):
        execution_error = {
            "ok": False,
            "errorClass": exc.error_class,
            "error": exc.reason,
        }
        if exc.context:
            execution_error["context"] = dict(exc.context)
        return execution_error
    payload: dict[str, Any] = {
        "ok": False,
        "errorClass": exc.__class__.__name__,
        "error": str(exc),
    }
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error_body = body.get("error")
        if isinstance(error_body, dict):
            error_type = error_body.get("type")
            if error_type:
                payload["errorType"] = str(error_type)
        request_id = body.get("request_id")
        if request_id:
            payload["requestId"] = str(request_id)
    for attr, key in (
        ("status_code", "statusCode"),
        ("request_id", "requestId"),
        ("code", "errorCode"),
        ("type", "errorType"),
    ):
        value = getattr(exc, attr, None)
        if value is not None:
            payload[key] = str(value)
    return payload


def _provider_worker_no_result_context() -> dict[str, Any]:
    return {"provider_metadata": {"worker_completion": "no_result"}}


def _provider_error_context(result: dict[str, Any]) -> dict[str, Any]:
    context = result.get("context")
    raw_context = dict(context) if isinstance(context, dict) else {}
    metadata = {
        key: result[key]
        for key in ("statusCode", "requestId", "errorCode", "errorType")
        if key in result
    }
    existing_metadata = raw_context.get("provider_metadata")
    if isinstance(existing_metadata, dict):
        metadata.update(
            {str(key): str(value) for key, value in existing_metadata.items()}
        )
    metadata.update(
        {
            str(key): _provider_metadata_value(value)
            for key, value in raw_context.items()
            if key != "provider_metadata"
        }
    )
    if not metadata:
        return {}
    return {"provider_metadata": metadata}


def _provider_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped
