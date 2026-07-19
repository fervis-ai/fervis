"""OpenAI-compatible provider adapters for Moonshot, DeepSeek, and Qwen."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import ModuleType
from fervis.types.enums import StrEnum
from typing import Any

from fervis.model_io.backbone.dto import (
    ToolSpec,
    ProviderOutputMode,
    ProviderRunRequest,
)
from fervis.model_io.backbone.tool_codec import (
    internal_tool_call_json,
    json_object_arguments_by_tool,
)
from fervis.model_io.providers.chat_runtime import (
    ConfiguredChatLoopRuntime,
    ProviderExecutionError,
    chat_json_system_prompt,
    chat_tool_system_prompt,
    provider_max_output_tokens,
    provider_max_retries,
    provider_error_payload,
    provider_timeout_seconds,
    strip_json_fence,
)

openai_sdk: ModuleType | None
try:  # pragma: no cover - optional dependency for runtime environments
    import openai as openai_sdk
except Exception:  # pragma: no cover - optional dependency not installed in tests
    openai_sdk = None


class OpenAIChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"


class OpenAIToolType(StrEnum):
    FUNCTION = "function"


class OpenAIToolChoice(StrEnum):
    REQUIRED = "required"


@dataclass(frozen=True)
class OpenAICompatibleRequestPayload:
    api_key: str | None
    base_url: str | None
    model: str
    max_tokens: int
    max_output_tokens_parameter: str
    max_retries: int
    timeout: float
    temperature: float
    prompt: str
    output_mode: ProviderOutputMode
    tool_specs: list[dict[str, Any]]
    json_object_arguments_by_tool: dict[str, tuple[str, ...]]
    system_prompt: str


class OpenAICompatibleLoopRuntime(ConfiguredChatLoopRuntime):
    def sdk_available(self) -> bool:
        return openai_sdk is not None

    def worker(self):
        return _openai_compatible_request_worker

    def request_payload(
        self, request: ProviderRunRequest
    ) -> OpenAICompatibleRequestPayload:
        return OpenAICompatibleRequestPayload(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=request.model_id or self.config.model_name,
            max_tokens=provider_max_output_tokens(),
            max_output_tokens_parameter=self.config.max_output_tokens_parameter,
            max_retries=provider_max_retries(),
            timeout=provider_timeout_seconds(),
            temperature=self.config.temperature,
            prompt=request.prompt,
            output_mode=request.output_mode,
            tool_specs=[_openai_tool_spec(item) for item in request.tool_specs],
            json_object_arguments_by_tool=json_object_arguments_by_tool(
                request.tool_specs
            ),
            system_prompt=request.system_prompt,
        )


def _openai_compatible_request_worker(
    payload: OpenAICompatibleRequestPayload, result_queue
) -> None:
    try:
        assert openai_sdk is not None
        client = _client(payload)
        response = client.chat.completions.create(**_completion_kwargs(payload))
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        result_queue.put(
            {
                "ok": True,
                "answer": _answer_from_choice(
                    choice,
                    output_mode=payload.output_mode,
                    json_object_arguments=payload.json_object_arguments_by_tool,
                ),
                "inputTokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "outputTokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "thinkingTokens": int(
                    getattr(completion_details, "reasoning_tokens", 0) or 0
                ),
            }
        )
    except BaseException as exc:
        result_queue.put(provider_error_payload(exc))


def _client(payload: OpenAICompatibleRequestPayload):
    assert openai_sdk is not None
    return openai_sdk.OpenAI(
        api_key=payload.api_key,
        base_url=payload.base_url,
        timeout=payload.timeout,
        max_retries=payload.max_retries,
    )


def _completion_kwargs(payload: OpenAICompatibleRequestPayload) -> dict[str, Any]:
    kwargs = _base_completion_kwargs(payload)
    if payload.output_mode == ProviderOutputMode.TOOL_CALL:
        kwargs.update(_tool_call_kwargs(payload))
    return kwargs


def _base_completion_kwargs(payload: OpenAICompatibleRequestPayload) -> dict[str, Any]:
    token_limit_parameter = payload.max_output_tokens_parameter or "max_tokens"
    system_prompt = (
        chat_tool_system_prompt(payload.system_prompt)
        if payload.output_mode == ProviderOutputMode.TOOL_CALL
        else chat_json_system_prompt(payload.system_prompt)
    )
    return {
        "model": payload.model,
        token_limit_parameter: payload.max_tokens,
        "temperature": payload.temperature,
        "messages": [
            {"role": OpenAIChatRole.SYSTEM.value, "content": system_prompt},
            {
                "role": OpenAIChatRole.USER.value,
                "content": payload.prompt,
            },
        ],
    }


def _tool_call_kwargs(payload: OpenAICompatibleRequestPayload) -> dict[str, Any]:
    tool_choice: object = OpenAIToolChoice.REQUIRED.value
    if len(payload.tool_specs) == 1:
        function = payload.tool_specs[0].get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            tool_choice = {
                "type": "function",
                "function": {"name": function["name"]},
            }
    return {
        "tools": payload.tool_specs,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
    }


def _openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return dict(schema)
    if schema.get("type") == "array" and isinstance(schema.get("prefixItems"), list):
        return _openai_tuple_array_schema(schema)
    transformed: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"allOf", "contains", "modelSchemas", "uniqueItems"}:
            continue
        if key == "const":
            transformed["enum"] = [value]
            continue
        if key == "oneOf" and isinstance(value, list):
            transformed["anyOf"] = [
                _openai_strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
            continue
        if isinstance(value, dict):
            transformed[key] = _openai_strict_schema(value)
            continue
        if isinstance(value, list):
            transformed[key] = [
                _openai_strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
            continue
        transformed[key] = value
    properties = transformed.get("properties")
    if not isinstance(properties, dict):
        return transformed
    original_required = set(transformed.get("required") or [])
    normalized_properties: dict[str, Any] = {}
    for name, value in properties.items():
        child = _openai_strict_schema(value) if isinstance(value, dict) else value
        if name not in original_required:
            child = _nullable_schema(child)
        normalized_properties[name] = child
    transformed["properties"] = normalized_properties
    transformed["required"] = list(normalized_properties.keys())
    transformed["additionalProperties"] = False
    return transformed


def _openai_tuple_array_schema(schema: dict[str, Any]) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"prefixItems", "items"}:
            continue
        if key in {"allOf", "contains"}:
            continue
        if key == "const":
            transformed["enum"] = [value]
            continue
        if key == "oneOf" and isinstance(value, list):
            transformed["anyOf"] = [
                _openai_strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
            continue
        if isinstance(value, dict):
            transformed[key] = _openai_strict_schema(value)
            continue
        if isinstance(value, list):
            transformed[key] = [
                _openai_strict_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
            continue
        transformed[key] = value
    prefix_items = [
        _openai_strict_schema(item) if isinstance(item, dict) else item
        for item in schema["prefixItems"]
    ]
    object_prefix_items = [item for item in prefix_items if isinstance(item, dict)]
    if len(object_prefix_items) == 1:
        transformed["items"] = object_prefix_items[0]
    else:
        transformed["items"] = {"anyOf": object_prefix_items}
    return transformed


def _nullable_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    if _schema_allows_null(schema):
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        return any(
            isinstance(item, dict) and item.get("type") == "null" for item in any_of
        )
    return False


def _openai_tool_spec(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": OpenAIToolType.FUNCTION.value,
        OpenAIToolType.FUNCTION.value: {
            "name": spec.name,
            "description": spec.description,
            "parameters": _openai_strict_schema(spec.input_schema),
            "strict": spec.strict,
        },
    }


def _answer_from_choice(
    choice: Any,
    *,
    output_mode: Any,
    json_object_arguments: dict[str, tuple[str, ...]],
) -> str:
    if output_mode == ProviderOutputMode.TOOL_CALL:
        tool_calls = list(getattr(choice.message, "tool_calls", None) or [])
        if len(tool_calls) != 1:
            raise ValueError(f"Expected exactly one tool call, got {len(tool_calls)}.")
        function = tool_calls[0].function
        raw_arguments = str(function.arguments or "{}")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ProviderExecutionError(
                error_class="JSONDecodeError",
                reason=str(exc),
                context=_tool_argument_decode_error_context(
                    choice=choice,
                    function=function,
                    raw_arguments=raw_arguments,
                    exc=exc,
                ),
            ) from exc
        return internal_tool_call_json(
            tool_name=function.name,
            arguments=arguments,
            json_object_arguments=json_object_arguments,
        )
    return strip_json_fence(str(choice.message.content or "").strip())


def _tool_argument_decode_error_context(
    *,
    choice: Any,
    function: Any,
    raw_arguments: str,
    exc: json.JSONDecodeError,
) -> dict[str, Any]:
    return {
        "tool_name": str(getattr(function, "name", "") or ""),
        "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
        "json_error_line": int(exc.lineno),
        "json_error_column": int(exc.colno),
        "json_error_pos": int(exc.pos),
        "raw_tool_arguments": raw_arguments,
        "raw_tool_arguments_len": len(raw_arguments),
        "raw_tool_arguments_sha256": hashlib.sha256(
            raw_arguments.encode("utf-8")
        ).hexdigest(),
        "raw_tool_arguments_around_error": _argument_window(
            raw_arguments,
            center=exc.pos,
            radius=1000,
        ),
        "raw_tool_arguments_tail": raw_arguments[-2000:],
    }


def _argument_window(raw_arguments: str, *, center: int, radius: int) -> str:
    start = max(0, center - radius)
    end = min(len(raw_arguments), center + radius)
    return raw_arguments[start:end]
