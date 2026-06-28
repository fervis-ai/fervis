"""Anthropic chat provider runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

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
    chat_json_system_prompt,
    chat_tool_system_prompt,
    provider_max_output_tokens,
    provider_max_retries,
    provider_error_payload,
    provider_timeout_seconds,
    strip_json_fence,
)
from fervis.model_io.providers.anthropic_adapter.conversation_resolution_transport import (
    compact_schema,
    normalize_tool_call,
    project_tool_specs,
)
from fervis.model_io.providers.schema_projection import strip_schema_keywords

try:  # pragma: no cover - optional dependency for runtime environments
    import anthropic
except Exception:  # pragma: no cover - optional dependency not installed in tests
    anthropic = None


_ANTHROPIC_FORBIDDEN_SCHEMA_KEYWORDS = frozenset(
    {
        "pattern",
        "modelSchemas",
        "allOf",
        "contains",
    }
)


class AnthropicContentType(StrEnum):
    TOOL_USE = "tool_use"


class AnthropicMessageRole(StrEnum):
    USER = "user"


class AnthropicToolChoiceType(StrEnum):
    ANY = "any"


class AnthropicToolChoiceKey(StrEnum):
    TYPE = "type"
    DISABLE_PARALLEL_TOOL_USE = "disable_parallel_tool_use"


@dataclass(frozen=True)
class AnthropicRequestPayload:
    api_key: str | None
    model: str
    max_tokens: int
    max_retries: int
    timeout: float
    temperature: float
    prompt: str
    output_mode: ProviderOutputMode
    tool_specs: list[dict[str, Any]]
    json_object_arguments_by_tool: dict[str, tuple[str, ...]]
    system_prompt: str


class AnthropicLoopRuntime(ConfiguredChatLoopRuntime):
    def sdk_available(self) -> bool:
        return anthropic is not None

    def worker(self):
        return _anthropic_request_worker

    def request_payload(self, request: ProviderRunRequest) -> AnthropicRequestPayload:
        anthropic_tool_specs = project_tool_specs(request.tool_specs)
        return AnthropicRequestPayload(
            api_key=self.config.api_key,
            model=request.model_id or self.config.model_name,
            max_tokens=provider_max_output_tokens(),
            max_retries=provider_max_retries(),
            timeout=provider_timeout_seconds(),
            temperature=self.config.temperature,
            prompt=request.prompt,
            output_mode=request.output_mode,
            tool_specs=[_anthropic_tool_spec(item) for item in anthropic_tool_specs],
            json_object_arguments_by_tool=json_object_arguments_by_tool(
                request.tool_specs
            ),
            system_prompt=request.system_prompt,
        )

    def timeout_reason(self) -> str:
        return "Anthropic provider hard timeout exceeded."


def _anthropic_request_worker(payload: AnthropicRequestPayload, result_queue) -> None:
    try:
        assert anthropic is not None
        client = _client(payload)
        response = client.messages.create(**_message_kwargs(payload))
        result_queue.put(
            {
                "ok": True,
                "answer": _answer_from_response(
                    response,
                    output_mode=payload.output_mode,
                    json_object_arguments=payload.json_object_arguments_by_tool,
                ),
                "inputTokens": int(getattr(response.usage, "input_tokens", 0) or 0),
                "outputTokens": int(getattr(response.usage, "output_tokens", 0) or 0),
                "thinkingTokens": int(
                    getattr(response.usage, "thinking_tokens", 0) or 0
                ),
            }
        )
    except BaseException as exc:
        result_queue.put(provider_error_payload(exc))


def _tool_call_arguments(response: Any) -> tuple[str, dict[str, Any]]:
    tool_uses = [
        block
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == AnthropicContentType.TOOL_USE.value
    ]
    if len(tool_uses) != 1:
        raise ValueError(f"Expected exactly one tool_use block, got {len(tool_uses)}.")
    tool_use = tool_uses[0]
    return str(tool_use.name), dict(tool_use.input or {})


def _client(payload: AnthropicRequestPayload):
    assert anthropic is not None
    return anthropic.Anthropic(
        api_key=payload.api_key,
        http_client=httpx.Client(timeout=payload.timeout, trust_env=False),
        max_retries=payload.max_retries,
    )


def _message_kwargs(payload: AnthropicRequestPayload) -> dict[str, Any]:
    kwargs = _base_message_kwargs(payload)
    if payload.output_mode == ProviderOutputMode.TOOL_CALL and payload.tool_specs:
        kwargs.update(_tool_call_kwargs(payload))
    return kwargs


def _base_message_kwargs(payload: AnthropicRequestPayload) -> dict[str, Any]:
    return {
        "model": payload.model,
        "max_tokens": payload.max_tokens,
        "temperature": payload.temperature,
        "system": _system_prompt(payload),
        "messages": [
            {
                "role": AnthropicMessageRole.USER.value,
                "content": payload.prompt,
            }
        ],
    }


def _system_prompt(payload: AnthropicRequestPayload) -> str:
    if payload.output_mode == ProviderOutputMode.TOOL_CALL:
        return chat_tool_system_prompt(payload.system_prompt)
    return chat_json_system_prompt(payload.system_prompt)


def _tool_call_kwargs(payload: AnthropicRequestPayload) -> dict[str, Any]:
    return {
        "tools": payload.tool_specs,
        "tool_choice": {
            AnthropicToolChoiceKey.TYPE.value: AnthropicToolChoiceType.ANY.value,
            AnthropicToolChoiceKey.DISABLE_PARALLEL_TOOL_USE.value: True,
        },
    }


def _anthropic_tool_spec(spec: ToolSpec) -> dict[str, Any]:
    input_schema = _anthropic_input_schema_for_tool(spec)
    payload = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": input_schema,
        "strict": spec.strict,
    }
    if spec.input_examples:
        payload["input_examples"] = list(spec.input_examples)
    return payload


def anthropic_tool_specs_for_budget(
    tool_specs: tuple[ToolSpec, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(_anthropic_tool_spec(spec) for spec in tool_specs)


def _anthropic_input_schema_for_tool(spec: ToolSpec) -> dict[str, Any]:
    if not spec.strict:
        return spec.input_schema
    schema = spec.input_schema
    schema = compact_schema(
        tool_name=spec.name,
        schema=schema,
    )
    return _anthropic_strict_schema(schema)


def _anthropic_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if isinstance(schema, list):
        return [_anthropic_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "oneOf" in schema:
        return {
            "anyOf": [
                _anthropic_strict_schema(item)
                for item in schema["oneOf"]
                if isinstance(item, dict)
            ]
        }
    stripped = strip_schema_keywords(
        schema,
        forbidden_keywords=_ANTHROPIC_FORBIDDEN_SCHEMA_KEYWORDS,
    )
    return {key: _anthropic_strict_schema(value) for key, value in stripped.items()}


def _answer_from_response(
    response: Any,
    *,
    output_mode: Any,
    json_object_arguments: dict[str, tuple[str, ...]],
) -> str:
    if output_mode == ProviderOutputMode.TOOL_CALL:
        tool_name, tool_arguments = _tool_call_arguments(response)
        (
            tool_name,
            tool_arguments,
        ) = normalize_tool_call(
            tool_name=tool_name,
            arguments=tool_arguments,
        )
        return internal_tool_call_json(
            tool_name=tool_name,
            arguments=tool_arguments,
            json_object_arguments=json_object_arguments,
        )
    return _text_from_response(response)


def _text_from_response(response: Any) -> str:
    fragments: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            fragments.append(str(text))
    return strip_json_fence("\n".join(fragments).strip())
