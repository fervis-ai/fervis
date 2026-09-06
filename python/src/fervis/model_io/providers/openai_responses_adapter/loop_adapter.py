"""OpenAI Responses API runtime."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from fervis.model_io.backbone.dto import (
    ProviderOutputMode,
    ProviderRunRequest,
    ToolSpec,
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
    provider_error_payload,
    provider_max_output_tokens,
    provider_max_retries,
    provider_timeout_seconds,
    strip_json_fence,
)
from fervis.model_io.providers.openai_compatible_adapter.loop_adapter import (
    openai_compatible_client,
    openai_sdk,
    openai_strict_schema,
)


@dataclass(frozen=True)
class OpenAIResponsesRequestPayload:
    api_key: str | None
    base_url: str | None
    model: str
    max_output_tokens: int
    max_retries: int
    timeout: float
    reasoning_effort: str
    prompt: str
    output_mode: ProviderOutputMode
    tool_specs: list[dict[str, Any]]
    json_object_arguments_by_tool: dict[str, tuple[str, ...]]
    system_prompt: str
    service_tier: str | None


class OpenAIResponsesLoopRuntime(ConfiguredChatLoopRuntime):
    def sdk_available(self) -> bool:
        return openai_sdk is not None

    def worker(self):
        return _openai_responses_request_worker

    def request_payload(
        self, request: ProviderRunRequest
    ) -> OpenAIResponsesRequestPayload:
        reasoning_effort = self.config.reasoning_effort
        if reasoning_effort is None:
            raise ValueError("OpenAI Responses reasoning effort must be configured.")
        return OpenAIResponsesRequestPayload(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=request.model_id or self.config.model_name,
            max_output_tokens=provider_max_output_tokens(),
            max_retries=provider_max_retries(),
            timeout=provider_timeout_seconds(),
            reasoning_effort=reasoning_effort,
            prompt=request.prompt,
            output_mode=request.output_mode,
            tool_specs=[_responses_tool_spec(item) for item in request.tool_specs],
            json_object_arguments_by_tool=json_object_arguments_by_tool(
                request.tool_specs
            ),
            system_prompt=request.system_prompt,
            service_tier=_configured_service_tier(),
        )


def _configured_service_tier() -> str | None:
    value = os.getenv("FERVIS_OPENAI_SERVICE_TIER", "").strip().lower()
    if not value:
        return None
    if value not in {"auto", "default", "flex"}:
        raise ProviderExecutionError(error_class="ProviderConfigurationError", reason="FERVIS_OPENAI_SERVICE_TIER is invalid")
    return value


def _openai_responses_request_worker(
    payload: OpenAIResponsesRequestPayload, result_queue: Any
) -> None:
    token_usage: dict[str, Any] = {}
    try:
        client = openai_compatible_client(
            api_key=payload.api_key,
            base_url=payload.base_url,
            timeout=payload.timeout,
            max_retries=payload.max_retries,
        )
        response = client.responses.create(**_response_kwargs(payload))
        usage = getattr(response, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
        # Responses includes reasoning in output_tokens. Fervis prices its
        # output and thinking counters separately, so they must be disjoint.
        token_usage = {
            "inputTokens": int(getattr(usage, "input_tokens", 0) or 0),
            "outputTokens": int(getattr(usage, "output_tokens", 0) or 0) - thinking_tokens,
            "thinkingTokens": thinking_tokens,
        }
        service_tier = getattr(response, "service_tier", None)
        if service_tier:
            token_usage["usageDetails"] = {"serviceTier": str(service_tier)}
        answer = _answer_from_response(
            response, output_mode=payload.output_mode,
            json_object_arguments=payload.json_object_arguments_by_tool,
        )
        result_queue.put({"ok": True, "answer": answer, **token_usage})
    except BaseException as exc:
        failure = provider_error_payload(exc)
        if token_usage:
            failure["usage"] = token_usage
        result_queue.put(failure)


def _response_kwargs(payload: OpenAIResponsesRequestPayload) -> dict[str, Any]:
    system_prompt = (
        chat_tool_system_prompt(payload.system_prompt)
        if payload.output_mode == ProviderOutputMode.TOOL_CALL
        else chat_json_system_prompt(payload.system_prompt)
    )
    kwargs: dict[str, Any] = {
        "model": payload.model,
        "instructions": system_prompt,
        "input": payload.prompt,
        "max_output_tokens": payload.max_output_tokens,
        "reasoning": {"effort": payload.reasoning_effort},
    }
    if payload.service_tier is not None:
        kwargs["service_tier"] = payload.service_tier
    if payload.output_mode == ProviderOutputMode.TOOL_CALL:
        kwargs.update(
            {
                "tools": payload.tool_specs,
                "tool_choice": _tool_choice(payload.tool_specs),
                "parallel_tool_calls": False,
            }
        )
    return kwargs


def _tool_choice(tool_specs: list[dict[str, Any]]) -> object:
    if len(tool_specs) == 1:
        return {"type": "function", "name": tool_specs[0]["name"]}
    return "required"


def _responses_tool_spec(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": openai_strict_schema(spec.input_schema),
        "strict": spec.strict,
    }


def _answer_from_response(
    response: Any,
    *,
    output_mode: ProviderOutputMode,
    json_object_arguments: dict[str, tuple[str, ...]],
) -> str:
    status = getattr(response, "status", None)
    if status == "incomplete":
        reason = str(getattr(getattr(response, "incomplete_details", None), "reason", "unknown"))
        raise ProviderExecutionError(
            error_class=("ProviderOutputLimitExceeded" if reason == "max_output_tokens" else "ProviderIncompleteResponse"),
            reason=f"OpenAI response incomplete: {reason}",
            context={"status": status, "reason": reason},
        )
    if output_mode != ProviderOutputMode.TOOL_CALL:
        return strip_json_fence(str(getattr(response, "output_text", "") or ""))
    tool_calls = [
        item
        for item in list(getattr(response, "output", ()) or ())
        if getattr(item, "type", "") == "function_call"
    ]
    if len(tool_calls) != 1:
        raise ValueError(f"Expected exactly one tool call, got {len(tool_calls)}.")
    call = tool_calls[0]
    raw_arguments = str(getattr(call, "arguments", "") or "{}")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ProviderExecutionError(
            error_class="JSONDecodeError",
            reason=str(exc),
            context={
                "tool_name": str(getattr(call, "name", "") or ""),
                "json_error_line": int(exc.lineno),
                "json_error_column": int(exc.colno),
                "json_error_pos": int(exc.pos),
                "raw_tool_arguments": raw_arguments,
                "raw_tool_arguments_len": len(raw_arguments),
                "raw_tool_arguments_sha256": hashlib.sha256(
                    raw_arguments.encode("utf-8")
                ).hexdigest(),
            },
        ) from exc
    return internal_tool_call_json(
        tool_name=str(getattr(call, "name", "")),
        arguments=arguments,
        json_object_arguments=json_object_arguments,
    )
