"""Structured-output generation through provider transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.model_io.backbone.dto import (
    ToolSpec,
    ProviderOutputMode,
)

from .errors import (
    RequiredToolOutputError,
    provider_error_code,
    provider_error_context,
)
from .parsing import raw_tool_output_text, tool_payload
from .validation import strip_null_properties, validate_tool_arguments


@dataclass(frozen=True)
class RequiredToolOutput:
    arguments: dict[str, Any]
    output: dict[str, Any]
    raw_output: str
    tool_spec: ToolSpec


def generate_one_of_tool_output(
    *,
    model_port: Any,
    provider: str,
    system_prompt: str,
    prompt: str,
    max_thinking_tokens: int,
    tool_specs: tuple[ToolSpec, ...],
) -> RequiredToolOutput:
    if not tool_specs:
        raise ValueError("At least one tool spec is required.")
    try:
        output = model_port.generate(
            provider=provider,
            system_prompt=system_prompt,
            prompt=prompt,
            max_thinking_tokens=max_thinking_tokens,
            output_mode=ProviderOutputMode.TOOL_CALL,
            tool_specs=tool_specs,
        )
    except Exception as exc:
        raise RequiredToolOutputError(
            "provider tool output failed",
            tool_specs=tool_specs,
            error_code=provider_error_code(exc),
            error_context=provider_error_context(exc),
        ) from exc
    try:
        payload = tool_payload(output.get("answer"))
    except Exception as exc:
        raise RequiredToolOutputError(
            "provider tool output is invalid",
            output=output,
            tool_specs=tool_specs,
        ) from exc
    actual_tool = str(payload.get("tool") or "")
    specs_by_name = {spec.name: spec for spec in tool_specs}
    tool_spec = specs_by_name.get(actual_tool)
    if tool_spec is None:
        expected = ", ".join(specs_by_name)
        arguments = payload.get("arguments")
        raise RequiredToolOutputError(
            f"Expected one of {expected}, got {actual_tool}.",
            output=output,
            arguments=arguments if isinstance(arguments, dict) else {},
            tool_specs=tool_specs,
        )
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise RequiredToolOutputError(
            f"Tool {actual_tool} requires object arguments.",
            output=output,
            tool_specs=tool_specs,
        )
    normalized_arguments = strip_null_properties(
        arguments,
        schema=tool_spec.input_schema,
    )
    validate_tool_arguments(
        tool_spec=tool_spec,
        arguments=normalized_arguments,
        output=output,
        tool_specs=tool_specs,
    )
    return RequiredToolOutput(
        arguments=dict(normalized_arguments),
        output=output,
        raw_output=raw_tool_output_text(output),
        tool_spec=tool_spec,
    )
