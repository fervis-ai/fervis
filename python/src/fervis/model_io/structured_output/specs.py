"""Structured-output spec builders."""

from __future__ import annotations

from typing import Any

from fervis.model_io.backbone.dto import ToolSpec


def required_tool_spec(
    *,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    input_examples: tuple[dict[str, Any], ...] = (),
    json_object_arguments: tuple[str, ...] = (),
    transport_context: dict[str, Any] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=tool_name,
        description=tool_description,
        input_schema=input_schema,
        input_examples=input_examples,
        json_object_arguments=json_object_arguments,
        strict=True,
        transport_context=dict(transport_context or {}),
    )
