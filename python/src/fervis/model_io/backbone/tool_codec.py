"""Provider-neutral helpers for native fervis tool calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from fervis.model_io.backbone.dto import ToolSpec


def json_object_arguments_by_tool(
    specs: Sequence[ToolSpec],
) -> dict[str, tuple[str, ...]]:
    return {
        spec.name: spec.json_object_arguments
        for spec in specs
        if spec.json_object_arguments
    }


def internal_tool_call_json(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    json_object_arguments: Mapping[str, Sequence[str]],
) -> str:
    normalized = normalize_tool_arguments(
        tool_name=tool_name,
        arguments=arguments,
        json_object_arguments=json_object_arguments,
    )
    return json.dumps(
        {"tool": tool_name, "arguments": normalized},
        separators=(",", ":"),
        default=str,
    )


def normalize_tool_arguments(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    json_object_arguments: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    normalized = dict(arguments)
    for key in json_object_arguments.get(tool_name, ()):
        normalized[key] = decode_json_object_argument(normalized.get(key))
    return normalized


def decode_json_object_argument(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_malformed_json_object": value}
        if isinstance(parsed, dict):
            return parsed
        return {"_malformed_json_object": value}
    raise ValueError("Expected a JSON object argument.")
