"""Anthropic-private conversation-resolution schema projection."""

from __future__ import annotations

from typing import Any

from fervis.model_io.backbone.dto import ToolSpec


def project_tool_specs(
    tool_specs: tuple[ToolSpec, ...],
) -> tuple[ToolSpec, ...]:
    return tool_specs


def normalize_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    return tool_name, arguments


def compact_schema(*, tool_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    del tool_name
    return schema
