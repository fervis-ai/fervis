"""Provider-neutral DTOs shared across adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderOutputMode(StrEnum):
    TEXT = "text"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    input_examples: tuple[dict[str, Any], ...] = ()
    json_object_arguments: tuple[str, ...] = ()
    strict: bool = True
    transport_context: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ProviderRunRequest:
    provider: str
    prompt: str
    max_thinking_tokens: int
    system_prompt: str
    output_mode: ProviderOutputMode = ProviderOutputMode.TEXT
    tool_specs: tuple[ToolSpec, ...] = ()
    model_id: str | None = None


@dataclass(frozen=True)
class ProviderRunChunk:
    event_id: str
    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ProviderRunResult:
    provider: str
    answer: str
    usage: dict[str, Any]
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    provider_session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDecision:
    decision: str
    reason: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    payload: dict[str, Any]
    correlation_id: str | None = None
