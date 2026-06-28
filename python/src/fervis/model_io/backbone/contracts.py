"""Public provider-backbone contracts.

Only adapter modules should depend on provider SDKs. Domain modules depend on these
protocols and DTOs.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from .dto import (
    ToolSpec,
    ProviderRunChunk,
    ProviderOutputMode,
    ProviderRunRequest,
    ProviderRunResult,
    SessionRef,
    ToolDecision,
    TraceEvent,
)


class ProviderModelAdapter(Protocol):
    provider_name: str

    def generate(
        self,
        *,
        model_id: str | None = None,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str,
        output_mode: ProviderOutputMode = ProviderOutputMode.TEXT,
        tool_specs: tuple[ToolSpec, ...] = (),
    ) -> dict[str, Any]: ...


class LoopRuntime(Protocol):
    def run(self, request: ProviderRunRequest) -> ProviderRunResult: ...


class StreamingRuntime(Protocol):
    def map_events(
        self, *, run_id: str, events: list[dict[str, Any]]
    ) -> Iterable[ProviderRunChunk]: ...


class SessionRuntime(Protocol):
    def continue_session(self, *, session_id: str | None) -> SessionRef: ...

    def resume_session(self, *, session_id: str) -> SessionRef: ...

    def fork_session(
        self, *, session_id: str, branch_point_event_id: str
    ) -> SessionRef: ...


class HitlRuntime(Protocol):
    def interruption_required(self, *, safety_classification: str) -> bool: ...

    def approve(self, *, reason: str | None = None) -> ToolDecision: ...

    def reject(self, *, reason: str | None = None) -> ToolDecision: ...


class HooksRuntime(Protocol):
    def build_hooks(self) -> list[Any]: ...


class TraceRuntime(Protocol):
    def record(self, event: TraceEvent) -> None: ...
