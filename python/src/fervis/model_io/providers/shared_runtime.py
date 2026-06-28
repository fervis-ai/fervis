"""Provider-neutral runtime helpers shared by model providers."""

from __future__ import annotations

from typing import Any

from fervis.model_io.backbone.dto import (
    ProviderRunChunk,
    ToolDecision,
    TraceEvent,
)


class DefaultHitlRuntime:
    def interruption_required(self, *, safety_classification: str) -> bool:
        return str(safety_classification).lower() == "high"

    def approve(self, *, reason: str | None = None) -> ToolDecision:
        return ToolDecision(decision="approve", reason=reason)

    def reject(self, *, reason: str | None = None) -> ToolDecision:
        return ToolDecision(decision="deny", reason=reason)


class DefaultProviderHook:
    def on_llm_start(self, *, prompt: str) -> dict[str, Any]:
        return {"event": "llm.start", "promptSize": len(prompt)}

    def on_llm_end(self, *, answer: str) -> dict[str, Any]:
        return {"event": "llm.end", "answerSize": len(answer)}

    def on_tool_start(self, *, operation: str) -> dict[str, Any]:
        return {"event": "tool.start", "operation": operation}

    def on_tool_end(self, *, operation: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": "tool.end",
            "operation": operation,
            "status": result.get("status"),
        }

    def pre_tool_use(
        self, *, operation: str, arguments: dict[str, Any], correlation_id: str
    ) -> dict[str, Any]:
        _ = self.on_tool_start(operation=operation)
        return {
            "status": "allowed",
            "operation": operation,
            "arguments": dict(arguments),
            "correlationId": correlation_id,
        }

    def post_tool_use(
        self,
        *,
        operation: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        correlation_id: str,
    ) -> dict[str, Any]:
        del arguments
        event = self.on_tool_end(operation=operation, result=result)
        event["correlationId"] = correlation_id
        return event


class DefaultHooksRuntime:
    def build_hooks(self) -> list[object]:
        return [DefaultProviderHook()]


class DefaultStreamingRuntime:
    def map_events(
        self, *, run_id: str, events: list[dict[str, Any]]
    ) -> list[ProviderRunChunk]:
        del run_id
        chunks: list[ProviderRunChunk] = []
        for event in events:
            chunks.append(
                ProviderRunChunk(
                    event_id=str(event.get("eventId") or ""),
                    event_type=str(event.get("eventType") or "run.progress"),
                    payload=dict(event.get("payload") or {}),
                )
            )
        return chunks


class DefaultTraceRuntime:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()
