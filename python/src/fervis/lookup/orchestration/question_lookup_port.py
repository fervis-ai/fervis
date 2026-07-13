"""Question lookup port adapter for the lookup service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fervis.questions.ports import (
    LookupExecutionRequest,
    LookupExecutionResult,
    QuestionLookupPort,
)

from .request import LookupProgressSink
from .service import LookupService


RequestContextBuilder = Callable[[LookupExecutionRequest], dict[str, Any]]
TerminalLineageCheck = Callable[[LookupExecutionRequest], bool]


def _conversation_context(request: LookupExecutionRequest) -> dict[str, Any]:
    return dict(request.conversation_context or {})


def _runtime_context(request: LookupExecutionRequest) -> dict[str, Any]:
    return dict(request.runtime_context or {})


@dataclass(frozen=True)
class LookupServiceQuestionLookupPort(QuestionLookupPort):
    lookup_service: LookupService
    terminal_lineage_recorded: TerminalLineageCheck
    conversation_context: RequestContextBuilder = _conversation_context
    runtime_context: RequestContextBuilder = _runtime_context

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink: LookupProgressSink | None = None,
    ) -> LookupExecutionResult:
        max_thinking_tokens = request.max_thinking_tokens
        if max_thinking_tokens is None:
            raise ValueError("lookup execution requires max_thinking_tokens")
        resolved_provider = self.lookup_service.provider_backbone.resolve_provider(
            request.provider,
            model_key=request.model_key,
        )
        self.lookup_service.provider_backbone.trace(
            event_type="run.requested",
            payload={
                "conversationId": request.conversation_id,
                "question": request.question,
                "provider": resolved_provider,
                "modelKey": request.model_key,
            },
            correlation_id=request.run_id,
        )
        result = self.lookup_service.run_lookup(
            run_id=request.run_id,
            conversation_id=request.conversation_id,
            tenant_id=request.tenant_id,
            question=request.question,
            read_context_ref=request.read_context_ref,
            delegated_credential=request.delegated_credential,
            provider=resolved_provider,
            model_key=request.model_key,
            conversation_context=self.conversation_context(request),
            max_budget_usd=request.max_budget_usd,
            max_thinking_tokens=max_thinking_tokens,
            user_context=self.runtime_context(request),
            active_attempt=request.active_attempt,
            progress_sink=progress_sink,
            clarification_response=request.clarification_response,
        )
        self.lookup_service.provider_backbone.trace(
            event_type=f"run.{result.status.lower()}",
            payload={"runId": request.run_id, "status": result.status},
            correlation_id=request.run_id,
        )
        return LookupExecutionResult(
            status=result.status,
            answer=result.answer,
            result_data=result.result_data,
            error=result.error,
            usage=dict(result.usage or {}),
            terminal_lineage_recorded=self.terminal_lineage_recorded(request),
        )
