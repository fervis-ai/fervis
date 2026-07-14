"""Runtime ports consumed by the framework-neutral goldset runner."""

from __future__ import annotations

from typing import Protocol

from fervis.questions import AskRequest, AskResult, ClarificationResponseRequest
from fervis.run_work.events import QuestionRunEventSink


class GoldsetQuestions(Protocol):
    def ask(
        self,
        request: AskRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...

    def respond_to_clarification(
        self,
        request: ClarificationResponseRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...


class GoldsetRunFollower(Protocol):
    def follow(
        self,
        result: AskResult,
        *,
        event_sink: QuestionRunEventSink | None = None,
        wait_seconds: float = 0.0,
    ) -> AskResult: ...
