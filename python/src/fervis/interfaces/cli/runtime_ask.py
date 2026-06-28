"""Runtime ask event-stream orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, TextIO

from fervis.interfaces.common.events import agent_run_event, jsonable
from fervis.interfaces.common.admission import (
    ConfiguredModelPolicy,
    ModelPolicyValidationError,
)
from fervis.interfaces.cli.principals import cli_question_principal
from fervis.questions import (
    AskRequest,
    AskRequestLimits,
    AskResult,
    ContinueQuestionRequest,
    ExecutionMode,
)
from fervis.lineage.enums import RunTriggerKind
from fervis.run_work.events import (
    QuestionRunEventPayload,
    QuestionRunEventSink,
)
from fervis.project import ProjectInspection


@dataclass(frozen=True)
class RuntimeAskEventStream:
    events: tuple[dict[str, object], ...]
    exit_code: int


class RuntimeAskQuestions(Protocol):
    def ask(
        self,
        request: AskRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...

    def continue_question(
        self,
        request: ContinueQuestionRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...


class RuntimeAskFollower(Protocol):
    def follow(
        self,
        result: AskResult,
        *,
        event_sink: QuestionRunEventSink | None = None,
        wait_seconds: float = 0.0,
    ) -> AskResult: ...


@dataclass(frozen=True)
class RuntimeAskPorts:
    questions: RuntimeAskQuestions
    question_run_limits: AskRequestLimits
    project: ProjectInspection
    question_run_follower: RuntimeAskFollower | None = None
    model_policy: ConfiguredModelPolicy | None = None


class RuntimeAskJsonlSink:
    def __init__(
        self,
        stdout: TextIO | None = None,
        *,
        tenant_id: str | None = None,
        principal_id: str | None = None,
    ) -> None:
        self.stdout = stdout
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self.events: list[dict[str, object]] = []
        self._run_context_by_id: dict[str, dict[str, str]] = {}

    def emit(self, event: QuestionRunEventPayload) -> None:
        payload = dict(event)
        self._remember_run_context(payload)
        self._enrich_run_context(payload)
        payload = agent_run_event(
            payload,
            tenant_id=self.tenant_id,
            principal_id=self.principal_id,
        )
        self.events.append(payload)
        if self.stdout is None:
            return
        self.stdout.write(json.dumps(jsonable(payload), sort_keys=True))
        self.stdout.write("\n")
        self.stdout.flush()

    def _remember_run_context(self, event: dict[str, object]) -> None:
        if event.get("event") != "run.accepted":
            return
        run_id = str(event.get("run_id") or "")
        if not run_id:
            return
        self._run_context_by_id[run_id] = {
            "question_id": str(event.get("question_id") or ""),
            "conversation_id": str(event.get("conversation_id") or ""),
        }

    def _enrich_run_context(self, event: dict[str, object]) -> None:
        if event.get("event") not in {
            "run.completed",
            "run.failed",
            "run.queued",
            "run.wait_unavailable",
            "run.active_conflict",
        }:
            return
        run_id = str(event.get("run_id") or "")
        context = self._run_context_by_id.get(run_id)
        if not context:
            return
        for key, value in context.items():
            if value and not event.get(key):
                event[key] = value


class _HoldingQueuedTerminalSink:
    _TERMINAL_EVENTS = frozenset(
        {
            "run.completed",
            "run.failed",
            "run.needs_clarification",
            "run.active_conflict",
        }
    )

    def __init__(self, sink: RuntimeAskJsonlSink) -> None:
        self.sink = sink
        self.held_queued_event: QuestionRunEventPayload | None = None

    def emit(self, event: QuestionRunEventPayload) -> None:
        if event.get("event") == "run.queued":
            self.held_queued_event = event
            return
        if event.get("event") in self._TERMINAL_EVENTS:
            self.held_queued_event = None
        self.sink.emit(event)

    def flush_queued(self) -> None:
        if self.held_queued_event is None:
            return
        self.sink.emit(self.held_queued_event)
        self.held_queued_event = None


def run_runtime_ask(
    args,
    *,
    ports: RuntimeAskPorts,
    event_sink: RuntimeAskJsonlSink | None = None,
) -> RuntimeAskEventStream:
    sink = event_sink or RuntimeAskJsonlSink(
        tenant_id=args.tenant_id,
        principal_id=args.principal_id,
    )
    try:
        wait_seconds = _wait_seconds(args)
        request = _ask_request(
            args,
            limits=ports.question_run_limits,
            model_policy=ports.model_policy or ConfiguredModelPolicy(),
            project=ports.project,
        )
    except (ModelPolicyValidationError, ValueError) as error:
        sink.emit(
            _error_event(
                event="run.invalid_request",
                status="INVALID_REQUEST",
                code="invalid_request",
                message=str(error),
            )
        )
        return RuntimeAskEventStream(events=tuple(sink.events), exit_code=2)

    hold_queued = wait_seconds > 0 and ports.question_run_follower is not None
    ask_sink: QuestionRunEventSink = (
        _HoldingQueuedTerminalSink(sink) if hold_queued else sink
    )
    try:
        if isinstance(request, ContinueQuestionRequest):
            result = ports.questions.continue_question(request, event_sink=ask_sink)
        else:
            result = ports.questions.ask(request, event_sink=ask_sink)
    except ValueError as error:
        sink.emit(
            _error_event(
                event="run.invalid_request",
                status="INVALID_REQUEST",
                code="invalid_request",
                message=str(error),
            )
        )
        return RuntimeAskEventStream(events=tuple(sink.events), exit_code=2)
    except RuntimeError as error:
        sink.emit(_runtime_error_event(error))
        return RuntimeAskEventStream(events=tuple(sink.events), exit_code=1)

    result = _follow_if_requested(
        result,
        wait_seconds=wait_seconds,
        follower=ports.question_run_follower,
        sink=sink,
        ask_sink=ask_sink,
    )
    return RuntimeAskEventStream(
        events=tuple(sink.events),
        exit_code=_exit_code(result),
    )


def _ask_request(
    args,
    *,
    limits: AskRequestLimits,
    model_policy: ConfiguredModelPolicy,
    project: ProjectInspection,
) -> AskRequest | ContinueQuestionRequest:
    continuation = _continuation_args(args)
    model = model_policy.admit(
        requested_provider="",
        requested_model_key=args.model_key,
    )
    principal = cli_question_principal(
        principal_id=args.principal_id,
        tenant_id=args.tenant_id,
        project=project,
    )
    if continuation is not None:
        question_id, previous_run_id, clarification_id = continuation
        return ContinueQuestionRequest(
            question_id=question_id,
            question=args.question,
            principal=principal,
            trigger_kind=RunTriggerKind.CLARIFICATION_RESPONSE,
            execution_mode=ExecutionMode.QUEUED,
            provider=model.provider,
            model_key=model.model_key,
            idempotency_key=args.idempotency_key,
            previous_run_id=None,
            trigger_clarification_response_run_id=previous_run_id,
            trigger_clarification_response_id=clarification_id,
            max_budget_usd=args.max_budget_usd,
            max_thinking_tokens=args.max_thinking_tokens,
            limits=limits,
        )
    return AskRequest(
        question=args.question,
        principal=principal,
        execution_mode=ExecutionMode.QUEUED,
        conversation_id=args.conversation_id or "",
        provider=model.provider,
        model_key=model.model_key,
        idempotency_key=args.idempotency_key,
        max_budget_usd=args.max_budget_usd,
        max_thinking_tokens=args.max_thinking_tokens,
        limits=limits,
    )


def _continuation_args(args) -> tuple[str, str, str] | None:
    values = {
        "question_id": getattr(args, "question_id", None),
        "previous_run_id": getattr(args, "previous_run_id", None),
        "clarification_id": getattr(args, "clarification_id", None),
    }
    present = {key: str(value).strip() for key, value in values.items() if value}
    if not present:
        return None
    missing = sorted(key for key in values if key not in present)
    if missing:
        raise ValueError(
            "clarification continuation requires "
            + ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        )
    return (
        present["question_id"],
        present["previous_run_id"],
        present["clarification_id"],
    )


def _follow_if_requested(
    result: AskResult,
    *,
    wait_seconds: float,
    follower: RuntimeAskFollower | None,
    sink: RuntimeAskJsonlSink,
    ask_sink: QuestionRunEventSink,
) -> AskResult:
    if wait_seconds <= 0 or result.status not in {"QUEUED", "RUNNING"}:
        _flush_held_queue(ask_sink)
        return result
    if follower is None:
        sink.emit(
            {
                "event": "run.wait_unavailable",
                "run_id": result.run_id,
                "status": result.status,
                "message": "local wait execution is not configured",
            }
        )
        return result
    try:
        followed = follower.follow(
            result,
            event_sink=sink,
            wait_seconds=wait_seconds,
        )
    except RuntimeError as error:
        sink.emit(_runtime_error_event(error))
        return AskResult(
            status="FAILED",
            conversation_id=result.conversation_id,
            question_id=result.question_id,
            run_id=result.run_id,
            error=str(error) or error.__class__.__name__,
        )
    if followed.status in {"QUEUED", "RUNNING"}:
        _flush_held_queue(ask_sink)
    return followed


def _flush_held_queue(sink: QuestionRunEventSink) -> None:
    if isinstance(sink, _HoldingQueuedTerminalSink):
        sink.flush_queued()


def _error_event(
    *,
    event: str,
    status: str,
    code: str,
    message: str,
    retryable: bool = False,
) -> dict[str, object]:
    return {
        "event": event,
        "status": status,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


def _runtime_error_event(error: RuntimeError) -> dict[str, object]:
    return _error_event(
        event="run.failed",
        status="RUNTIME_ERROR",
        code="runtime_ask_failed",
        message=str(error) or error.__class__.__name__,
    )


def _exit_code(result: AskResult) -> int:
    if result.status == "ACTIVE_RUN_CONFLICT":
        return 3
    if result.status == "FAILED":
        return 1
    return 0


def _wait_seconds(args) -> float:
    seconds = float(getattr(args, "wait", 0.0) or 0.0)
    if seconds < 0:
        raise ValueError("--wait must be greater than or equal to 0")
    return seconds
