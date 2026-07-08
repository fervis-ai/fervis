"""Agent-facing run event projection shared by external interfaces."""

from __future__ import annotations

from typing import Any

from fervis.interfaces.agent.actions import (
    inspect_question_action,
    provide_clarification_action,
)


def agent_run_event(
    event: dict[str, object],
    *,
    tenant_id: str | None = None,
    principal_id: str | None = None,
) -> dict[str, object]:
    payload = dict(event)
    name = str(payload.get("event") or "")
    if name == "run.needs_clarification":
        conversation_id = str(payload.get("conversation_id") or "")
        question_id = str(payload.get("question_id") or "")
        run_id = str(payload.get("run_id") or "")
        clarification_id = _actionable_clarification_id(payload)
        if conversation_id:
            payload["next_actions"] = [
                provide_clarification_action(
                    conversation_id,
                    question_id=question_id or None,
                    previous_run_id=run_id or None,
                    clarification_id=clarification_id or None,
                    tenant_id=tenant_id,
                    principal_id=principal_id,
                )
            ]
        return payload
    if name in {
        "run.completed",
        "run.failed",
        "run.queued",
        "run.wait_unavailable",
    }:
        _add_inspect_question_action(payload)
        return payload
    if name == "run.active_conflict":
        question_id = str(payload.get("question_id") or "")
        if question_id:
            payload["next_actions"] = [inspect_question_action(question_id)]
        return payload
    return payload


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _add_inspect_question_action(event: dict[str, object]) -> None:
    question_id = str(event.get("question_id") or "")
    if question_id:
        event["next_actions"] = [
            inspect_question_action(
                question_id,
                debug=event.get("event") == "run.failed",
            )
        ]


def _actionable_clarification_id(event: dict[str, object]) -> str:
    clarifications = event.get("clarifications")
    if not isinstance(clarifications, list) or not clarifications:
        raise ValueError("run.needs_clarification event requires clarifications")
    first = clarifications[0]
    if not isinstance(first, dict):
        raise ValueError("run.needs_clarification event requires clarification objects")
    clarification_id = str(first.get("id") or "")
    if not clarification_id.strip():
        raise ValueError("run.needs_clarification event requires clarification id")
    if not str(first.get("question") or "").strip():
        raise ValueError(
            "run.needs_clarification event requires clarification question"
        )
    return clarification_id
