"""Lineage summaries for fact-planning model output."""

from __future__ import annotations

from typing import Any

from fervis.lineage.step_summary import StepSummaryItem, step_summary_json


def fact_planning_step_summary(payload: dict[str, Any]) -> dict[str, object]:
    return step_summary_json(
        *(
            item
            for answer in _outcome_answers(payload)
            if (item := _fact_answer_binding(answer))
        )
    )


def _outcome_answers(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    outcome = _dict_or_empty(payload.get("outcome"))
    raw_answers = outcome.get("answers")
    if not isinstance(raw_answers, list):
        return ()
    return tuple(item for item in raw_answers if isinstance(item, dict))


def _fact_answer_binding(answer: dict[str, Any]) -> StepSummaryItem | None:
    values = {
        "group": _selected_value(answer.get("group"), key="field_id"),
        "metric": _selected_value(answer.get("metric"), key="field_id"),
        "function": _selected_value(answer.get("function"), key="value"),
        "rank": _rank_value(answer.get("rank")),
    }
    present_values = {key: value for key, value in values.items() if value}
    if not present_values:
        return None
    return StepSummaryItem(
        text="Binding: "
        + " ".join(f"{key}={value}" for key, value in present_values.items())
    )


def _rank_value(value: object) -> str:
    rank = _dict_or_empty(value)
    return " ".join(
        str(rank.get(key) or "") for key in ("sort", "limit") if rank.get(key)
    )


def _selected_value(value: object, *, key: str) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get(key) or value.get("id") or "")


def _dict_or_empty(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value
