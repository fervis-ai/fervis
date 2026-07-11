from __future__ import annotations

from typing import Any

from fervis.lineage.enums import QuestionRunKind
from fervis.questions.projection import (
    QuestionMemoryRunSelection,
    QuestionRunStatus,
    QuestionRunSummary,
    project_question_runs,
    select_conversation_memory_runs,
)
from tests.testkit.assertions import exact_mismatches


def run_questions_projection_case(payload: dict[str, Any]) -> list[str]:
    projection = project_question_runs(
        tuple(
            QuestionRunSummary(
                run_id=str(item["run_id"]),
                run_number=int(item["run_number"]),
                kind=QuestionRunKind(str(item["kind"])),
                status=QuestionRunStatus(str(item["status"])),
                answered=_answered(item),
                terminal=_terminal(item),
            )
            for item in payload["input"]["runs"]
        )
    )
    return exact_mismatches(
        actual={
            "primary_run_id": projection.primary_run_id,
            "latest_run_id": projection.latest_run_id,
            "active_run_id": projection.active_run_id,
        },
        expected=payload["expect"]["result_equals"],
    )


def _answered(item: dict[str, Any]) -> bool:
    value = item.get("answered", False)
    if not isinstance(value, bool):
        raise TypeError("questions.projection answered must be boolean")
    return value


def _terminal(item: dict[str, Any]) -> bool:
    value = item.get("terminal", item.get("answered", False))
    if not isinstance(value, bool):
        raise TypeError("questions.projection terminal must be boolean")
    return value


def run_questions_memory_projection_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    selected = input_payload.get("selected")
    actual = select_conversation_memory_runs(
        tuple(
            QuestionMemoryRunSelection(
                question_id=str(item["question_id"]),
                primary_run_id=(
                    str(item["primary_run_id"])
                    if item.get("primary_run_id") is not None
                    else None
                ),
            )
            for item in input_payload["questions"]
        ),
        selected_run_id=(str(selected["run_id"]) if selected is not None else None),
        selected_question_id=(
            str(selected["question_id"]) if selected is not None else None
        ),
    )
    return exact_mismatches(
        actual={"run_ids": list(actual)},
        expected=payload["expect"]["result_equals"],
    )
