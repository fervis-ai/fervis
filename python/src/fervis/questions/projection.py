"""Language-neutral question run selection semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fervis.lineage.enums import QuestionRunKind


class QuestionRunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FAILED = "FAILED"


_ACTIVE_STATUSES = frozenset(
    {QuestionRunStatus.QUEUED, QuestionRunStatus.RUNNING}
)


@dataclass(frozen=True)
class QuestionRunSummary:
    run_id: str
    run_number: int
    kind: QuestionRunKind
    status: QuestionRunStatus
    answered: bool = False
    terminal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("question run summary requires run_id")
        if (
            not isinstance(self.run_number, int)
            or isinstance(self.run_number, bool)
            or self.run_number < 1
        ):
            raise ValueError("question run summary requires positive run_number")
        if not isinstance(self.kind, QuestionRunKind):
            raise TypeError("question run summary kind must be QuestionRunKind")
        if not isinstance(self.status, QuestionRunStatus):
            raise TypeError("question run summary status must be QuestionRunStatus")
        if not isinstance(self.answered, bool):
            raise TypeError("question run summary answered must be bool")
        if not isinstance(self.terminal, bool):
            raise TypeError("question run summary terminal must be bool")
        if self.answered and not self.terminal:
            raise ValueError("answered question run summary must be terminal")


@dataclass(frozen=True)
class QuestionRunProjection:
    primary_run_id: str | None
    latest_run_id: str | None
    active_run_id: str | None


@dataclass(frozen=True)
class QuestionMemoryRunSelection:
    question_id: str
    primary_run_id: str | None


def select_conversation_memory_runs(
    questions: tuple[QuestionMemoryRunSelection, ...],
    *,
    selected_run_id: str | None = None,
    selected_question_id: str | None = None,
) -> tuple[str, ...]:
    if (selected_run_id is None) != (selected_question_id is None):
        raise ValueError("selected memory run requires its question")
    run_ids: list[str] = []
    selected = False
    for question in questions:
        if question.question_id == selected_question_id:
            run_ids.append(selected_run_id)  # type: ignore[arg-type]
            selected = True
        elif question.primary_run_id is not None:
            run_ids.append(question.primary_run_id)
    if selected_run_id is not None and not selected:
        run_ids.insert(0, selected_run_id)
    return tuple(run_ids)


def project_question_runs(
    runs: tuple[QuestionRunSummary, ...],
) -> QuestionRunProjection:
    if not runs:
        return QuestionRunProjection(None, None, None)
    ordered = tuple(sorted(runs, key=lambda run: (run.run_number, run.run_id)))
    model_runs = tuple(
        run for run in ordered if run.kind is QuestionRunKind.MODEL_ASSISTED
    )
    answered_model_runs = tuple(run for run in model_runs if run.answered)
    primary = (
        answered_model_runs[-1]
        if answered_model_runs
        else (model_runs[-1] if model_runs else None)
    )
    active_runs = tuple(
        run
        for run in ordered
        if not run.terminal and run.status in _ACTIVE_STATUSES
    )
    return QuestionRunProjection(
        primary_run_id=primary.run_id if primary is not None else None,
        latest_run_id=ordered[-1].run_id,
        active_run_id=active_runs[-1].run_id if active_runs else None,
    )
