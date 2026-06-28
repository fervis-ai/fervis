"""Public contracts for path-loaded Fervis goldset suites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fervis.questions import AskResult


@dataclass(frozen=True)
class GoldsetCase:
    case_id: str
    question: str
    setup_questions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_questions", tuple(self.setup_questions))


@dataclass(frozen=True)
class GoldsetMatch:
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldsetCaseResult:
    case_id: str
    status: str
    question: str
    conversation_id: str
    question_id: str
    run_id: str
    answer: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "question": self.question,
            "conversation_id": self.conversation_id,
            "question_id": self.question_id,
            "run_id": self.run_id,
            "answer": self.answer,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class GoldsetRunResult:
    suite_name: str
    case_count: int
    passed_count: int
    failed_count: int
    cases: tuple[GoldsetCaseResult, ...]

    @property
    def exit_code(self) -> int:
        return 0 if self.failed_count == 0 else 1

    def to_payload(self) -> dict[str, object]:
        return {
            "suite_name": self.suite_name,
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "cases": [case.to_payload() for case in self.cases],
        }


@dataclass(frozen=True)
class GoldsetSuite:
    name: str
    cases: tuple[GoldsetCase, ...]
    match_answer: Callable[[GoldsetCase, AskResult], GoldsetMatch]
    prepare_case: Callable[[GoldsetCase], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
