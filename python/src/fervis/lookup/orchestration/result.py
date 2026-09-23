"""Lookup runtime result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fervis.lookup.outcomes.model import FactResult
    from fervis.lookup.answer_rendering import RenderedFact


class RunStatus:
    COMPLETED = "COMPLETED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    FAILED = "FAILED"


class AnswerSource:
    RENDERED_FACT = "rendered_fact"
    PLANNER_TERMINAL = "planner_terminal"


@dataclass(frozen=True)
class LookupResult:
    status: str
    answer: str = ""
    result_data: dict[str, Any] | None = None
    fact_result: FactResult | None = None
    rendered_fact: RenderedFact | None = None
    fact_addresses: tuple[dict[str, Any], ...] = ()
    fact_outcome_addresses: tuple[dict[str, Any], ...] = ()
    error: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerRunResult:
    status: str
    answer: str | None = None
    result_data: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    answer_source: str = AnswerSource.RENDERED_FACT
