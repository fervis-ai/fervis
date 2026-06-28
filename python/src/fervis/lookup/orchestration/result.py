"""Lookup runtime result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fervis.observability.event_contracts import EventPayloadKey

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


_DELIVERY_RESULT_AUDIT_KEYS = frozenset({EventPayloadKey.PROOF_REFS, "proof_refs"})


def delivery_result_data(result_data: Any) -> dict[str, Any] | None:
    if not isinstance(result_data, dict):
        return None
    return _strip_delivery_result_audit_keys(result_data)


def _strip_delivery_result_audit_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_delivery_result_audit_keys(item)
            for key, item in value.items()
            if key not in _DELIVERY_RESULT_AUDIT_KEYS
        }
    if isinstance(value, list):
        return [_strip_delivery_result_audit_keys(item) for item in value]
    return value
