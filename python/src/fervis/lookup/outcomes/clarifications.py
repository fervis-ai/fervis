"""Shared structured clarification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ClarificationBasis(StrEnum):
    MISSING_ANSWER_METRIC = "missing_answer_metric"
    MISSING_COMPARISON_BASELINE = "missing_comparison_baseline"
    CATALOG_REQUIRES_CHOICE = "catalog_requires_choice"
    MULTIPLE_MATCHING_ENTITIES = "multiple_matching_entities"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    UNSUPPORTED_REFERENCE = "unsupported_reference"


@dataclass(frozen=True)
class ClarificationOption:
    id: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("clarification option requires id")


@dataclass(frozen=True)
class Clarification:
    id: str
    requested_fact_id: str
    basis: ClarificationBasis
    question: str
    known_input_id: str = ""
    required_catalog_input_id: str = ""
    required_catalog_choice_input_id: str = ""
    available_options: tuple[ClarificationOption, ...] = ()
    candidate_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    ambiguous_metric_phrase: str = ""
    metric_needed_to_answer: str = ""
    comparison_phrase: str = ""
    comparison_baseline_needed: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.basis, ClarificationBasis):
            raise ValueError("clarification requires structured basis")
        if not self.id:
            raise ValueError("clarification requires id")
        if not self.requested_fact_id:
            raise ValueError("clarification requires requested fact")
        if not self.question:
            raise ValueError("clarification requires question")
