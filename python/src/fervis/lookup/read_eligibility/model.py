"""Typed read-eligibility retention contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.conversation_resolution.overlay import (
    ConversationResolutionOverlay,
)
from fervis.lookup.fact_plan.values import FactValue
from fervis.lookup.question_contract import QuestionContract, RequestedFact
from fervis.lookup.turn_prompts.context import HostPromptContext


READ_ELIGIBILITY_RECALL_READS_PER_FACT = 10
RETENTION_DECISION_VALUES = ("RETAIN", "DROP")


@dataclass(frozen=True)
class ReadEligibilityRequest:
    question: str
    question_contract: QuestionContract
    requested_facts: tuple[RequestedFact, ...]
    catalog_selection: CatalogSelectionResult
    conversation_context: dict[str, Any]
    available_values: tuple[FactValue, ...]
    conversation_resolution_overlay: ConversationResolutionOverlay | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)


@dataclass(frozen=True)
class ReadAssessment:
    source_candidate_id: str
    source_candidate_signature: str
    requested_fact_id: str
    read_id: str
    retention_decision: str
    relevant_row_path_ids: tuple[str, ...] = ()
    relevant_field_refs: tuple[str, ...] = ()
    retention_basis: str = ""

    def __post_init__(self) -> None:
        if not self.source_candidate_id.strip():
            raise ValueError("read assessment requires source candidate id")
        if not self.source_candidate_signature.strip():
            raise ValueError("read assessment requires source candidate signature")
        if not self.requested_fact_id.strip():
            raise ValueError("read assessment requires requested fact id")
        if not self.read_id.strip():
            raise ValueError("read assessment requires read id")
        if self.retention_decision not in RETENTION_DECISION_VALUES:
            raise ValueError("unsupported retention decision")
        if not self.retention_basis.strip():
            raise ValueError("read assessment requires retention basis")
        if len(set(self.relevant_row_path_ids)) != len(self.relevant_row_path_ids):
            raise ValueError("read assessment repeats row path id")
        if len(set(self.relevant_field_refs)) != len(self.relevant_field_refs):
            raise ValueError("read assessment repeats field ref")
        if not self.is_retained and (
            self.relevant_row_path_ids or self.relevant_field_refs
        ):
            raise ValueError("dropped read must not include evidence hints")

    @property
    def is_retained(self) -> bool:
        return self.retention_decision == "RETAIN"


@dataclass(frozen=True)
class ReadEligibilityResult:
    read_assessments: tuple[ReadAssessment, ...]

    def retained_read_ids_by_requested_fact(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for item in self.read_assessments:
            if not item.is_retained:
                continue
            if item.read_id not in output.setdefault(item.requested_fact_id, []):
                output[item.requested_fact_id].append(item.read_id)
        return {key: tuple(value) for key, value in output.items()}
