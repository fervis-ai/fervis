"""Canonical lookup clarification model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ClarificationNeed(StrEnum):
    TARGET_REFERENCE = "target_reference"
    ANSWER_METRIC = "answer_metric"
    COMPARISON_BASELINE = "comparison_baseline"
    CATALOG_INPUT = "catalog_input"
    QUESTION_INTERPRETATION = "question_interpretation"


class ClarificationReason(StrEnum):
    UNRESOLVED_REFERENCE = "unresolved_reference"
    MULTIPLE_MATCHING_ENTITIES = "multiple_matching_entities"
    UNSUPPORTED_REFERENCE = "unsupported_reference"
    MISSING_ANSWER_METRIC = "missing_answer_metric"
    MISSING_COMPARISON_BASELINE = "missing_comparison_baseline"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    CATALOG_REQUIRES_CHOICE = "catalog_requires_choice"
    AMBIGUOUS_INTERPRETATION = "ambiguous_interpretation"


class ClarificationSubjectKind(StrEnum):
    QUESTION_INPUT = "question_input"
    CATALOG_INPUT = "catalog_input"
    CATALOG_CHOICE = "catalog_choice"
    METRIC_PHRASE = "metric_phrase"
    COMPARISON_PHRASE = "comparison_phrase"
    INTERPRETATION = "interpretation"


class ClarificationEvidenceKind(StrEnum):
    KNOWN_INPUT = "known_input"
    RESOLVER_READ = "resolver_read"
    GROUNDING = "grounding"
    QUESTION_CONTRACT = "question_contract"
    CATALOG_INPUT = "catalog_input"
    CANDIDATE = "candidate"
    PROOF_REF = "proof_ref"


@dataclass(frozen=True)
class ClarificationOption:
    id: str
    label: str = ""
    value: str = ""
    entity_kind: str = ""
    matched_label: str = ""
    matched_field: str = ""
    matched_value: str = ""
    resolver_read_id: str = ""
    resolver_label: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("clarification option requires id")


@dataclass(frozen=True)
class ClarificationSubject:
    kind: ClarificationSubjectKind
    id: str
    label: str = ""
    source_text: str = ""
    options: tuple[ClarificationOption, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ClarificationSubjectKind):
            raise ValueError("clarification subject requires structured kind")
        if not self.id:
            raise ValueError("clarification subject requires id")


@dataclass(frozen=True)
class ClarificationEvidence:
    kind: ClarificationEvidenceKind
    id: str
    read_id: str = ""
    endpoint_name: str = ""
    field_id: str = ""
    identity_field: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ClarificationEvidenceKind):
            raise ValueError("clarification evidence requires structured kind")
        if not self.id:
            raise ValueError("clarification evidence requires id")


@dataclass(frozen=True)
class Clarification:
    id: str
    requested_fact_id: str
    need: ClarificationNeed
    reason: ClarificationReason
    subjects: tuple[ClarificationSubject, ...]
    evidence: tuple[ClarificationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("clarification requires id")
        if not self.requested_fact_id:
            raise ValueError("clarification requires requested fact")
        if not isinstance(self.need, ClarificationNeed):
            raise ValueError("clarification requires structured need")
        if not isinstance(self.reason, ClarificationReason):
            raise ValueError("clarification requires structured reason")
        if not self.subjects:
            raise ValueError("clarification requires at least one subject")
