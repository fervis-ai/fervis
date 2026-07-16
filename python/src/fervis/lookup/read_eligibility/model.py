"""Typed read-eligibility retention contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.grounding.model import (
    CanonicalInputLedger,
    CompatibleInputBinding,
    KnownInputBindingTask,
)
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
    binding_tasks: tuple[KnownInputBindingTask, ...] = ()
    compatible_reference_bindings: tuple[CompatibleInputBinding, ...] = ()
    canonical_values: tuple[FactValue, ...] = ()
    resolver_catalog: RelationCatalog | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)

    def __post_init__(self) -> None:
        if self.compatible_reference_bindings and self.resolver_catalog is None:
            raise ValueError(
                "read eligibility resolver options require their relation catalog"
            )


@dataclass(frozen=True)
class CanonicalInputOption:
    id: str
    requested_fact_id: str
    known_input_id: str
    known_input_token: str
    entity_kind: str
    key_id: str
    component_ids: tuple[str, ...]
    resolver_binding: CompatibleInputBinding | None = None
    canonical_value_id: str = ""

    def __post_init__(self) -> None:
        if (
            not self.id
            or not self.requested_fact_id
            or not self.known_input_id
            or not self.known_input_token
            or not self.entity_kind
            or not self.key_id
            or not self.component_ids
        ):
            raise ValueError("canonical input option is incomplete")
        if (self.resolver_binding is not None) == bool(self.canonical_value_id):
            raise ValueError("canonical input option requires one authority")

    @property
    def authority_id(self) -> str:
        if self.resolver_binding is not None:
            return self.resolver_binding.option_id
        return self.canonical_value_id

    @property
    def result(self) -> tuple[str, str]:
        return (self.entity_kind, self.key_id)


@dataclass(frozen=True)
class CanonicalInputSelection:
    option: CanonicalInputOption
    interpretation_question: str
    because: str

    def __post_init__(self) -> None:
        if not self.interpretation_question.strip() or not self.because.strip():
            raise ValueError("canonical input selection requires its assessment")

    @property
    def requested_fact_id(self) -> str:
        return self.option.requested_fact_id

    @property
    def known_input_id(self) -> str:
        return self.option.known_input_id

    @property
    def known_input_token(self) -> str:
        return self.option.known_input_token

    @property
    def canonical_option_id(self) -> str:
        return self.option.id


@dataclass(frozen=True)
class _ReadAssessmentBase:
    source_candidate_id: str
    source_candidate_signature: str
    requested_fact_id: str
    read_id: str
    retention_basis: str

    def __post_init__(self) -> None:
        if not self.source_candidate_id.strip():
            raise ValueError("read assessment requires source candidate id")
        if not self.source_candidate_signature.strip():
            raise ValueError("read assessment requires source candidate signature")
        if not self.requested_fact_id.strip():
            raise ValueError("read assessment requires requested fact id")
        if not self.read_id.strip():
            raise ValueError("read assessment requires read id")
        if not self.retention_basis.strip():
            raise ValueError("read assessment requires retention basis")


@dataclass(frozen=True)
class RetainedReadAssessment(_ReadAssessmentBase):
    relevant_row_path_ids: tuple[str, ...]
    relevant_field_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(set(self.relevant_row_path_ids)) != len(self.relevant_row_path_ids):
            raise ValueError("read assessment repeats row path id")
        if len(set(self.relevant_field_refs)) != len(self.relevant_field_refs):
            raise ValueError("read assessment repeats field ref")

    @property
    def is_retained(self) -> bool:
        return True

    @property
    def retention_decision(self) -> str:
        return "RETAIN"


@dataclass(frozen=True)
class DroppedReadAssessment(_ReadAssessmentBase):
    @property
    def is_retained(self) -> bool:
        return False

    @property
    def retention_decision(self) -> str:
        return "DROP"


ReadAssessment = RetainedReadAssessment | DroppedReadAssessment


@dataclass(frozen=True)
class ReadEligibilityResult:
    read_assessments: tuple[ReadAssessment, ...]
    canonical_inputs: tuple[CanonicalInputSelection, ...] = ()

    def retained_read_ids_by_requested_fact(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for item in self.read_assessments:
            if not item.is_retained:
                continue
            if item.read_id not in output.setdefault(item.requested_fact_id, []):
                output[item.requested_fact_id].append(item.read_id)
        return {key: tuple(value) for key, value in output.items()}


@dataclass(frozen=True)
class ResolvedRetainedReadSet:
    retained_reads: tuple[RetainedReadAssessment, ...]
    ledger: CanonicalInputLedger = CanonicalInputLedger()

    def retained_read_ids_by_requested_fact(self) -> dict[str, tuple[str, ...]]:
        output: dict[str, list[str]] = {}
        for item in self.retained_reads:
            reads = output.setdefault(item.requested_fact_id, [])
            if item.read_id not in reads:
                reads.append(item.read_id)
        return {key: tuple(value) for key, value in output.items()}
