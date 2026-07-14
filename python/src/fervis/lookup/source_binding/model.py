"""Typed source-binding model for Lookup planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
)
from fervis.lookup.fact_plan.fact_plan import PlanClarification, PlanImpossible
from fervis.lookup.source_binding.compiler_ir import (
    DraftRelationSource,
    SourceAppliedFilter,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.question_contract import QuestionContract, RequestedFact
from fervis.lookup.read_eligibility import ReadEligibilityResult
from fervis.lookup.plan_selection import PlanSelectionSet
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.source_binding.candidates.contracts import (
    EntityEvidence,
)

if TYPE_CHECKING:
    from fervis.lookup.source_binding.candidates.model import SourceCandidateRegistry


@dataclass(frozen=True)
class SourceCandidateDiscoveryRequest:
    question: str
    question_contract: QuestionContract
    requested_facts: tuple[RequestedFact, ...]
    relation_catalog: RelationCatalog
    catalog_selection: CatalogSelectionResult
    same_scope_relation_catalog: RelationCatalog | None = None
    memory_inputs: dict[str, Any] = field(default_factory=dict)
    active_memory_ids: tuple[str, ...] = ()
    available_values: tuple[FactValue, ...] = ()
    available_value_uses: tuple[GroundedInputUse, ...] = ()
    read_eligibility: ReadEligibilityResult | None = None
    conversation_context: dict[str, Any] = field(default_factory=dict)
    conversation_resolution: CompiledConversationResolution | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)


@dataclass(frozen=True)
class SourceBindingRequest:
    question: str
    question_contract: QuestionContract
    requested_facts: tuple[RequestedFact, ...]
    relation_catalog: RelationCatalog
    catalog_selection: CatalogSelectionResult
    plan_selection: PlanSelectionSet
    source_candidates: SourceCandidateRegistry
    same_scope_relation_catalog: RelationCatalog | None = None
    memory_inputs: dict[str, Any] = field(default_factory=dict)
    active_memory_ids: tuple[str, ...] = ()
    available_values: tuple[FactValue, ...] = ()
    available_value_uses: tuple[GroundedInputUse, ...] = ()
    read_eligibility: ReadEligibilityResult | None = None
    conversation_context: dict[str, Any] = field(default_factory=dict)
    conversation_resolution: CompiledConversationResolution | None = None
    host: HostPromptContext = field(default_factory=HostPromptContext)


@dataclass(frozen=True)
class AnswerPopulation:
    population_binding_id: str
    intent_text: str
    match_basis_explanation: str

    def __post_init__(self) -> None:
        if not self.population_binding_id.strip():
            raise ValueError("answer population requires population binding id")
        if not self.intent_text.strip():
            raise ValueError("answer population requires intent text")
        if not self.match_basis_explanation.strip():
            raise ValueError("answer population requires match basis explanation")


@dataclass(frozen=True)
class SourceFulfillment:
    requested_fact_id: str
    answer_output_id: str
    match_basis_explanation: str
    fulfillment_support_set_id: str = ""
    entity_evidence: EntityEvidence | None = None
    value_evidence_ids: tuple[str, ...] = ()
    metric_measure_evidence_ids: tuple[str, ...] = ()
    row_count_basis_evidence_ids: tuple[str, ...] = ()
    metric_fit_bases: tuple["SourceMetricFitBasis", ...] = ()

    def __post_init__(self) -> None:
        if not self.requested_fact_id:
            raise ValueError("source fulfillment requires requested fact")
        if not self.answer_output_id:
            raise ValueError("source fulfillment requires answer output")
        if not (
            self.metric_measure_evidence_ids
            or self.value_evidence_ids
            or self.row_count_basis_evidence_ids
            or self.entity_evidence is not None
        ):
            raise ValueError("source fulfillment requires evidence")
        if not self.match_basis_explanation.strip():
            raise ValueError("source fulfillment requires match basis explanation")
        fulfillment_evidence_ids = set(self.all_evidence_ids())
        for basis in self.metric_fit_bases:
            if basis.evidence_id not in fulfillment_evidence_ids:
                raise ValueError(
                    "source metric fit basis must reference fulfillment evidence"
                )

    def all_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.metric_measure_evidence_ids,
                    *self.value_evidence_ids,
                    *self.row_count_basis_evidence_ids,
                    *(
                        (self.entity_evidence.evidence_id,)
                        if self.entity_evidence is not None
                        else ()
                    ),
                )
            )
        )

    def field_evidence_ids(self) -> tuple[str, ...]:
        if self.entity_evidence is None:
            return self.all_evidence_ids()
        return tuple(
            dict.fromkeys(
                (
                    *self.metric_measure_evidence_ids,
                    *self.value_evidence_ids,
                    *self.row_count_basis_evidence_ids,
                    *(
                        component.field_evidence_id
                        for component in self.entity_evidence.components
                    ),
                )
            )
        )


@dataclass(frozen=True)
class SourceMetricFitBasis:
    evidence_id: str
    metric_meaning: str
    fit_basis: str

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("source metric fit basis requires evidence id")
        if not self.metric_meaning.strip():
            raise ValueError("source metric fit basis requires metric meaning")
        if not self.fit_basis.strip():
            raise ValueError("source metric fit basis requires fit basis")


@dataclass(frozen=True)
class SourceField:
    field_id: str
    type: str = ""
    roles: tuple[str, ...] = ()
    label: str = ""
    row_cardinality: str = ""

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("source field requires field id")


@dataclass(frozen=True)
class SourceEvidenceItem:
    evidence_id: str
    field_id: str = ""
    value_id: str = ""
    type: str = ""
    row_cardinality: str = ""
    row_source_id: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("source evidence item requires evidence id")
        if not self.field_id and not self.value_id:
            raise ValueError("source evidence item requires field or value")
        if self.type == "row_population" and not self.row_source_id:
            raise ValueError("row population evidence requires row source")


@dataclass(frozen=True)
class BoundSource:
    id: str
    requested_fact_id: str = ""
    binding_target_id: str = ""
    requirement_id: str = ""
    answer_population: AnswerPopulation | None = None
    source: DraftRelationSource | None = None
    source_invocations: tuple[DraftRelationSource, ...] = ()
    value_id: str = ""
    source_candidate_id: str = ""
    cardinality: str = ""
    fulfillments: tuple[SourceFulfillment, ...] = ()
    evidence_items: tuple[SourceEvidenceItem, ...] = ()
    available_field_ids: tuple[str, ...] = ()
    available_fields: tuple[SourceField, ...] = ()
    applied_filters: tuple[SourceAppliedFilter, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("bound source requires id")
        if not self.requested_fact_id and not self.fulfillments:
            raise ValueError("bound source requires requested fact or fulfillment")
        if self.answer_population is None:
            raise ValueError("bound source requires answer population")
        if self.source is None and not self.value_id:
            raise ValueError("bound source requires relation source or value id")
        if self.source is not None and self.value_id:
            raise ValueError(
                "bound source cannot have both relation source and value id"
            )
        if self.source_invocations and self.source is None:
            raise ValueError("source invocations require relation source")

    @property
    def is_auxiliary_value(self) -> bool:
        return (
            bool(self.value_id)
            and not self.binding_target_id
            and self.source is None
            and not self.fulfillments
        )


@dataclass(frozen=True)
class SourceBindingPlan:
    bound_sources: tuple[BoundSource, ...]

    def __post_init__(self) -> None:
        if not self.bound_sources:
            raise ValueError("source binding requires at least one bound source")


SourceBindingOutcome = SourceBindingPlan | PlanClarification | PlanImpossible


@dataclass(frozen=True)
class SourceBindingResult:
    outcome: SourceBindingOutcome
