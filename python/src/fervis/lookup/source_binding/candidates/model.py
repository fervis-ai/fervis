"""Typed source-binding candidate registry models."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    SourceAppliedFilter,
)

from fervis.lookup.source_binding.candidates.contracts import (
    EntityTarget,
    EvidenceItem,
    FulfillmentSupportSet,
    JsonObject,
    JsonValue,
)


@dataclass(frozen=True)
class CandidateBindingValue:
    value: str
    label: str = ""
    source: str = ""
    value_component: str = ""


@dataclass(frozen=True)
class CandidateParamDecision:
    id: str
    decision: str
    value: str = ""
    value_component: str = ""


@dataclass(frozen=True)
class CandidateNormalInstanceProfile:
    test_id: str
    excluded_role_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateParameter:
    id: str
    type: str
    required: bool
    choices: tuple[str, ...]
    decision_options: tuple[CandidateParamDecision, ...]
    binding_values: tuple[CandidateBindingValue, ...] = ()
    entity_target: EntityTarget | None = None
    has_default: bool = False
    default: JsonValue = None
    finite_choice_review: bool = False
    omission_kind: str = ""
    omission_default_value: str = ""
    normal_instance_profiles: tuple[CandidateNormalInstanceProfile, ...] = ()
    owned_membership_test_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidatePopulationBinding:
    id: str
    kind: str
    memory_relation_id: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRowPredicate:
    id: str
    field_id: str
    field_type: str
    operator: str
    allowed_values: tuple[str, ...]
    owned_membership_test_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceCandidate:
    id: str
    applies_to_requested_fact_ids: tuple[str, ...]
    kind: str
    source: DraftRelationSource | None = None
    value_id: str = ""
    source_relation_id: str = ""
    source_field_id: str = ""
    cardinality: str = ""
    result_row_path_ids: tuple[str, ...] = ()
    params: tuple[CandidateParameter, ...] = ()
    applied_param_bindings: tuple[DraftEndpointParamBinding, ...] = ()
    applied_param_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...] = ()
    applied_filters: tuple[SourceAppliedFilter, ...] = ()
    evidence_items: tuple[EvidenceItem, ...] = ()
    fulfillment_support_sets: tuple[FulfillmentSupportSet, ...] = ()
    population_bindings: tuple[CandidatePopulationBinding, ...] = ()
    row_predicates: tuple[CandidateRowPredicate, ...] = ()
    population_role_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("source candidate requires id")
        if not self.kind:
            raise ValueError("source candidate requires kind")
        if not self.applies_to_requested_fact_ids:
            raise ValueError("source candidate requires requested-fact applicability")

    @property
    def read_id(self) -> str:
        return self.source.read_id if self.source is not None else ""

    @property
    def memory_relation_id(self) -> str:
        return self.source.memory_relation_id if self.source is not None else ""

    @property
    def calendar_id(self) -> str:
        return self.source.calendar_id if self.source is not None else ""

    @property
    def applied_filter_field_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                field_id
                for applied_filter in self.applied_filters
                for field_id in applied_filter.predicate_field_ids
            )
        )


@dataclass(frozen=True)
class SourceCandidateRegistry:
    prompt_payload: JsonObject
    candidates_by_id: dict[str, SourceCandidate]
    prompt_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(key != candidate.id for key, candidate in self.candidates_by_id.items()):
            raise ValueError("source candidate registry key mismatch")
        if not set(self.prompt_candidate_ids) <= set(self.candidates_by_id):
            raise ValueError("source candidate prompt references unknown candidate")

    def candidates_for(self, requested_fact_id: str) -> tuple[SourceCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates_by_id.values()
            if requested_fact_id in candidate.applies_to_requested_fact_ids
        )
