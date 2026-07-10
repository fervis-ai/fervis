"""Closed relation model for canonical executable answer programs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fervis.lookup.answer_program.values import ParameterRef, ValueExpression


class FieldBindingRole(StrEnum):
    IDENTITY = "identity"
    OUTPUT = "output"
    PREDICATE = "predicate"


class SourceKind(StrEnum):
    API_READ = "api_read"
    GENERATED_CALENDAR = "generated_calendar"
    MEMORY_READ = "memory_read"


class PopulationChoiceControllerKind(StrEnum):
    QUERY_PARAM = "query_param"
    ROW_PREDICATE = "row_predicate"


class ReviewScopeDecisionKind(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class RelationSourceReviewScopeDecision:
    membership_test_id: str
    decision: ReviewScopeDecisionKind
    axis_kind: str
    axis_id: str
    owner_surface_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.membership_test_id:
            raise ValueError("review scope decision requires membership test")
        if not self.axis_kind:
            raise ValueError("review scope decision requires axis kind")
        if not self.axis_id:
            raise ValueError("review scope decision requires axis id")


@dataclass(frozen=True)
class EndpointParamBinding:
    param_id: str
    value_expr: ValueExpression
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.param_id:
            raise ValueError("endpoint param binding requires param")


@dataclass(frozen=True)
class RelationSourceAppliedFilter:
    predicate_field_ids: tuple[str, ...]
    value_expr: ValueExpression

    def __post_init__(self) -> None:
        if not self.predicate_field_ids:
            raise ValueError("relation source applied filter requires predicate fields")


@dataclass(frozen=True)
class RelationSourceRowFilter:
    field_id: str
    operator: str
    value_expr: ValueExpression
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("relation source row filter requires field")
        if not self.operator:
            raise ValueError("relation source row filter requires operator")


@dataclass(frozen=True)
class RelationSourcePopulationChoice:
    controller_kind: PopulationChoiceControllerKind
    controller_id: str
    field_id: str
    requested_fact_ids: tuple[str, ...]
    selection_expr: ParameterRef
    allowed_values: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    review_scope_decisions: tuple[RelationSourceReviewScopeDecision, ...] = ()

    def __post_init__(self) -> None:
        if not self.controller_id:
            raise ValueError("relation source population choice requires controller")
        if not self.field_id:
            raise ValueError("relation source population choice requires field")
        if not self.requested_fact_ids:
            raise ValueError(
                "relation source population choice requires requested facts"
            )
        if len(set(self.requested_fact_ids)) != len(self.requested_fact_ids):
            raise ValueError(
                "relation source population choice requested facts must be unique"
            )


@dataclass(frozen=True)
class RelationSource:
    kind: SourceKind
    read_id: str = ""
    row_source_id: str = ""
    calendar_id: str = ""
    memory_relation_id: str = ""
    param_bindings: tuple[EndpointParamBinding, ...] = ()
    applied_filters: tuple[RelationSourceAppliedFilter, ...] = ()
    row_filters: tuple[RelationSourceRowFilter, ...] = ()
    population_choices: tuple[RelationSourcePopulationChoice, ...] = ()
    proof_refs: tuple[str, ...] = ()

@dataclass(frozen=True)
class RelationField:
    field_id: str
    roles: tuple[FieldBindingRole, ...]


@dataclass(frozen=True)
class Relation:
    id: str
    source: RelationSource
    fields: tuple[RelationField, ...] = ()

    @property
    def grain_keys(self) -> tuple[str, ...]:
        return tuple(
            item.field_id
            for item in self.fields
            if FieldBindingRole.IDENTITY in item.roles
        )

    def field(self, field_id: str) -> RelationField | None:
        for item in self.fields:
            if item.field_id == field_id:
                return item
        return None
