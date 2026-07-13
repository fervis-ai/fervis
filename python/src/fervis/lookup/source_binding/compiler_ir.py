"""Compiler-front source inputs that are closed before answer-program creation."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import TypeAlias

from fervis.lookup.answer_program.relations import (
    PopulationChoiceControllerKind,
    RelationSourceReviewScopeDecision,
    SourceKind,
)
from fervis.lookup.answer_program.values import (
    ParameterRef,
    ValueExpression,
)


_SourceAppliedFilterPayloadValue: TypeAlias = str | list[str]


class RelationInputOrigin(StrEnum):
    QUESTION_INPUT = "question_input"
    SEMANTIC_CONTROL = "semantic_control"
    CONTEXT_CONSTANT = "context_constant"


@dataclass(frozen=True)
class DraftEndpointParamBinding:
    param_id: str
    value: object | None = None
    value_expr: ValueExpression | None = None
    origin_kind: RelationInputOrigin = RelationInputOrigin.CONTEXT_CONSTANT
    value_id: str = ""
    value_component: str = "value"
    value_item_index: int | None = None
    parameter_id: str = ""
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.param_id:
            raise ValueError("endpoint param binding requires param")
        if (self.value is None) == (self.value_expr is None):
            raise ValueError(
                "endpoint param binding requires exactly one value or expression"
            )

    @property
    def compiler_value(self) -> object | None:
        if self.value_item_index is None:
            return self.value
        if not isinstance(self.value, tuple) or self.value_item_index >= len(
            self.value
        ):
            raise ValueError("endpoint param binding item is outside its value")
        return self.value[self.value_item_index]


@dataclass(frozen=True)
class DraftRelationSourceAppliedFilter:
    predicate_field_ids: tuple[str, ...]
    known_input_id: str
    value_id: str = ""
    value_expr: ValueExpression | None = None
    value_kind: str = ""
    operator: str = "equals"

    def __post_init__(self) -> None:
        if not self.predicate_field_ids:
            raise ValueError("relation source applied filter requires predicate fields")
        if not self.known_input_id and self.value_expr is None:
            raise ValueError("relation source applied filter requires input expression")
        if self.known_input_id and self.value_expr is not None:
            raise ValueError(
                "relation source applied filter cannot contain input and expression"
            )

@dataclass(frozen=True)
class SourceAppliedFilter:
    known_input_id: str
    predicate_field_ids: tuple[str, ...] = ()
    value_id: str = ""
    value_kind: str = ""
    display_value: str = ""
    matched_field_ref: str = ""
    matched_field_path: str = ""
    resolved_start: str = ""
    resolved_end: str = ""
    literal_type: str = ""
    operator: str = "equals"

    def __post_init__(self) -> None:
        if not self.known_input_id and not self.value_id:
            raise ValueError("source applied filter requires a value authority")

    def relation_filter(self) -> DraftRelationSourceAppliedFilter | None:
        if not self.predicate_field_ids:
            return None
        return DraftRelationSourceAppliedFilter(
            predicate_field_ids=self.predicate_field_ids,
            known_input_id=self.known_input_id,
            value_id=self.value_id,
            value_kind=self.value_kind,
            operator=self.operator,
        )

    def to_payload(self) -> dict[str, _SourceAppliedFilterPayloadValue]:
        payload: dict[str, _SourceAppliedFilterPayloadValue] = {}
        if self.predicate_field_ids:
            payload["field_ids"] = list(self.predicate_field_ids)
        if self.known_input_id:
            payload["known_input_id"] = self.known_input_id
        if self.value_id:
            payload["value_id"] = self.value_id
        if self.value_kind:
            payload["kind"] = self.value_kind
        if self.operator != "equals":
            payload["operator"] = self.operator
        for key, value in (
            ("display_value", self.display_value),
            ("matched_field_ref", self.matched_field_ref),
            ("matched_field_path", self.matched_field_path),
            ("resolved_start", self.resolved_start),
            ("resolved_end", self.resolved_end),
            ("literal_type", self.literal_type),
        ):
            if value:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class DraftRelationSourceRowFilter:
    field_id: str
    operator: str
    values: tuple[object, ...] = ()
    value_expr: ValueExpression | None = None
    parameter_id: str = ""
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("relation source row filter requires field")
        if not self.operator:
            raise ValueError("relation source row filter requires operator")
        if bool(self.values) == (self.value_expr is not None):
            raise ValueError(
                "relation source row filter requires values or one expression"
            )


@dataclass(frozen=True)
class DraftRelationSourcePopulationChoice:
    controller_kind: PopulationChoiceControllerKind
    controller_id: str
    field_id: str
    requested_fact_ids: tuple[str, ...]
    included_values: tuple[str, ...] = ()
    excluded_values: tuple[str, ...] = ()
    selection_expr: ParameterRef | None = None
    allowed_values: tuple[str, ...] = ()
    parameter_id: str = ""
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
        if not self.included_values and self.selection_expr is None:
            raise ValueError(
                "relation source population choice requires included values"
            )
        if self.included_values and self.selection_expr is not None:
            raise ValueError(
                "relation source population choice cannot contain values and expression"
            )
        if set(self.included_values) & set(self.excluded_values):
            raise ValueError("relation source population choice values cannot overlap")


@dataclass(frozen=True)
class DraftRelationSource:
    kind: SourceKind
    read_id: str = ""
    row_source_id: str = ""
    calendar_id: str = ""
    memory_relation_id: str = ""
    param_bindings: tuple[DraftEndpointParamBinding, ...] = ()
    applied_filters: tuple[DraftRelationSourceAppliedFilter, ...] = ()
    row_filters: tuple[DraftRelationSourceRowFilter, ...] = ()
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...] = ()
    proof_refs: tuple[str, ...] = ()
