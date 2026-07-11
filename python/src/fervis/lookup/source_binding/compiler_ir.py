"""Compiler-front source inputs that are closed before answer-program creation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from fervis.lookup.answer_program.relations import (
    PopulationChoiceControllerKind,
    RelationSourceReviewScopeDecision,
    SourceKind,
)
from fervis.lookup.answer_program.values import (
    ParameterRef,
    ValueExpression,
)


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
        if not isinstance(self.value, tuple) or self.value_item_index >= len(self.value):
            raise ValueError("endpoint param binding item is outside its value")
        return self.value[self.value_item_index]


@dataclass(frozen=True)
class DraftRelationSourceAppliedFilter:
    predicate_field_ids: tuple[str, ...]
    known_input_id: str
    value_id: str = ""
    value_expr: ValueExpression | None = None
    value_kind: str = ""
    identity_type: str = ""

    def __post_init__(self) -> None:
        if not self.predicate_field_ids:
            raise ValueError("relation source applied filter requires predicate fields")
        if not self.known_input_id and self.value_expr is None:
            raise ValueError("relation source applied filter requires input expression")
        if self.known_input_id and self.value_expr is not None:
            raise ValueError(
                "relation source applied filter cannot contain input and expression"
            )

    @classmethod
    def from_payloads(
        cls,
        payloads: Iterable[Mapping[str, Any]],
    ) -> tuple[DraftRelationSourceAppliedFilter, ...]:
        filters: list[DraftRelationSourceAppliedFilter] = []
        for payload in payloads:
            item = cls.from_payload(payload)
            if item is not None:
                filters.append(item)
        return tuple(filters)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> DraftRelationSourceAppliedFilter | None:
        predicate_field_ids = tuple(
            str(field_id)
            for field_id in payload.get("field_ids") or ()
            if str(field_id)
        )
        if not predicate_field_ids:
            return None
        return cls(
            predicate_field_ids=predicate_field_ids,
            known_input_id=str(payload.get("known_input_id") or ""),
            value_id=str(payload.get("value_id") or ""),
            value_kind=str(payload.get("kind") or ""),
            identity_type=str(payload.get("identity_type") or ""),
        )


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
