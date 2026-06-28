"""Relation model for row-based fact extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping


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


@dataclass(frozen=True)
class EndpointParamBinding:
    param_id: str
    value: object
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationSourceAppliedFilter:
    predicate_field_ids: tuple[str, ...]
    known_input_id: str
    value_kind: str = ""
    identity_type: str = ""

    def __post_init__(self) -> None:
        if not self.predicate_field_ids:
            raise ValueError("relation source applied filter requires predicate fields")
        if not self.known_input_id:
            raise ValueError("relation source applied filter requires known input")

    @classmethod
    def from_payloads(
        cls,
        payloads: Iterable[Mapping[str, Any]],
    ) -> tuple["RelationSourceAppliedFilter", ...]:
        filters: list[RelationSourceAppliedFilter] = []
        for payload in payloads:
            item = cls.from_payload(payload)
            if item is not None:
                filters.append(item)
        return tuple(filters)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RelationSourceAppliedFilter | None":
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
            value_kind=str(payload.get("kind") or ""),
            identity_type=str(payload.get("identity_type") or ""),
        )


@dataclass(frozen=True)
class RelationSourceRowFilter:
    field_id: str
    operator: str
    values: tuple[object, ...]
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("relation source row filter requires field")
        if not self.operator:
            raise ValueError("relation source row filter requires operator")
        if not self.values:
            raise ValueError("relation source row filter requires values")


@dataclass(frozen=True)
class RelationSourcePopulationChoice:
    controller_kind: PopulationChoiceControllerKind
    controller_id: str
    field_id: str
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...]
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.controller_id:
            raise ValueError("relation source population choice requires controller")
        if not self.field_id:
            raise ValueError("relation source population choice requires field")
        if not self.included_values:
            raise ValueError(
                "relation source population choice requires included values"
            )
        if set(self.included_values) & set(self.excluded_values):
            raise ValueError("relation source population choice values cannot overlap")


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
