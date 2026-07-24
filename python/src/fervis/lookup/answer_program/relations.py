"""Closed relation model for canonical executable answer programs."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum

from fervis.lookup.answer_program.expressions import Expression


class FieldBindingRole(StrEnum):
    IDENTITY = "identity"
    OUTPUT = "output"
    PREDICATE = "predicate"


class SourceKind(StrEnum):
    API_READ = "api_read"
    GENERATED_CALENDAR = "generated_calendar"
    MEMORY_READ = "memory_read"


@dataclass(frozen=True)
class EndpointParamBinding:
    param_id: str
    value_expr: Expression
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.param_id:
            raise ValueError("endpoint param binding requires param")


@dataclass(frozen=True)
class RelationSource:
    kind: SourceKind
    read_id: str = ""
    row_source_id: str = ""
    calendar_id: str = ""
    memory_relation_id: str = ""
    param_bindings: tuple[EndpointParamBinding, ...] = ()
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
