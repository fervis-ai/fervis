"""Canonical result projection for answer-program outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from fervis.lookup.canonical_data import (
    EntityKeyComponentValue,
    EntityKeyValue,
    ResultValue,
    RuntimeValue,
)


class ResultProjectionError(ValueError):
    """A declared answer projection cannot be applied to its result data."""


def verify_entity_display_type(display_field_id: str, field_types: Mapping[str, str]) -> None:
    """Validate presentation types wherever a result projection is authored or used."""
    if display_field_id and field_types.get(display_field_id) not in {"string", "text"}:
        raise ResultProjectionError("entity display field must be textual")


@dataclass(frozen=True)
class ProjectedResultRow:
    relation_id: str
    row_index: int
    values: Mapping[str, ResultValue]
    display_values: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityKeyProjectionComponent:
    component_id: str
    field_id: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.field_id:
            raise ResultProjectionError("entity key projection component is incomplete")


@dataclass(frozen=True)
class EntityKeyProjection:
    entity_kind: str
    key_id: str
    components: tuple[EntityKeyProjectionComponent, ...]

    def __post_init__(self) -> None:
        if not self.entity_kind:
            raise ResultProjectionError("entity key projection requires entity kind")
        if not self.key_id:
            raise ResultProjectionError("entity key projection requires key id")
        if not self.components:
            raise ResultProjectionError("entity key projection requires components")
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ResultProjectionError(
                "entity key projection component ids must be unique"
            )

    def project(self, row: Mapping[str, RuntimeValue]) -> EntityKeyValue:
        missing_fields = tuple(
            component.field_id
            for component in self.components
            if component.field_id not in row
        )
        if missing_fields:
            raise ResultProjectionError("entity key field is unavailable")
        return EntityKeyValue(
            entity_kind=self.entity_kind,
            key_id=self.key_id,
            components=tuple(
                EntityKeyComponentValue(
                    component_id=component.component_id,
                    value=row[component.field_id],
                )
                for component in self.components
            ),
        )


@dataclass(frozen=True)
class RelationResultOutput:
    id: str
    relation_id: str
    field_id: str = ""
    entity_key: EntityKeyProjection | None = None
    label: str = ""
    role: str = ""
    display_field_id: str = ""

    def __post_init__(self) -> None:
        if self.display_field_id and self.entity_key is None:
            raise ResultProjectionError("display fields require an entity output")
        if bool(self.field_id) == bool(self.entity_key):
            raise ResultProjectionError(
                "relation result output requires exactly one field or entity key"
            )

    @property
    def source_node_id(self) -> str:
        return f"relation:{self.relation_id}"

    def project(self, row: Mapping[str, RuntimeValue]) -> ResultValue:
        if self.entity_key is not None:
            return self.entity_key.project(row)
        if self.field_id not in row:
            raise ResultProjectionError("result field is unavailable")
        return row[self.field_id]


    def project_display(self, row: Mapping[str, RuntimeValue]) -> str:
        if not self.display_field_id:
            return ''
        if self.display_field_id not in row:
            raise ResultProjectionError('entity display field is unavailable')
        value=row[self.display_field_id]
        if value is None:
            return ''
        if not isinstance(value,str):
            raise ResultProjectionError('entity display value must be text')
        return value


@dataclass(frozen=True)
class ScalarResultOutput:
    id: str
    scalar_id: str
    label: str = ""
    role: str = ""

    @property
    def source_node_id(self) -> str:
        return f"scalar:{self.scalar_id}"


@dataclass(frozen=True)
class ResultProjection:
    relation_outputs: tuple[RelationResultOutput, ...] = ()
    scalar_outputs: tuple[ScalarResultOutput, ...] = ()

    def project_rows(
        self,
        relations: Mapping[str, tuple[Mapping[str, RuntimeValue], ...]],
    ) -> tuple[ProjectedResultRow, ...]:
        outputs_by_relation: dict[str, list[RelationResultOutput]] = {}
        for output in self.relation_outputs:
            outputs_by_relation.setdefault(output.relation_id, []).append(output)
        projected: list[ProjectedResultRow] = []
        for relation_id, outputs in outputs_by_relation.items():
            rows = relations.get(relation_id)
            if rows is None:
                raise ResultProjectionError("result relation is unavailable")
            projected.extend(
                ProjectedResultRow(
                    relation_id=relation_id,
                    row_index=row_index,
                    values={output.id: output.project(row) for output in outputs},
                    display_values={output.id:output.project_display(row) for output in outputs if output.display_field_id},
                )
                for row_index, row in enumerate(rows)
            )
        return tuple(projected)
