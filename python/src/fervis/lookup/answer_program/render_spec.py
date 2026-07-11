"""Fact-plan render contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderRelationOutput:
    id: str
    relation_id: str
    field_id: str
    label: str = ""
    role: str = ""

    @property
    def source_node_id(self) -> str:
        return f"relation:{self.relation_id}"


@dataclass(frozen=True)
class RenderScalarOutput:
    id: str
    scalar_id: str
    label: str = ""
    role: str = ""

    @property
    def source_node_id(self) -> str:
        return f"scalar:{self.scalar_id}"


@dataclass(frozen=True)
class RenderSpec:
    relation_outputs: tuple[RenderRelationOutput, ...] = ()
    scalar_outputs: tuple[RenderScalarOutput, ...] = ()
