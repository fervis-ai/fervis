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


@dataclass(frozen=True)
class RenderScalarOutput:
    id: str
    scalar_id: str
    label: str = ""
    role: str = ""


@dataclass(frozen=True)
class RenderSpec:
    relation_outputs: tuple[RenderRelationOutput, ...] = ()
    scalar_outputs: tuple[RenderScalarOutput, ...] = ()
