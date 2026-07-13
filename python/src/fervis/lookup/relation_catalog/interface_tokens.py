"""Stable model-facing tokens for catalog interface members."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Protocol

from fervis.lookup.relation_catalog.model import RelationCatalog


class CatalogParamLike(Protocol):
    @property
    def name(self) -> str: ...


class CatalogFieldLike(Protocol):
    @property
    def path(self) -> str: ...


class CatalogInterfaceSide(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class CatalogInterfaceKind(StrEnum):
    PARAM = "param"
    FIELD = "field"


@dataclass(frozen=True)
class CatalogInterfaceToken:
    read_id: str
    side: CatalogInterfaceSide
    kind: CatalogInterfaceKind
    path: str

    @property
    def value(self) -> str:
        path = self.path or "root"
        return f"{self.read_id}.{self.side.value}.{self.kind.value}.{path}"


def catalog_input_param_token(*, read_id: str, param: CatalogParamLike) -> str:
    return CatalogInterfaceToken(
        read_id=read_id,
        side=CatalogInterfaceSide.INPUT,
        kind=CatalogInterfaceKind.PARAM,
        path=param.name,
    ).value


def catalog_output_field_token(*, read_id: str, field: CatalogFieldLike) -> str:
    return CatalogInterfaceToken(
        read_id=read_id,
        side=CatalogInterfaceSide.OUTPUT,
        kind=CatalogInterfaceKind.FIELD,
        path=field.path,
    ).value


def catalog_output_field_refs_by_token(
    catalog: RelationCatalog,
) -> dict[str, dict[str, str]]:
    return {
        read.id: {
            catalog_output_field_token(read_id=read.id, field=field): field.ref
            for field in read.fields
        }
        for read in catalog.reads
    }
