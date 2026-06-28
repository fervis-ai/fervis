"""Ports for supplying relation catalog data."""

from __future__ import annotations

from typing import Any, Protocol

from fervis.lookup.relation_catalog.model import RelationCatalog


class RelationCatalogProvider(Protocol):
    def build_relation_catalog(self) -> RelationCatalog:
        raise NotImplementedError


class RelationDataAccessPort(Protocol):
    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
