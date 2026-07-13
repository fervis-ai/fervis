"""Declared relation-key queries used across lookup turns."""

from __future__ import annotations

from typing import Protocol

from fervis.lookup.relation_catalog.model import EndpointRead


class SourceFieldRoles(Protocol):
    @property
    def field_id(self) -> str: ...

    @property
    def roles(self) -> tuple[str, ...]: ...

def read_has_primary_stable_key(read: EndpointRead) -> bool:
    return any(key.primary and key.stable for key in read.candidate_keys)


def primary_stable_key_entity_kinds(read: EndpointRead) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            key.entity_kind for key in read.candidate_keys if key.primary and key.stable
        )
    )


def source_field_is_entity_identity(field: SourceFieldRoles) -> bool:
    return "identity" in field.roles


def entity_identity_field_ids(
    fields: tuple[SourceFieldRoles, ...],
) -> tuple[str, ...]:
    return tuple(
        field.field_id
        for field in fields
        if source_field_is_entity_identity(field)
        if field.field_id
    )
