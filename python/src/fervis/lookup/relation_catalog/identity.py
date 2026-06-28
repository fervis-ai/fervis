"""Catalog identity predicates used across lookup turns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fervis.lookup.relation_catalog.model import CatalogField, IdentityMetadata


def identity_is_primary_stable(identity: IdentityMetadata | None) -> bool:
    """Return whether identity metadata names the default canonical row identity."""

    return bool(identity is not None and identity.primary_key and identity.stable)


def identity_payload_is_primary_stable(identity: Any) -> bool:
    """Return whether serialized identity metadata names a canonical row identity."""

    return bool(
        isinstance(identity, dict)
        and identity
        and identity.get("primary_key")
        and identity.get("stable", True)
    )


def catalog_field_has_primary_stable_identity(field: CatalogField) -> bool:
    """Return whether a catalog field is the default canonical row identity."""

    return identity_is_primary_stable(field.identity)


def source_field_has_primary_stable_identity(field: Any) -> bool:
    """Return whether a source-binding field carries canonical identity metadata."""

    return identity_is_primary_stable(getattr(field, "identity", None))


def primary_stable_identity_field_ids(fields: Iterable[Any]) -> tuple[str, ...]:
    """Return field IDs that are legal default canonical identity choices."""

    return tuple(
        dict.fromkeys(
            field_id
            for field in fields
            if source_field_has_primary_stable_identity(field)
            for field_id in (str(getattr(field, "field_id", "") or ""),)
            if field_id
        )
    )


def read_has_primary_stable_identity(read: Any) -> bool:
    """Return whether a read exposes at least one canonical row identity field."""

    return any(
        source_field_has_primary_stable_identity(field)
        and getattr(field.identity, "entity_ref", "")
        and getattr(field.identity, "identity_field", "")
        for field in getattr(read, "fields", ())
    )


def catalog_field_is_count_anchor(field: CatalogField) -> bool:
    """Return whether a field can identify one countable row instance."""

    return catalog_field_has_primary_stable_identity(field)
