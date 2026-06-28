"""Row-source field path and catalog metadata helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from fervis.lookup.relation_catalog.model import (
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    EndpointRead,
    RelationCatalog,
    RowPath,
)
from fervis.lookup.relation_catalog import catalog_field_is_count_anchor
from fervis.lookup.relation_catalog.row_paths import infer_field_row_path_id
from fervis.lookup.fact_plan.relations import FieldBindingRole


def _field_row_path_id(
    field: CatalogField,
    *,
    row_paths: tuple[RowPath, ...],
) -> str:
    return infer_field_row_path_id(field, row_paths=row_paths)


def _field_row_path(
    field: CatalogField,
    *,
    row_paths: tuple[RowPath, ...],
) -> str:
    row_path_id = _field_row_path_id(field, row_paths=row_paths)
    for row_path in row_paths:
        if row_path.id == row_path_id:
            return row_path.path
    return ""


def _ancestor_row_path_ids(
    row_path_id: str,
    *,
    row_paths: tuple[RowPath, ...],
) -> tuple[str, ...]:
    by_id = {item.id: item for item in row_paths}
    by_path = {item.path: item for item in row_paths}
    current = by_id.get(row_path_id)
    output: list[str] = []
    while current is not None and current.parent_path:
        parent = by_path.get(current.parent_path)
        if parent is None:
            break
        output.append(parent.id)
        current = parent
    return tuple(output)


def _field_ids(
    fields: tuple[CatalogField, ...],
    *,
    row_path: str,
    row_paths: tuple[RowPath, ...],
) -> dict[str, str]:
    return executable_field_ids_for_row_path(
        fields,
        row_path=row_path,
        row_paths=row_paths,
    )


def executable_field_ids_for_row_path(
    fields: tuple[CatalogField, ...],
    *,
    row_path: str,
    row_paths: tuple[RowPath, ...],
) -> dict[str, str]:
    proposed = {
        field.ref: _field_public_id(
            field,
            row_paths=row_paths,
            parent_row_path=row_path,
        )
        for field in fields
    }
    counts: dict[str, int] = {}
    output: dict[str, str] = {}
    for ref, value in proposed.items():
        count = counts.get(value, 0)
        counts[value] = count + 1
        output[ref] = value if count == 0 else f"{value}_{count + 1}"
    return output


def _field_public_id(
    field: CatalogField,
    *,
    row_paths: tuple[RowPath, ...],
    parent_row_path: str = "",
) -> str:
    return _symbol(
        _relative_field_path(
            field.path,
            _field_row_path(field, row_paths=row_paths) or parent_row_path,
        )
        or field.ref
    )


def _param_ids(params: tuple[CatalogParam, ...]) -> dict[str, str]:
    counts: dict[str, int] = {}
    output: dict[str, str] = {}
    for param in params:
        value = _symbol(param.name or param.ref)
        count = counts.get(value, 0)
        counts[value] = count + 1
        output[param.ref] = value if count == 0 else f"{value}_{count + 1}"
    return output


def _read_description(read: EndpointRead) -> str:
    metadata = read.source_metadata if isinstance(read.source_metadata, dict) else {}
    return str(metadata.get("description") or "")


def _field_label(field: CatalogField, *, row_path: str, field_id: str) -> str:
    if field.path:
        return _relative_field_path(field.path, row_path) or field.path
    return _display_label(field_id)


def _relative_field_path(field_path: str, row_path: str) -> str:
    if not row_path:
        return field_path
    prefix = f"{row_path}."
    if field_path == row_path:
        return ""
    if field_path.startswith(prefix):
        return field_path[len(prefix) :]
    return field_path


def _allowed_roles(field: CatalogField) -> tuple[FieldBindingRole, ...]:
    roles = [FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE]
    if catalog_field_is_count_anchor(field):
        roles.insert(0, FieldBindingRole.IDENTITY)
    return tuple(roles)


def _field_fact_refs(
    field: CatalogField,
    *,
    catalog: RelationCatalog,
    read_id: str,
) -> tuple[str, ...]:
    refs = [field.ref]
    for fact_read_id, fact in _catalog_facts(catalog):
        if fact.availability != CatalogFactAvailability.AVAILABLE:
            continue
        if (
            fact.field_ref == field.ref
            and (not fact_read_id or fact_read_id == read_id)
            and fact.ref not in refs
        ):
            refs.append(fact.ref)
    return tuple(refs)


def _read_catalog_facts(
    read: EndpointRead,
    *,
    catalog: RelationCatalog,
) -> tuple[Any, ...]:
    facts: list[Any] = []
    facts.extend(read.facts)
    facts.extend(
        fact for fact in catalog.facts if not fact.read_id or fact.read_id == read.id
    )
    return tuple(facts)


def _catalog_facts(catalog: RelationCatalog) -> tuple[tuple[str, Any], ...]:
    return (
        *((str(getattr(fact, "read_id", "") or ""), fact) for fact in catalog.facts),
        *(
            (str(getattr(fact, "read_id", "") or read.id), fact)
            for read in catalog.reads
            for fact in read.facts
        ),
    )


def _symbol(value: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return symbol or "value"


def _display_label(value: str) -> str:
    return str(value).replace("_", " ")


def _opaque_id(prefix: str, *parts: str) -> str:
    digest = hashlib.blake2s(
        "\x1f".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"{prefix}_{digest}"
