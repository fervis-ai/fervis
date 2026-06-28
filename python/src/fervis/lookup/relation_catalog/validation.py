"""Mechanical validation for relation catalogs."""

from __future__ import annotations

from fervis.lookup.relation_catalog.model import (
    PaginationMode,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.row_paths import infer_field_row_path_id


class CatalogValidationError(ValueError):
    pass


def validate_relation_catalog(catalog: RelationCatalog) -> RelationCatalog:
    _validate_unique(
        (read.id for read in catalog.reads),
        label="read id",
    )
    _validate_catalog_facts(catalog)
    for read in catalog.reads:
        _validate_read(read)
    return catalog


def _validate_catalog_facts(catalog: RelationCatalog) -> None:
    _validate_unique(
        (getattr(fact, "ref", "") for fact in catalog.facts),
        label="catalog fact ref",
    )
    for fact in catalog.facts:
        if not str(getattr(fact, "ref", "") or ""):
            raise CatalogValidationError("catalog fact ref is required")
        if not str(getattr(fact, "availability", "") or ""):
            raise CatalogValidationError(
                f"catalog fact {fact.ref} availability is required"
            )


def _validate_read(read: object) -> None:
    read_id = str(getattr(read, "id", "") or "")
    if not read_id:
        raise CatalogValidationError("read id is required")
    if not str(getattr(read, "endpoint_name", "") or ""):
        raise CatalogValidationError(f"read {read_id} endpoint name is required")
    resource_names = tuple(
        str(value).strip()
        for value in getattr(read, "resource_names", ())
        if str(value).strip()
    )
    if not resource_names:
        raise CatalogValidationError(f"read {read_id} resource names are required")
    _validate_params(read)
    _validate_row_paths(read)
    _validate_fields(read)
    _validate_read_facts(read)
    _validate_pagination(read)


def _validate_params(read: object) -> None:
    read_id = str(getattr(read, "id", "") or "")
    _validate_unique(
        (getattr(param, "ref", "") for param in getattr(read, "params", ())),
        label=f"read {read_id} param ref",
    )
    for param in getattr(read, "params", ()):
        if not str(getattr(param, "ref", "") or ""):
            raise CatalogValidationError(f"read {read_id} param ref is required")
        if not str(getattr(param, "name", "") or ""):
            raise CatalogValidationError(f"read {read_id} param name is required")
        if not str(getattr(param, "source", "") or ""):
            raise CatalogValidationError(f"read {read_id} param source is required")
        if not str(getattr(param, "type", "") or ""):
            raise CatalogValidationError(f"read {read_id} param type is required")


def _validate_row_paths(read: object) -> None:
    read_id = str(getattr(read, "id", "") or "")
    _validate_unique(
        (getattr(row_path, "id", "") for row_path in getattr(read, "row_paths", ())),
        label=f"read {read_id} row path id",
    )
    for row_path in getattr(read, "row_paths", ()):
        if not str(getattr(row_path, "id", "") or ""):
            raise CatalogValidationError(f"read {read_id} row path id is required")
        if row_path.id != "root" and not str(getattr(row_path, "path", "") or ""):
            raise CatalogValidationError(f"read {read_id} row path is required")
        if not str(getattr(row_path, "cardinality", "") or ""):
            raise CatalogValidationError(
                f"read {read_id} row path cardinality is required"
            )


def _validate_fields(read: object) -> None:
    read_id = str(getattr(read, "id", "") or "")
    row_path_ids = {item.id for item in getattr(read, "row_paths", ())}
    param_refs = {item.ref for item in getattr(read, "params", ())}
    fields = tuple(getattr(read, "fields", ()))
    _validate_unique(
        (getattr(field, "ref", "") for field in fields),
        label=f"read {read_id} field ref",
    )
    _validate_unique(
        (getattr(field, "path", "") for field in fields),
        label=f"read {read_id} field path",
    )
    for field in fields:
        if not str(getattr(field, "ref", "") or ""):
            raise CatalogValidationError(f"read {read_id} field ref is required")
        if not str(getattr(field, "path", "") or ""):
            raise CatalogValidationError(f"read {read_id} field path is required")
        if not str(getattr(field, "type", "") or ""):
            raise CatalogValidationError(f"read {read_id} field type is required")
        field_row_path_id = infer_field_row_path_id(
            field,
            row_paths=tuple(getattr(read, "row_paths", ())),
        )
        if row_path_ids and field_row_path_id not in row_path_ids:
            raise CatalogValidationError(
                f"read {read_id} field {field.ref} references unknown row path"
            )
        if not row_path_ids and field.row_path_id:
            raise CatalogValidationError(
                f"read {read_id} field {field.ref} references unknown row path"
            )
        for requirement in getattr(field, "requirements", ()):
            if requirement.param_ref not in param_refs:
                raise CatalogValidationError(
                    f"read {read_id} field {field.ref} requires unknown param"
                )


def _validate_read_facts(read: object) -> None:
    read_id = str(getattr(read, "id", "") or "")
    field_refs = {item.ref for item in getattr(read, "fields", ())}
    facts = tuple(getattr(read, "facts", ()))
    _validate_unique(
        (getattr(fact, "ref", "") for fact in facts),
        label=f"read {read_id} fact ref",
    )
    for fact in facts:
        if not str(getattr(fact, "ref", "") or ""):
            raise CatalogValidationError(f"read {read_id} fact ref is required")
        if not str(getattr(fact, "availability", "") or ""):
            raise CatalogValidationError(
                f"read {read_id} fact {fact.ref} availability is required"
            )
        if getattr(fact, "field_ref", "") and fact.field_ref not in field_refs:
            raise CatalogValidationError(
                f"read {read_id} fact {fact.ref} references unknown field"
            )


def _validate_pagination(read: object) -> None:
    read_id = str(getattr(read, "id", "") or "")
    pagination = getattr(read, "pagination", None)
    if pagination is None:
        raise CatalogValidationError(f"read {read_id} pagination metadata is required")
    if not isinstance(pagination.mode, PaginationMode):
        raise CatalogValidationError(f"read {read_id} pagination mode is required")


def _validate_unique(values: object, *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        item = str(value or "")
        if not item:
            continue
        if item in seen:
            raise CatalogValidationError(f"duplicate {label}: {item}")
        seen.add(item)
