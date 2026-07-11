"""Mechanical validation for relation catalogs."""

from __future__ import annotations

from dataclasses import replace

from fervis.lookup.relation_catalog.model import (
    CatalogParam,
    EndpointRead,
    PaginationMode,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValueError,
    parse_catalog_parameter_value,
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
    parsed_reads = tuple(_validate_read(read) for read in catalog.reads)
    return replace(catalog, reads=parsed_reads)


def parse_relation_catalog_values(catalog: RelationCatalog) -> RelationCatalog:
    """Parse declared endpoint values without applying unrelated catalog policy."""

    return replace(
        catalog,
        reads=tuple(_parse_read_values(read) for read in catalog.reads),
    )


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


def _validate_read(read: EndpointRead) -> EndpointRead:
    read_id = read.id
    if not read_id:
        raise CatalogValidationError("read id is required")
    if not read.endpoint_name:
        raise CatalogValidationError(f"read {read_id} endpoint name is required")
    resource_names = tuple(
        str(value).strip()
        for value in read.resource_names
        if str(value).strip()
    )
    if not resource_names:
        raise CatalogValidationError(f"read {read_id} resource names are required")
    _validate_params(read)
    _validate_row_paths(read)
    _validate_fields(read)
    _validate_read_facts(read)
    _validate_pagination(read)
    return _parse_read_values(read)


def _validate_params(read: EndpointRead) -> None:
    read_id = read.id
    _validate_unique(
        (param.ref for param in read.params),
        label=f"read {read_id} param ref",
    )
    for param in read.params:
        if not param.ref:
            raise CatalogValidationError(f"read {read_id} param ref is required")
        if not param.name:
            raise CatalogValidationError(f"read {read_id} param name is required")
        if not param.source:
            raise CatalogValidationError(f"read {read_id} param source is required")
        if not param.type:
            raise CatalogValidationError(f"read {read_id} param type is required")


def _validate_row_paths(read: EndpointRead) -> None:
    read_id = read.id
    _validate_unique(
        (row_path.id for row_path in read.row_paths),
        label=f"read {read_id} row path id",
    )
    for row_path in read.row_paths:
        if not row_path.id:
            raise CatalogValidationError(f"read {read_id} row path id is required")
        if row_path.id != "root" and not row_path.path:
            raise CatalogValidationError(f"read {read_id} row path is required")
        if not row_path.cardinality:
            raise CatalogValidationError(
                f"read {read_id} row path cardinality is required"
            )


def _validate_fields(read: EndpointRead) -> None:
    read_id = read.id
    row_path_ids = {item.id for item in read.row_paths}
    param_refs = {item.ref for item in read.params}
    fields = read.fields
    _validate_unique(
        (field.ref for field in fields),
        label=f"read {read_id} field ref",
    )
    _validate_unique(
        (field.path for field in fields),
        label=f"read {read_id} field path",
    )
    for field in fields:
        if not field.ref:
            raise CatalogValidationError(f"read {read_id} field ref is required")
        if not field.path:
            raise CatalogValidationError(f"read {read_id} field path is required")
        if not field.type:
            raise CatalogValidationError(f"read {read_id} field type is required")
        field_row_path_id = infer_field_row_path_id(
            field,
            row_paths=read.row_paths,
        )
        if row_path_ids and field_row_path_id not in row_path_ids:
            raise CatalogValidationError(
                f"read {read_id} field {field.ref} references unknown row path"
            )
        if not row_path_ids and field.row_path_id:
            raise CatalogValidationError(
                f"read {read_id} field {field.ref} references unknown row path"
            )
        for requirement in field.requirements:
            if requirement.param_ref not in param_refs:
                raise CatalogValidationError(
                    f"read {read_id} field {field.ref} requires unknown param"
                )


def _validate_read_facts(read: EndpointRead) -> None:
    read_id = read.id
    field_refs = {item.ref for item in read.fields}
    facts = read.facts
    _validate_unique(
        (fact.ref for fact in facts),
        label=f"read {read_id} fact ref",
    )
    for fact in facts:
        if not fact.ref:
            raise CatalogValidationError(f"read {read_id} fact ref is required")
        if not fact.availability:
            raise CatalogValidationError(
                f"read {read_id} fact {fact.ref} availability is required"
            )
        if fact.field_ref and fact.field_ref not in field_refs:
            raise CatalogValidationError(
                f"read {read_id} fact {fact.ref} references unknown field"
            )


def _validate_pagination(read: EndpointRead) -> None:
    read_id = read.id
    pagination = read.pagination
    if pagination is None:
        raise CatalogValidationError(f"read {read_id} pagination metadata is required")
    if not isinstance(pagination.mode, PaginationMode):
        raise CatalogValidationError(f"read {read_id} pagination mode is required")


def _parse_param_value(param: CatalogParam, *, value: object):
    try:
        return parse_catalog_parameter_value(
            value,
            type_name=param.type,
            choices=param.choices,
        )
    except CatalogParameterValueError as exc:
        raise CatalogValidationError(
            f"catalog param {param.ref} has an invalid typed value"
        ) from exc


def _parse_read_values(read: EndpointRead) -> EndpointRead:
    parsed_params = tuple(
        replace(
            param,
            default=_parse_param_value(param, value=param.default),
        )
        for param in read.params
    )
    params_by_ref = {param.ref: param for param in parsed_params}
    parsed_fields = tuple(
        replace(
            field,
            requirements=tuple(
                replace(
                    requirement,
                    value=_parse_param_value(
                        params_by_ref[requirement.param_ref],
                        value=requirement.value,
                    ),
                )
                for requirement in field.requirements
            ),
        )
        for field in read.fields
    )
    return replace(read, params=parsed_params, fields=parsed_fields)


def _validate_unique(values: object, *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        item = str(value or "")
        if not item:
            continue
        if item in seen:
            raise CatalogValidationError(f"duplicate {label}: {item}")
        seen.add(item)
