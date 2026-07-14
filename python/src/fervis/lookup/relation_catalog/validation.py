"""Mechanical validation for relation catalogs."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Iterable

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


_UNRESOLVED_CATALOG_TYPES = frozenset({"any", "pk", "unknown"})


class CatalogValidationError(ValueError):
    pass


def parse_relation_catalog(catalog: RelationCatalog) -> RelationCatalog:
    _validate_unique(
        (read.id for read in catalog.reads),
        label="read id",
    )
    _validate_catalog_facts(catalog)
    parsed_reads = tuple(_validate_read(read) for read in catalog.reads)
    parsed_catalog = replace(catalog, reads=parsed_reads)
    _validate_candidate_key_authorities(parsed_catalog)
    _validate_entity_references(parsed_catalog)
    _validate_parameter_targets(parsed_catalog)
    return parsed_catalog


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
        str(value).strip() for value in read.resource_names if str(value).strip()
    )
    if not resource_names:
        raise CatalogValidationError(f"read {read_id} resource names are required")
    _validate_params(read)
    _validate_row_paths(read)
    _validate_fields(read)
    _validate_candidate_keys(read)
    _validate_local_entity_references(read)
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


def _validate_candidate_keys(read: EndpointRead) -> None:
    _validate_unique(
        ((key.entity_kind, key.id) for key in read.candidate_keys),
        label=f"read {read.id} candidate key identity",
    )
    fields_by_ref = {field.ref: field for field in read.fields}
    for key in read.candidate_keys:
        if not key.id:
            raise CatalogValidationError(f"read {read.id} candidate key id is required")
        if not key.entity_kind:
            raise CatalogValidationError(
                f"read {read.id} candidate key entity kind is required"
            )
        if not key.components:
            raise CatalogValidationError(
                f"read {read.id} candidate key fields are required"
            )
        _validate_unique(
            (component.id for component in key.components),
            label=f"read {read.id} candidate key {key.id} component id",
        )
        field_refs = tuple(component.field_ref for component in key.components)
        if any(
            not component.id or not component.field_ref for component in key.components
        ):
            raise CatalogValidationError(
                f"read {read.id} candidate key components are incomplete"
            )
        unknown_refs = set((*field_refs, *key.context_field_refs)) - set(fields_by_ref)
        if unknown_refs:
            raise CatalogValidationError(
                f"read {read.id} candidate key references unknown field"
            )
        row_path_ids = {
            infer_field_row_path_id(
                fields_by_ref[field_ref],
                row_paths=read.row_paths,
            )
            for field_ref in field_refs
        }
        if len(row_path_ids) != 1:
            raise CatalogValidationError(
                f"read {read.id} candidate key crosses relation grains"
            )
        key_row_path_id = next(iter(row_path_ids))
        if any(
            infer_field_row_path_id(
                fields_by_ref[field_ref],
                row_paths=read.row_paths,
            )
            != key_row_path_id
            for field_ref in key.context_field_refs
        ):
            raise CatalogValidationError(
                f"read {read.id} candidate key context crosses relation grains"
            )


def _validate_local_entity_references(read: EndpointRead) -> None:
    _validate_unique(
        (reference.id for reference in read.entity_references),
        label=f"read {read.id} entity reference id",
    )
    fields_by_ref = {field.ref: field for field in read.fields}
    for reference in read.entity_references:
        if (
            not reference.id
            or not reference.target_entity_kind
            or not reference.target_key_id
        ):
            raise CatalogValidationError(
                f"read {read.id} entity reference identity is required"
            )
        if not reference.components:
            raise CatalogValidationError(
                f"read {read.id} entity reference components are required"
            )
        _validate_unique(
            (component.target_component_id for component in reference.components),
            label=f"read {read.id} entity reference {reference.id} target component",
        )
        local_refs = tuple(
            component.local_field_ref for component in reference.components
        )
        if any(
            not component.target_component_id or not component.local_field_ref
            for component in reference.components
        ):
            raise CatalogValidationError(
                f"read {read.id} entity reference components are incomplete"
            )
        unknown_refs = set((*local_refs, *reference.context_field_refs)) - set(
            fields_by_ref
        )
        if unknown_refs:
            raise CatalogValidationError(
                f"read {read.id} entity reference references unknown field"
            )
        row_path_ids = {
            infer_field_row_path_id(
                fields_by_ref[field_ref],
                row_paths=read.row_paths,
            )
            for field_ref in local_refs
        }
        if len(row_path_ids) != 1:
            raise CatalogValidationError(
                f"read {read.id} entity reference crosses relation grains"
            )
        reference_row_path_id = next(iter(row_path_ids))
        if any(
            infer_field_row_path_id(
                fields_by_ref[field_ref],
                row_paths=read.row_paths,
            )
            != reference_row_path_id
            for field_ref in reference.context_field_refs
        ):
            raise CatalogValidationError(
                f"read {read.id} entity reference context crosses relation grains"
            )


def _validate_entity_references(catalog: RelationCatalog) -> None:
    keys = _candidate_key_contracts(catalog)
    for read in catalog.reads:
        fields_by_ref = {field.ref: field for field in read.fields}
        for reference in read.entity_references:
            target = keys.get((reference.target_entity_kind, reference.target_key_id))
            if target is None:
                raise CatalogValidationError(
                    f"read {read.id} entity reference targets unknown candidate key"
                )
            target_components = target[0]
            mapped_component_ids = tuple(
                component.target_component_id for component in reference.components
            )
            if set(mapped_component_ids) != set(target_components):
                raise CatalogValidationError(
                    f"read {read.id} entity reference does not map the complete target key"
                )
            for component in reference.components:
                local_type = fields_by_ref[component.local_field_ref].type
                target_type = target_components[component.target_component_id]
                if not _catalog_types_compatible(local_type, target_type):
                    raise CatalogValidationError(
                        f"read {read.id} entity reference component type mismatch"
                    )


def _validate_parameter_targets(catalog: RelationCatalog) -> None:
    keys = _candidate_key_contracts(catalog)
    for read in catalog.reads:
        for param in read.params:
            target = param.entity_target
            if target is None:
                continue
            key = keys.get((target.entity_kind, target.key_id))
            if key is None:
                raise CatalogValidationError(
                    f"read {read.id} param {param.ref} targets unknown candidate key"
                )
            target_type = key[0].get(target.component_id)
            if target_type is None:
                raise CatalogValidationError(
                    f"read {read.id} param {param.ref} targets unknown key component"
                )
            if not _catalog_types_compatible(param.type, target_type):
                raise CatalogValidationError(
                    f"read {read.id} param {param.ref} target type mismatch"
                )


def _catalog_types_compatible(left: str, right: str) -> bool:
    return (
        left == right
        or left in _UNRESOLVED_CATALOG_TYPES
        or right in _UNRESOLVED_CATALOG_TYPES
    )


def _candidate_key_contracts(
    catalog: RelationCatalog,
) -> dict[tuple[str, str], tuple[dict[str, str], bool, bool]]:
    contracts: dict[tuple[str, str], tuple[dict[str, str], bool, bool]] = {}
    for authority in catalog.candidate_key_authorities:
        identity = (authority.entity_kind, authority.id)
        contract = (
            {component.id: component.type for component in authority.components},
            authority.primary,
            authority.stable,
        )
        existing = contracts.get(identity)
        contracts[identity] = (
            contract
            if existing is None
            else _merge_candidate_key_contracts(existing, contract)
        )
    for read in catalog.reads:
        fields_by_ref = {field.ref: field for field in read.fields}
        for key in read.candidate_keys:
            identity = (key.entity_kind, key.id)
            contract = (
                {
                    component.id: fields_by_ref[component.field_ref].type
                    for component in key.components
                },
                key.primary,
                key.stable,
            )
            existing = contracts.get(identity)
            contracts[identity] = (
                contract
                if existing is None
                else _merge_candidate_key_contracts(existing, contract)
            )
    return contracts


def _validate_candidate_key_authorities(catalog: RelationCatalog) -> None:
    for authority in catalog.candidate_key_authorities:
        if not authority.id or not authority.entity_kind or not authority.components:
            raise CatalogValidationError("candidate key authority is incomplete")
        component_ids = tuple(component.id for component in authority.components)
        if any(not component.id or not component.type for component in authority.components):
            raise CatalogValidationError(
                "candidate key authority component is incomplete"
            )
        if len(set(component_ids)) != len(component_ids):
            raise CatalogValidationError(
                "candidate key authority component ids must be unique"
            )
    _candidate_key_contracts(catalog)


def _merge_candidate_key_contracts(
    left: tuple[dict[str, str], bool, bool],
    right: tuple[dict[str, str], bool, bool],
) -> tuple[dict[str, str], bool, bool]:
    left_components, left_primary, left_stable = left
    right_components, right_primary, right_stable = right
    if (
        set(left_components) != set(right_components)
        or left_primary != right_primary
        or left_stable != right_stable
    ):
        raise CatalogValidationError(
            "candidate key authority is inconsistent across reads"
        )
    component_types = {
        component_id: _resolved_catalog_type(
            left_components[component_id],
            right_components[component_id],
        )
        for component_id in left_components
    }
    return component_types, left_primary, left_stable


def _resolved_catalog_type(left: str, right: str) -> str:
    if not _catalog_types_compatible(left, right):
        raise CatalogValidationError(
            "candidate key authority is inconsistent across reads"
        )
    if (
        left in _UNRESOLVED_CATALOG_TYPES
        and right not in _UNRESOLVED_CATALOG_TYPES
    ):
        return right
    return left


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


def _validate_unique(values: Iterable[object], *, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        item = str(value or "")
        if not item:
            continue
        if item in seen:
            raise CatalogValidationError(f"duplicate {label}: {item}")
        seen.add(item)
