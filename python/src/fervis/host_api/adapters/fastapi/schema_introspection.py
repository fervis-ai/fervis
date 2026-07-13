"""Typed FastAPI route and response-model introspection."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from types import GenericAlias, UnionType
from typing import Protocol, TypeAlias, Union, get_args, get_origin
from uuid import UUID

from fastapi._compat import ModelField
from fastapi.routing import APIRoute
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sqlalchemy import Column, ForeignKeyConstraint, Index, Table, UniqueConstraint

from fervis.host_api.contracts import (
    CandidateKeyAuthorityComponentContract,
    CandidateKeyAuthorityContract,
    CandidateKeyComponentContract,
    CandidateKeyContract,
    EntityReferenceComponentContract,
    EntityReferenceContract,
    EntityKeyComponentTargetContract,
    ParameterContract,
    ResponseFieldContract,
)

ResponseAnnotation: TypeAlias = type | GenericAlias | UnionType


@dataclass(frozen=True)
class FastAPIResponseInspection:
    fields: tuple[ResponseFieldContract, ...]
    candidate_keys: tuple[CandidateKeyContract, ...]
    candidate_key_authorities: tuple[CandidateKeyAuthorityContract, ...]
    entity_references: tuple[EntityReferenceContract, ...]
    cardinality: str

    @property
    def schema(self) -> dict[str, dict[str, str]]:
        return {field.path: {"type": field.type} for field in self.fields}


@dataclass(frozen=True)
class _MappedResponseModel:
    table: Table
    field_paths_by_column: dict[str, str]


@dataclass(frozen=True)
class _InspectedModelField:
    name: str
    alias: str
    annotation: ResponseAnnotation | None
    description: str


class _PydanticV1FieldInfo(Protocol):
    description: str | None


class _PydanticV1Field(Protocol):
    alias: str
    outer_type_: ResponseAnnotation
    field_info: _PydanticV1FieldInfo


def inspect_fastapi_response(route: APIRoute) -> FastAPIResponseInspection:
    response_annotation = route.response_model
    model_annotations = _response_model_annotations(response_annotation)
    if not model_annotations:
        return FastAPIResponseInspection((), (), (), (), "one")
    variant_fields: list[tuple[ResponseFieldContract, ...]] = []
    variant_mappings: list[tuple[_MappedResponseModel, ...]] = []
    for annotation in model_annotations:
        collected_fields: list[ResponseFieldContract] = []
        collected_mappings: list[_MappedResponseModel] = []
        _collect_response_model(
            annotation,
            fields=collected_fields,
            mapped_models=collected_mappings,
            prefix="",
        )
        variant_fields.append(tuple(collected_fields))
        variant_mappings.append(tuple(collected_mappings))
    fields = _common_response_fields(tuple(variant_fields))
    mapped_models = _common_mapped_models(
        tuple(variant_mappings),
        common_fields=fields,
    )
    candidate_keys = tuple(
        key for mapped in mapped_models for key in _candidate_keys(mapped)
    )
    entity_references = tuple(
        reference
        for mapped in mapped_models
        for reference in _entity_references(mapped)
    )
    candidate_key_authorities = tuple(
        dict.fromkeys(
            authority
            for mapped in mapped_models
            for authority in _referenced_candidate_key_authorities(mapped)
        )
    )
    cardinality = (
        "many"
        if all(_collection_item(annotation) is not None for annotation in _union_items(response_annotation))
        else "one"
    )
    return FastAPIResponseInspection(
        fields=fields,
        candidate_keys=candidate_keys,
        candidate_key_authorities=candidate_key_authorities,
        entity_references=entity_references,
        cardinality=cardinality,
    )


def fastapi_route_parameters(
    route: APIRoute,
    *,
    source: str,
) -> tuple[ParameterContract, ...]:
    fields = (
        route.dependant.path_params
        if source == "path"
        else route.dependant.query_params
    )
    return tuple(_parameter(field, source=source) for field in fields)


def fastapi_detail_path_parameters(
    route: APIRoute,
    *,
    response: FastAPIResponseInspection,
) -> tuple[ParameterContract, ...]:
    parameters = fastapi_route_parameters(route, source="path")
    target = _single_component_detail_target(parameters, response=response)
    if target is None:
        return parameters
    return (replace(parameters[0], entity_target=target),)


def _single_component_detail_target(
    parameters: tuple[ParameterContract, ...],
    *,
    response: FastAPIResponseInspection,
) -> EntityKeyComponentTargetContract | None:
    if response.cardinality != "one" or len(parameters) != 1:
        return None
    keys = tuple(
        key
        for key in response.candidate_keys
        if key.primary and key.stable and len(key.components) == 1
    )
    if len(keys) != 1:
        return None
    key = keys[0]
    component = key.components[0]
    field = next(
        (field for field in response.fields if field.path == component.field_path),
        None,
    )
    if field is None or field.type != parameters[0].type:
        return None
    return EntityKeyComponentTargetContract(
        entity_kind=key.entity_kind,
        key_id=key.key_id,
        component_id=component.component_id,
    )


def _parameter(field: ModelField, *, source: str) -> ParameterContract:
    annotation = _response_annotation(field.field_info.annotation)
    return ParameterContract(
        name=field.alias or field.name,
        type=_annotation_type(annotation),
        required=_field_is_required(field),
        description=str(field.field_info.description or ""),
        choices=_enum_choices(annotation),
        source=source,
    )


def _collect_response_model(
    annotation: ResponseAnnotation,
    *,
    fields: list[ResponseFieldContract],
    mapped_models: list[_MappedResponseModel],
    prefix: str,
) -> None:
    model = _model_type(annotation)
    if model is None:
        return
    model_fields = _model_fields(model)
    field_paths_by_column: dict[str, str] = {}
    for field in model_fields:
        name = field.alias
        path = f"{prefix}.{name}" if prefix else name
        field_annotation = field.annotation
        nested_model = (
            _nested_model_type(field_annotation)
            if field_annotation is not None
            else None
        )
        fields.append(
            ResponseFieldContract(
                name=name,
                path=path,
                type=_annotation_type(field_annotation),
                description=field.description,
                choices=_enum_choices(field_annotation),
            )
        )
        if nested_model is not None:
            assert field_annotation is not None
            _collect_response_model(
                field_annotation,
                fields=fields,
                mapped_models=mapped_models,
                prefix=path,
            )
        field_paths_by_column[field.name] = path
    table = _response_model_table(model)
    if table is not None:
        mapped_models.append(
            _MappedResponseModel(
                table=table,
                field_paths_by_column=field_paths_by_column,
            )
        )


def _response_model_table(model: type[BaseModel]) -> Table | None:
    direct_table = getattr(model, "__table__", None)
    if isinstance(direct_table, Table):
        return direct_table
    shared_base_tables = _shared_base_tables(model)
    if len(shared_base_tables) == 1:
        return shared_base_tables[0]
    return None


def _shared_base_tables(model: type[BaseModel]) -> tuple[Table, ...]:
    tables: list[Table] = []
    for base in model.__bases__:
        if base is BaseModel or base.__module__ != model.__module__:
            continue
        for sibling in base.__subclasses__():
            if sibling is model or sibling.__module__ != model.__module__:
                continue
            table = getattr(sibling, "__table__", None)
            if isinstance(table, Table):
                tables.append(table)
    return tuple(dict.fromkeys(tables))


def _common_response_fields(
    variants: tuple[tuple[ResponseFieldContract, ...], ...],
) -> tuple[ResponseFieldContract, ...]:
    if not variants:
        return ()
    signatures = tuple(
        {(field.path, field.type) for field in fields}
        for fields in variants[1:]
    )
    return tuple(
        field
        for field in variants[0]
        if all((field.path, field.type) in variant for variant in signatures)
    )


def _common_mapped_models(
    variants: tuple[tuple[_MappedResponseModel, ...], ...],
    *,
    common_fields: tuple[ResponseFieldContract, ...],
) -> tuple[_MappedResponseModel, ...]:
    if not variants:
        return ()
    mappings = tuple(mapped for variant in variants for mapped in variant)
    tables = tuple(dict.fromkeys(mapped.table for mapped in mappings))
    if len(tables) != 1:
        return ()
    table = tables[0]
    common_field_paths = {field.path for field in common_fields}
    field_paths_by_column: dict[str, str] = {}
    for mapped in mappings:
        for column, path in mapped.field_paths_by_column.items():
            if path in common_field_paths:
                field_paths_by_column[column] = path
    return (
        _MappedResponseModel(
            table=table,
            field_paths_by_column=field_paths_by_column,
        ),
    )


def _model_fields(model: type[BaseModel]) -> tuple[_InspectedModelField, ...]:
    pydantic_v2_fields = getattr(model, "model_fields", None)
    if isinstance(pydantic_v2_fields, dict):
        return tuple(
            _pydantic_v2_field(name, field_info)
            for name, field_info in pydantic_v2_fields.items()
            if isinstance(field_info, FieldInfo)
        )
    pydantic_v1_fields = getattr(model, "__fields__", None)
    if not isinstance(pydantic_v1_fields, dict):
        return ()
    return tuple(
        _pydantic_v1_field(name, field) for name, field in pydantic_v1_fields.items()
    )


def _pydantic_v2_field(
    name: str,
    field_info: FieldInfo,
) -> _InspectedModelField:
    return _InspectedModelField(
        name=name,
        alias=str(field_info.serialization_alias or field_info.alias or name),
        annotation=_response_annotation(field_info.annotation),
        description=str(field_info.description or ""),
    )


def _pydantic_v1_field(
    name: str,
    field: _PydanticV1Field,
) -> _InspectedModelField:
    field_info = field.field_info
    return _InspectedModelField(
        name=name,
        alias=str(field.alias or name),
        annotation=_response_annotation(field.outer_type_),
        description=str(field_info.description or ""),
    )


def _field_is_required(field: ModelField) -> bool:
    is_required = getattr(field.field_info, "is_required", None)
    if callable(is_required):
        return bool(is_required())
    return bool(getattr(field, "required", False))


def _candidate_keys(mapped: _MappedResponseModel) -> tuple[CandidateKeyContract, ...]:
    keys: list[CandidateKeyContract] = []
    for key_id, columns, primary in _table_candidate_keys(mapped.table):
        field_paths = _column_field_paths(
            columns,
            field_paths_by_column=mapped.field_paths_by_column,
        )
        if not field_paths:
            continue
        components = tuple(
            CandidateKeyComponentContract(
                component_id=column.key,
                field_path=field_path,
            )
            for column, field_path in zip(columns, field_paths, strict=True)
        )
        keys.append(
            CandidateKeyContract(
                key_id=key_id,
                entity_kind=_table_entity_kind(mapped.table),
                components=components,
                primary=primary,
            )
        )
    return tuple(keys)


def _entity_references(
    mapped: _MappedResponseModel,
) -> tuple[EntityReferenceContract, ...]:
    references: list[EntityReferenceContract] = []
    constraints = tuple(
        constraint
        for constraint in mapped.table.foreign_key_constraints
        if isinstance(constraint, ForeignKeyConstraint)
    )
    for constraint in constraints:
        local_columns = tuple(element.parent for element in constraint.elements)
        target_columns = tuple(element.column for element in constraint.elements)
        field_paths = _column_field_paths(
            local_columns,
            field_paths_by_column=mapped.field_paths_by_column,
        )
        if not field_paths or not target_columns:
            continue
        target_table = target_columns[0].table
        target_key_id = _candidate_key_id(target_table, target_columns)
        if not target_key_id:
            continue
        components = tuple(
            EntityReferenceComponentContract(
                target_component_id=target_column.key,
                local_field_path=field_path,
            )
            for target_column, field_path in zip(
                target_columns,
                field_paths,
                strict=True,
            )
        )
        references.append(
            EntityReferenceContract(
                reference_id=str(
                    constraint.name
                    or f"{'_'.join(column.key for column in local_columns)}_reference"
                ),
                target_entity_kind=_table_entity_kind(target_table),
                target_key_id=target_key_id,
                components=components,
            )
        )
    return tuple(references)


def _referenced_candidate_key_authorities(
    mapped: _MappedResponseModel,
) -> tuple[CandidateKeyAuthorityContract, ...]:
    authorities: list[CandidateKeyAuthorityContract] = []
    for constraint in mapped.table.foreign_key_constraints:
        if not isinstance(constraint, ForeignKeyConstraint):
            continue
        target_columns = tuple(element.column for element in constraint.elements)
        if not target_columns:
            continue
        target_table = target_columns[0].table
        target_key_id = _candidate_key_id(target_table, target_columns)
        if not target_key_id:
            continue
        components = tuple(
            CandidateKeyAuthorityComponentContract(
                component_id=column.key,
                type=_column_type(column),
            )
            for column in target_columns
        )
        authorities.append(
            CandidateKeyAuthorityContract(
                key_id=target_key_id,
                entity_kind=_table_entity_kind(target_table),
                components=components,
                primary=target_key_id == "primary_key",
            )
        )
    return tuple(authorities)


def _column_type(column: Column) -> str:
    try:
        python_type = column.type.python_type
    except NotImplementedError:
        return "unknown"
    return _annotation_type(_response_annotation(python_type))


def _table_candidate_keys(
    table: Table,
) -> tuple[tuple[str, tuple[Column, ...], bool], ...]:
    keys: list[tuple[str, tuple[Column, ...], bool]] = []
    primary_columns = tuple(table.primary_key.columns)
    if primary_columns:
        keys.append(("primary_key", primary_columns, True))
    keys.extend(
        (f"unique_{column.key}", (column,), False)
        for column in table.columns
        if column.unique and not column.nullable and column not in primary_columns
    )
    for constraint in table.constraints:
        if not isinstance(constraint, UniqueConstraint):
            continue
        columns = tuple(constraint.columns)
        if not _columns_form_total_key(columns):
            continue
        key_id = str(
            constraint.name or f"unique_{'_'.join(column.key for column in columns)}"
        )
        keys.append((key_id, columns, False))
    for index in table.indexes:
        if not _index_forms_total_key(index):
            continue
        columns = tuple(index.columns)
        key_id = str(
            index.name or f"unique_{'_'.join(column.key for column in columns)}"
        )
        keys.append((key_id, columns, False))
    distinct_keys: list[tuple[str, tuple[Column, ...], bool]] = []
    seen_components: set[tuple[str, ...]] = set()
    for key in keys:
        component_ids = tuple(column.key for column in key[1])
        if component_ids in seen_components:
            continue
        seen_components.add(component_ids)
        distinct_keys.append(key)
    return tuple(distinct_keys)


def _columns_form_total_key(columns: tuple[Column, ...]) -> bool:
    return bool(columns) and all(not column.nullable for column in columns)


def _index_forms_total_key(index: Index) -> bool:
    columns = tuple(index.columns)
    return (
        bool(index.unique)
        and not tuple(index.dialect_kwargs)
        and (_columns_form_total_key(columns))
    )


def _candidate_key_id(table: Table, columns: tuple[Column, ...]) -> str:
    target_names = tuple(column.key for column in columns)
    for key_id, candidate_columns, _ in _table_candidate_keys(table):
        if tuple(column.key for column in candidate_columns) == target_names:
            return key_id
    return ""


def _column_field_paths(
    columns: tuple[Column, ...],
    *,
    field_paths_by_column: dict[str, str],
) -> tuple[str, ...]:
    if any(column.key not in field_paths_by_column for column in columns):
        return ()
    return tuple(field_paths_by_column[column.key] for column in columns)


def _model_type(annotation: ResponseAnnotation) -> type[BaseModel] | None:
    item = _sequence_item(annotation)
    candidate = item or annotation
    return (
        candidate
        if isinstance(candidate, type) and issubclass(candidate, BaseModel)
        else None
    )


def _response_model_annotations(
    annotation: ResponseAnnotation | None,
) -> tuple[ResponseAnnotation, ...]:
    output: list[ResponseAnnotation] = []
    for variant in _union_items(annotation):
        item = _collection_item(variant)
        model_annotation = item or variant
        if _model_type(model_annotation) is None:
            return ()
        output.append(model_annotation)
    return tuple(dict.fromkeys(output))


def _union_items(
    annotation: ResponseAnnotation | None,
) -> tuple[ResponseAnnotation, ...]:
    if annotation is None:
        return ()
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return (annotation,)
    return tuple(
        item
        for item in get_args(annotation)
        if item is not type(None)
    )


def _collection_item(annotation: ResponseAnnotation) -> ResponseAnnotation | None:
    sequence_item = _sequence_item(annotation)
    if sequence_item is not None:
        return sequence_item
    model = _model_type(annotation)
    if model is None:
        return None
    items_field = next(
        (field for field in _model_fields(model) if field.name == "items"),
        None,
    )
    if items_field is None or items_field.annotation is None:
        return None
    return _sequence_item(items_field.annotation)


def _nested_model_type(annotation: ResponseAnnotation) -> type[BaseModel] | None:
    return _model_type(annotation)


def _sequence_item(annotation: ResponseAnnotation) -> ResponseAnnotation | None:
    origin = get_origin(annotation)
    if origin not in {list, tuple, set, frozenset, Sequence}:
        return None
    args = get_args(annotation)
    return _response_annotation(args[0]) if args else None


def _response_annotation(value: ResponseAnnotation | None) -> ResponseAnnotation | None:
    if value is None:
        return None
    origin = get_origin(value)
    if origin in {Union, UnionType}:
        values = tuple(item for item in get_args(value) if item is not type(None))
        return _response_annotation(values[0]) if len(values) == 1 else None
    return value


def _annotation_type(annotation: ResponseAnnotation | None) -> str:
    if annotation is None:
        return "string"
    if _sequence_item(annotation) is not None:
        return "array"
    if _model_type(annotation) is not None or get_origin(annotation) is dict:
        return "object"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "float"
    if annotation is Decimal:
        return "decimal"
    if annotation is date:
        return "date"
    if annotation is datetime:
        return "datetime"
    if annotation is time:
        return "time"
    if annotation is UUID:
        return "uuid"
    return "string"


def _enum_choices(annotation: ResponseAnnotation | None) -> tuple[str, ...]:
    if not isinstance(annotation, type) or not issubclass(annotation, Enum):
        return ()
    return tuple(str(member.value) for member in annotation)


def _table_entity_kind(table: Table) -> str:
    words = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+",
        table.name.replace("-", "_").replace("_", " "),
    )
    return "_".join(word.lower() for word in words if word)
