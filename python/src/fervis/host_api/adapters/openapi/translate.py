"""Translate normalized OpenAPI operations into Fervis endpoint contracts."""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

from fervis.host_api.contracts import (
    CatalogEndpointContract,
    CandidateKeyContract,
    EndpointContract,
    EntityReferenceContract,
    PaginationContract,
    ParameterContract,
    ResponseFieldContract,
)

from .document import normalized_get_operations
from .model import OpenApiOperation
from ..resource_names import endpoint_resource_names


@dataclass(frozen=True)
class OpenApiEndpointEvidence:
    operation_id: str
    method: str
    path_template: str
    summary: str
    tags: tuple[str, ...]
    resource_names: tuple[str, ...]
    path_params: tuple[ParameterContract, ...]
    query_params: tuple[ParameterContract, ...]
    response_fields: tuple[ResponseFieldContract, ...]
    response_schema: dict[str, Any]
    response_cardinality: str
    pagination: PaginationContract | None
    candidate_keys: tuple[CandidateKeyContract, ...]
    entity_references: tuple[EntityReferenceContract, ...]


def endpoint_contracts_from_openapi(
    schema: dict[str, Any],
    *,
    source_name: str,
    import_path: str,
    path_prefixes: tuple[str, ...],
    framework_kind: str,
    source_namespace_kind: str,
    source_namespace_path: tuple[str, ...],
) -> tuple[EndpointContract, ...]:
    return tuple(
        _endpoint_contract(
            evidence,
            source_name=source_name,
            import_path=import_path,
            framework_kind=framework_kind,
            source_namespace_kind=source_namespace_kind,
            source_namespace_path=source_namespace_path,
        )
        for evidence in endpoint_evidence_from_openapi(
            schema,
            path_prefixes=path_prefixes,
        )
    )


def endpoint_evidence_from_openapi(
    schema: dict[str, Any],
    *,
    path_prefixes: tuple[str, ...],
) -> tuple[OpenApiEndpointEvidence, ...]:
    return tuple(
        OpenApiEndpointEvidence(
            operation_id=operation.operation_id,
            method=operation.method,
            path_template=operation.path_template,
            summary=operation.summary,
            tags=operation.tags,
            resource_names=endpoint_resource_names(
                tags=operation.tags,
                operation_id=operation.operation_id,
                path_template=operation.path_template,
            ),
            path_params=_parameters(operation, source="path"),
            query_params=_parameters(operation, source="query"),
            response_fields=_response_fields(operation.response_schema),
            response_schema=_schema_properties(operation.response_schema),
            response_cardinality=_response_cardinality(operation.response_schema),
            pagination=operation.pagination,
            candidate_keys=operation.candidate_keys,
            entity_references=operation.entity_references,
        )
        for operation in normalized_get_operations(schema, path_prefixes=path_prefixes)
    )


def _endpoint_contract(
    evidence: OpenApiEndpointEvidence,
    *,
    source_name: str,
    import_path: str,
    framework_kind: str,
    source_namespace_kind: str,
    source_namespace_path: tuple[str, ...],
) -> EndpointContract:
    return EndpointContract(
        endpoint_name=evidence.operation_id,
        url_name=evidence.operation_id,
        method=evidence.method,
        path_template=evidence.path_template,
        docstring=evidence.summary,
        view_class=import_path,
        path_params=evidence.path_params,
        query_params=evidence.query_params,
        response_fields=evidence.response_fields,
        response_schema=evidence.response_schema,
        response_schema_source="openapi",
        response_cardinality=evidence.response_cardinality,
        pagination=evidence.pagination,
        query_schema_source="openapi",
        tags=evidence.tags,
        resource_names=evidence.resource_names or (source_name,),
        candidate_keys=evidence.candidate_keys,
        entity_references=evidence.entity_references,
        catalog_endpoint=CatalogEndpointContract(
            framework_kind=framework_kind,
            source_namespace_kind=source_namespace_kind,
            source_namespace_path=source_namespace_path,
            handler_ref=import_path,
            api_schema_operation_id=evidence.operation_id,
            domain_resource_names=evidence.resource_names,
        ),
    )


def _parameters(
    operation: OpenApiOperation,
    *,
    source: str,
) -> tuple[ParameterContract, ...]:
    return tuple(
        ParameterContract(
            name=parameter.name,
            type=_schema_type(parameter.schema),
            required=parameter.required,
            description=parameter.description,
            choices=tuple(str(value) for value in parameter.schema.get("enum") or ()),
            source=source,
        )
        for parameter in operation.parameters
        if parameter.location == source
    )

def _response_fields(schema: dict[str, Any]) -> tuple[ResponseFieldContract, ...]:
    fields: list[ResponseFieldContract] = []
    _collect_response_fields(_row_schema(schema), fields=fields, prefix="")
    return tuple(fields)


def _collect_response_fields(
    schema: dict[str, Any],
    *,
    fields: list[ResponseFieldContract],
    prefix: str,
) -> None:
    for name, raw_field_schema in _object_properties(schema).items():
        field_schema = raw_field_schema if isinstance(raw_field_schema, dict) else {}
        path = f"{prefix}.{name}" if prefix else name
        field_type = _schema_type(field_schema)
        fields.append(
            ResponseFieldContract(
                name=name,
                path=path,
                type=field_type,
                description=str(field_schema.get("description") or ""),
                choices=tuple(str(value) for value in field_schema.get("enum") or ()),
            )
        )
        child_schema = _nested_object_schema(field_schema)
        if child_schema:
            _collect_response_fields(child_schema, fields=fields, prefix=path)


def _row_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") != "array":
        return schema
    items = schema.get("items")
    return items if isinstance(items, dict) else {}


def _nested_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "array":
        items = schema.get("items")
        return items if isinstance(items, dict) else {}
    return schema if _object_properties(schema) else {}


def _object_properties(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _response_cardinality(schema: dict[str, Any]) -> str:
    return "many" if schema.get("type") == "array" else "one"


def _schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    current = schema
    if current.get("type") == "array" and isinstance(current.get("items"), dict):
        current = current["items"]
    properties = current.get("properties") if isinstance(current, dict) else {}
    return properties if isinstance(properties, dict) else {}


def _schema_type(schema: dict[str, Any]) -> str:
    value = str(schema.get("type") or "string")
    if value == "number":
        return "decimal"
    return value
