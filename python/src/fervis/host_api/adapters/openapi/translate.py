"""Translate normalized OpenAPI operations into Fervis endpoint contracts."""

from __future__ import annotations

import re
from typing import Any

from dataclasses import dataclass

from fervis.host_api.contracts import (
    CatalogEndpointContract,
    EndpointContract,
    ParameterContract,
    ResponseFieldContract,
)

from .document import normalized_get_operations
from .model import OpenApiOperation


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
            resource_names=_resource_names(operation),
            path_params=_parameters(operation, source="path"),
            query_params=_parameters(operation, source="query"),
            response_fields=_response_fields(operation.response_schema),
            response_schema=_schema_properties(operation.response_schema),
            response_cardinality=_response_cardinality(operation.response_schema),
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
        query_schema_source="openapi",
        tags=evidence.tags,
        resource_names=evidence.resource_names or (source_name,),
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
    return tuple(
        ResponseFieldContract(
            name=name,
            path=name,
            type=_schema_type(value if isinstance(value, dict) else {}),
        )
        for name, value in _schema_properties(schema).items()
    )


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


def _resource_names(operation: OpenApiOperation) -> tuple[str, ...]:
    names = [
        *_phrases_from_tags(operation.tags),
        _operation_resource_phrase(operation.operation_id),
        _path_resource_phrase(operation.path_template),
    ]
    return tuple(dict.fromkeys(name for name in names if name))


def _phrases_from_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        phrase
        for tag in tags
        if (phrase := _words_to_phrase(_split_words(tag)))
    )


def _operation_resource_phrase(operation_id: str) -> str:
    words = _split_words(operation_id)
    while words and words[0] in {
        "list",
        "get",
        "read",
        "retrieve",
        "search",
        "find",
        "fetch",
    }:
        words = words[1:]
    return _words_to_phrase(words)


def _path_resource_phrase(path_template: str) -> str:
    segments = [
        segment
        for segment in str(path_template or "").strip("/").split("/")
        if segment and not segment.startswith("{")
    ]
    return _words_to_phrase(_split_words(segments[-1] if segments else ""))


def _split_words(value: str) -> tuple[str, ...]:
    normalized = str(value or "").replace("-", "_").replace(" ", "_")
    words: list[str] = []
    for part in normalized.split("_"):
        words.extend(
            match.group(0).lower()
            for match in re.finditer(
                r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+",
                part,
            )
        )
    return tuple(words)


def _words_to_phrase(words: tuple[str, ...]) -> str:
    values = tuple(word for word in words if word)
    if not values:
        return ""
    normalized = (*values[:-1], _singularize(values[-1]))
    return " ".join(normalized)


def _singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return f"{word[:-3]}y"
    if word.endswith("ses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word
