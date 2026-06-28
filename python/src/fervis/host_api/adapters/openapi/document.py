"""Small in-house OpenAPI/Swagger document normalizer.

This module owns OpenAPI mechanics: basePath, path-level parameters, direct
Swagger 2 schemas, OpenAPI 3 media schemas, and local JSON references.
"""

from __future__ import annotations

from typing import Any

from fervis.project.source_paths import (
    normalize_source_path_prefixes,
    source_path_matches,
)

from .model import OpenApiOperation, OpenApiParameter


def normalized_get_operations(
    document: dict[str, Any],
    *,
    path_prefixes: tuple[str, ...],
) -> tuple[OpenApiOperation, ...]:
    prefixes = normalize_source_path_prefixes(path_prefixes)
    paths = document.get("paths")
    if not isinstance(paths, dict):
        return ()

    operations: list[OpenApiOperation] = []
    base_path = _normalize_base_path(document.get("basePath"))
    for raw_path, path_item in paths.items():
        path_template = _normalize_public_path(f"{base_path}{str(raw_path)}")
        if not source_path_matches(path_template, prefixes):
            continue
        if not isinstance(path_item, dict):
            continue
        operation = path_item.get("get")
        if not isinstance(operation, dict):
            continue
        operations.append(
            _operation(
                document,
                raw_path=str(raw_path),
                path_template=path_template,
                path_item=path_item,
                operation=operation,
            )
        )
    return tuple(operations)


def _operation(
    document: dict[str, Any],
    *,
    raw_path: str,
    path_template: str,
    path_item: dict[str, Any],
    operation: dict[str, Any],
) -> OpenApiOperation:
    operation_id = str(operation.get("operationId") or _endpoint_name(raw_path))
    parameters = [
        *_parameters(path_item.get("parameters"), document),
        *_parameters(operation.get("parameters"), document),
    ]
    response_schema = _normalize_schema(_response_schema(operation), document)
    return OpenApiOperation(
        operation_id=operation_id,
        method="GET",
        path_template=path_template,
        summary=str(operation.get("summary") or ""),
        tags=tuple(str(tag) for tag in operation.get("tags") or ()),
        parameters=tuple(parameters),
        response_schema=response_schema,
    )


def _parameters(
    value: object,
    document: dict[str, Any],
) -> tuple[OpenApiParameter, ...]:
    if not isinstance(value, list):
        return ()
    parameters: list[OpenApiParameter] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item = _normalize_object(item, document)
        location = str(item.get("in") or "")
        if location not in {"path", "query"}:
            continue
        parameters.append(
            OpenApiParameter(
                name=str(item.get("name") or ""),
                location=location,
                schema=_normalize_schema(_parameter_schema(item), document),
                required=location == "path" or bool(item.get("required")),
                description=str(item.get("description") or ""),
            )
        )
    return tuple(parameters)


def _parameter_schema(parameter: dict[str, Any]) -> dict[str, Any]:
    schema = parameter.get("schema")
    if isinstance(schema, dict):
        return schema
    direct_type = parameter.get("type")
    if isinstance(direct_type, str):
        result: dict[str, Any] = {"type": direct_type}
        if isinstance(parameter.get("enum"), list):
            result["enum"] = list(parameter["enum"])
        return result
    return {}


def _response_schema(operation: dict[str, Any]) -> dict[str, Any]:
    responses = operation.get("responses")
    if not isinstance(responses, dict):
        return {}
    response = responses.get("200") or responses.get(200) or {}
    if not isinstance(response, dict):
        return {}
    media = _json_media(response)
    schema = media.get("schema") if isinstance(media, dict) else None
    if not isinstance(schema, dict):
        schema = response.get("schema") or {}
    return schema if isinstance(schema, dict) else {}


def _json_media(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    if not isinstance(content, dict):
        return {}
    media = content.get("application/json")
    return media if isinstance(media, dict) else {}


def _normalize_schema(
    schema: dict[str, Any],
    document: dict[str, Any],
    *,
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}

    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen:
            return {}
        resolved = _local_ref(document, ref)
        if resolved is None:
            return schema
        return _normalize_schema(resolved, document, seen=seen | {ref})

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        return _normalize_all_of(schema, all_of, document, seen=seen)

    resolved: dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            resolved[key] = _normalize_schema(value, document, seen=seen)
        elif isinstance(value, list):
            resolved[key] = [
                _normalize_schema(item, document, seen=seen)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def _normalize_object(
    value: dict[str, Any], document: dict[str, Any]
) -> dict[str, Any]:
    ref = value.get("$ref")
    if not isinstance(ref, str):
        return dict(value)
    resolved = _local_ref(document, ref)
    return (
        _normalize_object(resolved, document) if resolved is not None else dict(value)
    )


def _normalize_all_of(
    schema: dict[str, Any],
    all_of: list[object],
    document: dict[str, Any],
    *,
    seen: frozenset[str],
) -> dict[str, Any]:
    merged = {
        key: value for key, value in schema.items() if key not in {"allOf", "$ref"}
    }
    for item in all_of:
        if not isinstance(item, dict):
            continue
        merged = _merge_schema(
            merged,
            _normalize_schema(item, document, seen=seen),
        )
    return _normalize_schema(merged, document, seen=seen)


def _merge_schema(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key == "properties":
            merged[key] = {
                **_dict_value(merged.get(key)),
                **_dict_value(value),
            }
            if "type" not in merged:
                merged["type"] = "object"
        elif key == "required":
            merged[key] = [
                *list(merged.get(key) or ()),
                *[item for item in value or () if item not in merged.get(key, ())],
            ]
        elif key not in merged:
            merged[key] = value
    return merged


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _local_ref(document: dict[str, Any], ref: str) -> dict[str, Any] | None:
    prefix = "#/"
    if not ref.startswith(prefix):
        return None
    value: object = document
    for part in ref.removeprefix(prefix).split("/"):
        if not isinstance(value, dict):
            return None
        value = value.get(part.replace("~1", "/").replace("~0", "~"))
    return value if isinstance(value, dict) else None


def _normalize_public_path(path: str) -> str:
    text = str(path).strip()
    if not text.startswith("/"):
        text = f"/{text}"
    return text


def _normalize_base_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() == "/":
        return ""
    return _normalize_public_path(value).rstrip("/")


def _endpoint_name(path: str) -> str:
    parts: list[str] = []
    for part in path.strip("/").split("/"):
        if not part:
            continue
        if part.startswith("{") and part.endswith("}"):
            parts.extend(("by", part.strip("{}")))
            continue
        parts.append(part)
    return "_".join(["get", *parts]) or "get_root"
