"""Structured-output schema normalization and validation."""

from __future__ import annotations

from typing import Any

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

from fervis.model_io.backbone.dto import ToolSpec


def strip_null_properties(value: Any, *, schema: dict[str, Any] | None = None) -> Any:
    if isinstance(value, dict):
        schema = _matching_union_object_schema(value, schema) or schema
        properties = schema.get("properties") if isinstance(schema, dict) else None
        required = (
            set(schema.get("required") or ()) if isinstance(schema, dict) else set()
        )
        output: dict[str, Any] = {}
        for key, item in value.items():
            property_schema = (
                properties.get(key) if isinstance(properties, dict) else None
            )
            is_known_property = isinstance(properties, dict) and key in properties
            if item is None:
                if not is_known_property:
                    output[key] = None
                elif key in required and _schema_accepts_null(property_schema):
                    output[key] = None
                continue
            output[key] = strip_null_properties(item, schema=property_schema)
        return output
    if isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema, dict) else None
        return [strip_null_properties(item, schema=item_schema) for item in value]
    return value


def validate_tool_arguments(
    *,
    tool_spec: ToolSpec,
    arguments: dict[str, Any],
    output: dict[str, Any],
    tool_specs: tuple[ToolSpec, ...],
) -> None:
    from .errors import RequiredToolOutputError

    if not tool_spec.strict:
        return
    try:
        validate_json_schema(instance=arguments, schema=tool_spec.input_schema)
    except JsonSchemaValidationError as exc:
        raise RequiredToolOutputError(
            f"Tool {tool_spec.name} arguments do not match schema: {exc.message}",
            output=output,
            arguments=arguments,
            tool_specs=tool_specs,
        ) from exc


def _matching_union_object_schema(
    value: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    branches = schema.get("oneOf") or schema.get("anyOf")
    if not isinstance(branches, list):
        return None
    candidates = [
        branch
        for branch in branches
        if isinstance(branch, dict) and _object_branch_matches(value, branch)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _object_branch_matches(value: dict[str, Any], schema: dict[str, Any]) -> bool:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return False
    required = set(schema.get("required") or ())
    if any(key not in value for key in required):
        return False
    for key, property_schema in properties.items():
        if key not in value or not isinstance(property_schema, dict):
            continue
        if "const" in property_schema and value[key] != property_schema["const"]:
            return False
        enum = property_schema.get("enum")
        if isinstance(enum, list) and value[key] not in enum:
            return False
    return True


def _schema_accepts_null(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        return any(_schema_accepts_null(item) for item in any_of)
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        return any(_schema_accepts_null(item) for item in one_of)
    return False
