"""Shared provider-schema helpers for operation-family contracts."""

from __future__ import annotations


def non_empty_array_items(item_schema: dict[str, object]) -> dict[str, object]:
    return {"type": "array", "items": item_schema, "minItems": 1}


def strict_object(
    properties: dict[str, object],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


def handle_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


def field_id_schema(
    allowed_field_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if allowed_field_ids is not None:
        return {"enum": list(allowed_field_ids)}
    return {"type": "string", "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$"}


def non_empty_string_array() -> dict[str, object]:
    return {"type": "array", "items": handle_schema(), "minItems": 1}


def non_empty_field_id_array(field_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "array",
        "items": field_id_schema(field_ids),
        "minItems": 1,
    }
