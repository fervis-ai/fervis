"""Provider-facing schema projection helpers."""

from __future__ import annotations

from typing import Any


def strip_schema_keywords(
    schema: Any,
    *,
    forbidden_keywords: frozenset[str],
) -> Any:
    if isinstance(schema, dict):
        return {
            key: strip_schema_keywords(value, forbidden_keywords=forbidden_keywords)
            for key, value in schema.items()
            if key not in forbidden_keywords
        }
    if isinstance(schema, list):
        return [
            strip_schema_keywords(item, forbidden_keywords=forbidden_keywords)
            for item in schema
        ]
    return schema
