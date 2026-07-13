"""Small SQL row helpers shared by storage adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any

import sqlalchemy as sa


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row._mapping)


def row_mappings(rows: Any) -> tuple[dict[str, Any], ...]:
    return tuple(row_mapping(row) for row in rows)


def required_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"SQL row {field} must be an integer")
    return value


def optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return required_int(value, field=field)


def optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"SQL row {field} must be text")
    return value


def json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"SQL row {field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def json_objects(value: object, *, field: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"SQL row {field} must be a JSON array")
    return tuple(
        {str(key): nested for key, nested in item.items()}
        for item in value
        if isinstance(item, Mapping)
    )


def select_one(connection, table: sa.Table, lookup: dict[str, Any]):
    return connection.execute(sa.select(table).where(_where(table, lookup))).first()


def _where(table: sa.Table, lookup: dict[str, Any]):
    expressions = [table.c[field] == value for field, value in lookup.items()]
    return sa.and_(*expressions) if expressions else sa.true()


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    return value
