"""Small SQL row helpers shared by storage adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import sqlalchemy as sa


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row._mapping)


def row_object(row: Any) -> SimpleNamespace:
    return SimpleNamespace(**row_mapping(row))


def row_mappings(rows: Any) -> tuple[dict[str, Any], ...]:
    return tuple(row_mapping(row) for row in rows)


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
