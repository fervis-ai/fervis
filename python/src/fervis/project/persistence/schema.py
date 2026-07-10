"""Current Fervis-owned persistence schema.

The current head schema is imported from a frozen revision snapshot. Alembic
revisions must import their own snapshot directly so historical migrations stay
immutable when a future head schema changes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

from .schema_snapshots.v0001 import metadata

HEAD_SCHEMA_FINGERPRINT = (
    "cb35cc5585cc1f38488cbd66128a8310464cbcfa760eca36f9bd4c200fc30b76"
)
FERVIS_TABLES = tuple(metadata.tables)


def assert_head_schema_fingerprint_is_current() -> None:
    current = metadata_fingerprint()
    if current != HEAD_SCHEMA_FINGERPRINT:
        raise RuntimeError(
            "Fervis head persistence schema changed without updating its "
            "fingerprint. Add a new frozen schema snapshot and migration "
            "revision, or update the head fingerprint intentionally."
        )


def metadata_fingerprint(metadata_: sa.MetaData = metadata) -> str:
    payload = {
        "tables": [
            {
                "name": table.name,
                "columns": [_column_fingerprint(column) for column in table.columns],
                "constraints": [
                    _constraint_fingerprint(constraint)
                    for constraint in sorted(table.constraints, key=_schema_item_key)
                ],
                "indexes": [
                    _index_fingerprint(index)
                    for index in sorted(table.indexes, key=_schema_item_key)
                ],
            }
            for table in sorted(metadata_.tables.values(), key=lambda item: item.name)
        ]
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _column_fingerprint(column: sa.Column) -> dict[str, object]:
    return {
        "name": column.name,
        "type": column.type.compile(dialect=sqlite.dialect()),
        "nullable": column.nullable,
        "primary_key": column.primary_key,
        "unique": column.unique,
        "autoincrement": column.autoincrement,
        "foreign_keys": sorted(
            foreign_key.target_fullname for foreign_key in column.foreign_keys
        ),
    }


def _constraint_fingerprint(constraint: sa.Constraint) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": constraint.__class__.__name__,
        "name": constraint.name,
        "columns": [column.name for column in getattr(constraint, "columns", ())],
    }
    if isinstance(constraint, sa.ForeignKeyConstraint):
        payload["referred_table"] = next(iter(constraint.elements)).column.table.name
        payload["referred_columns"] = [
            element.column.name for element in constraint.elements
        ]
    sqltext = getattr(constraint, "sqltext", None)
    if sqltext is not None:
        payload["sqltext"] = _compile_expression(sqltext)
    return payload


def _index_fingerprint(index: sa.Index) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": index.name,
        "columns": [column.name for column in index.columns],
        "unique": index.unique,
    }
    sqlite_where = index.dialect_options["sqlite"].get("where")
    if sqlite_where is not None:
        payload["where"] = _compile_expression(sqlite_where)
    return payload


def _compile_expression(expression: Any) -> str:
    return str(
        expression.compile(
            dialect=sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _schema_item_key(item: Any) -> tuple[str, str]:
    return (item.__class__.__name__, item.name or "")
