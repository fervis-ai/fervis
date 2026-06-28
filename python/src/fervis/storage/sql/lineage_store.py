"""SQLAlchemy Core store for the framework-neutral lineage recorder."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from fervis.lineage import records
from fervis.lineage.records import LineageRow
from fervis.project.persistence.schema import metadata

from .rows import normalize_json_value, now_utc, row_mapping, select_one
from .transaction import sql_connection, sql_transaction


class SQLLineageRecorderStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def transaction(self) -> AbstractContextManager[object]:
        return sql_transaction(self.engine)

    def get_or_insert_row(self, row: LineageRow) -> LineageRow:
        with sql_connection(self.engine) as connection:
            table = _table_for(row.key)
            existing = select_one(connection, table, row.identity)
            if existing is None:
                values = _insert_values(table, row.values)
                try:
                    connection.execute(sa.insert(table).values(**values))
                except IntegrityError as exc:
                    existing = self._get_existing_after_conflict(connection, row, exc)
                else:
                    existing = select_one(connection, table, row.identity)
            if existing is None:
                raise RuntimeError(f"failed to persist lineage row {row.key}")
            return _stored_row(row, row_mapping(existing))

    def insert_row(self, row: LineageRow) -> None:
        with sql_connection(self.engine) as connection:
            table = _table_for(row.key)
            values = _insert_values(table, row.values)
            connection.execute(sa.insert(table).values(**values))

    def find_row(
        self,
        *,
        key: str,
        lookup: dict[str, object],
        fields: tuple[str, ...],
    ) -> LineageRow | None:
        with sql_connection(self.engine) as connection:
            table = _table_for(key)
            selected_fields = tuple(dict.fromkeys((*lookup.keys(), *fields)))
            row = connection.execute(
                sa.select(*(table.c[field] for field in selected_fields)).where(
                    sa.and_(
                        *(table.c[field] == value for field, value in lookup.items())
                    )
                )
            ).first()
        if row is None:
            return None
        values = row_mapping(row)
        return LineageRow(
            key=key,
            identity={field: values[field] for field in lookup},
            values={field: values[field] for field in fields},
            conflict_lookup={},
        )

    def _get_existing_after_conflict(
        self,
        connection: Connection,
        row: LineageRow,
        original_error: IntegrityError,
    ):
        table = _table_for(row.key)
        existing = select_one(connection, table, row.identity)
        if existing is not None:
            return existing
        if row.conflict_lookup:
            existing = select_one(connection, table, row.conflict_lookup)
            if existing is not None:
                return existing
        raise original_error


def _table_for(record_key: str) -> sa.Table:
    table_name = f"fervis_{record_key}"
    try:
        return metadata.tables[table_name]
    except KeyError as exc:
        raise KeyError(f"no Fervis table for lineage record {record_key!r}") from exc


def _insert_values(table: sa.Table, values: dict[str, Any]) -> dict[str, Any]:
    now = now_utc()
    output = {
        field: _insert_value(table.c[field], value)
        for field, value in values.items()
        if field in table.c
    }
    if "created_at" in table.c and "created_at" not in output:
        output["created_at"] = now
    if "updated_at" in table.c and "updated_at" not in output:
        output["updated_at"] = now
    return output


def _insert_value(column: sa.Column, value: Any) -> Any:
    if value is None and isinstance(column.type, sa.JSON):
        return sa.null()
    return normalize_json_value(value)


def _stored_row(row: LineageRow, values: dict[str, Any]) -> LineageRow:
    record = SimpleNamespace(**values)
    return LineageRow(
        key=row.key,
        identity={field: getattr(record, field) for field in row.identity},
        values={
            field: _stored_value(row.key, field, getattr(record, field))
            for field in row.storage_fields
        },
        conflict_lookup=row.conflict_lookup,
        same_run_refs=row.same_run_refs,
    )


def _stored_value(record_key: str, field: str, value: Any) -> Any:
    spec = records.RECORD_SPECS_BY_KEY[record_key]
    for row_field in (spec.identity, *spec.fields):
        if row_field.storage_field == field and row_field.json_array:
            return value if isinstance(value, list) else []
    return value
