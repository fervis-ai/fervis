"""Shared SQL persistence backend behavior."""

from __future__ import annotations

import re
from typing import TypeVar

import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.type_api import TypeEngine

from fervis.interfaces.agent.actions import run_migrate_action

from .alembic_runner import (
    current_public_revision,
    migrations_applied,
    pending_public_revisions,
    upgrade_to_head,
)
from .contracts import (
    MigrationResult,
    MigrationStatus,
    PersistenceCheck,
    ResolvedPersistenceTarget,
)
from .revisions import TARGET_REVISION
from .schema import metadata


class SqlPersistenceBackend:
    target_revision = TARGET_REVISION

    def __init__(self, *, target: ResolvedPersistenceTarget) -> None:
        self.target = target

    def engine(self, *, create: bool = False) -> Engine | None:
        del create
        raise NotImplementedError

    def target_check(self) -> PersistenceCheck:
        raise NotImplementedError

    def connection_fix(self) -> dict[str, object] | None:
        return _migrate_fix()

    def target_unavailable_message(self) -> str:
        return "Persistence target is not available."

    def inspect(self) -> list[PersistenceCheck]:
        target = self.target_check()
        if not target.passed:
            return [target]
        return [
            target,
            self.connection_check(),
            self.migration_check(),
            self.tables_check(),
        ]

    def migrate(self) -> MigrationResult:
        target = self.target_check()
        if not target.passed:
            return MigrationResult(
                target=self.target,
                status=MigrationStatus.BLOCKED,
                current_revision=None,
                target_revision=TARGET_REVISION,
                error=target.message,
            )
        try:
            engine = self.engine(create=True)
            if engine is None:
                return MigrationResult(
                    target=self.target,
                    status=MigrationStatus.BLOCKED,
                    current_revision=None,
                    target_revision=TARGET_REVISION,
                    error=self.target_unavailable_message(),
                )
            with engine.connect() as connection:
                pending_revisions = pending_public_revisions(connection)
                if not pending_revisions:
                    schema_errors = _schema_differences(connection)
                    if schema_errors:
                        return MigrationResult(
                            target=self.target,
                            status=MigrationStatus.FAILED,
                            current_revision=current_public_revision(connection),
                            target_revision=TARGET_REVISION,
                            error=_schema_error_message(schema_errors),
                        )
                    return MigrationResult(
                        target=self.target,
                        status=MigrationStatus.UP_TO_DATE,
                        current_revision=current_public_revision(connection),
                        target_revision=TARGET_REVISION,
                        already_applied=True,
                    )
                _upgrade_with_constraints_restored(connection)
                schema_errors = _schema_differences(connection)
                if schema_errors:
                    return MigrationResult(
                        target=self.target,
                        status=MigrationStatus.FAILED,
                        current_revision=current_public_revision(connection),
                        target_revision=TARGET_REVISION,
                        error=_schema_error_message(schema_errors),
                    )
                return MigrationResult(
                    target=self.target,
                    status=MigrationStatus.APPLIED,
                    current_revision=TARGET_REVISION,
                    target_revision=TARGET_REVISION,
                    pending_revisions=pending_revisions,
                    applied_revisions=pending_revisions,
                    already_applied=False,
                )
        except Exception as exc:
            return MigrationResult(
                target=self.target,
                status=MigrationStatus.FAILED,
                current_revision=None,
                target_revision=TARGET_REVISION,
                error=str(exc) or exc.__class__.__name__,
            )

    def connection_check(self) -> PersistenceCheck:
        engine = self.engine()
        if engine is None:
            return PersistenceCheck(
                id="persistence.connection",
                passed=False,
                message=self.target_unavailable_message(),
                fix=self.connection_fix(),
            )
        try:
            with engine.connect():
                pass
        except Exception as exc:
            return PersistenceCheck(
                id="persistence.connection",
                passed=False,
                message=str(exc) or exc.__class__.__name__,
                fix=self.connection_fix(),
            )
        return PersistenceCheck(
            id="persistence.connection",
            passed=True,
            message=f"Connected to {self.target.kind} persistence target.",
        )

    def migration_check(self) -> PersistenceCheck:
        engine = self.engine()
        if engine is None:
            return PersistenceCheck(
                id="persistence.migrations",
                passed=False,
                message="Fervis migrations are not applied.",
                fix=_migrate_fix(),
            )
        try:
            with engine.connect() as connection:
                passed = migrations_applied(connection)
        except Exception as exc:
            return PersistenceCheck(
                id="persistence.migrations",
                passed=False,
                message=str(exc) or exc.__class__.__name__,
                fix=_migrate_fix(),
            )
        return PersistenceCheck(
            id="persistence.migrations",
            passed=passed,
            message=(
                "Fervis migrations are applied."
                if passed
                else "Fervis migrations are not applied."
            ),
            fix=None if passed else _migrate_fix(),
        )

    def tables_check(self) -> PersistenceCheck:
        engine = self.engine()
        if engine is None:
            return PersistenceCheck(
                id="persistence.tables",
                passed=False,
                message="Fervis tables are missing.",
                fix=_migrate_fix(),
            )
        try:
            actual_tables = set(inspect(engine).get_table_names())
        except Exception as exc:
            return PersistenceCheck(
                id="persistence.tables",
                passed=False,
                message=str(exc) or exc.__class__.__name__,
                fix=_migrate_fix(),
            )
        missing = [table for table in metadata.tables if table not in actual_tables]
        schema_errors = [] if missing else _schema_differences(engine)
        return PersistenceCheck(
            id="persistence.tables",
            passed=not missing and not schema_errors,
            message=(
                "Fervis persistence tables are present."
                if not missing and not schema_errors
                else f"Missing Fervis tables: {', '.join(missing)}."
                if missing
                else _schema_error_message(schema_errors)
            ),
            fix=None if not missing and not schema_errors else _migrate_fix(),
        )


def _migrate_fix() -> dict[str, object]:
    return run_migrate_action()


def _upgrade_with_constraints_restored(connection: Connection) -> None:
    connection.commit()
    sqlite = connection.dialect.name == "sqlite"
    if sqlite:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    try:
        upgrade_to_head(connection)
        connection.commit()
    finally:
        if connection.in_transaction():
            connection.rollback()
        if sqlite:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    if sqlite:
        violations = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("persistence migration violated foreign-key integrity")


def _schema_differences(engine_or_connection) -> list[str]:
    inspector = inspect(engine_or_connection)
    errors: list[str] = []
    actual_tables = set(inspector.get_table_names())
    for table_name, table in metadata.tables.items():
        if table_name not in actual_tables:
            errors.append(f"{table_name}: missing table")
            continue
        actual_column_rows = {
            column["name"]: column for column in inspector.get_columns(table_name)
        }
        actual_columns = set(actual_column_rows)
        expected_columns = set(table.columns.keys())
        missing_columns = sorted(expected_columns - actual_columns)
        extra_columns = sorted(actual_columns - expected_columns)
        if missing_columns:
            errors.append(f"{table_name}: missing columns {', '.join(missing_columns)}")
        if extra_columns:
            errors.append(f"{table_name}: extra columns {', '.join(extra_columns)}")
        if missing_columns:
            continue
        errors.extend(
            _column_differences(
                engine=engine_or_connection,
                table=table,
                actual_columns=actual_column_rows,
            )
        )
        errors.extend(_primary_key_differences(inspector, table))
        errors.extend(_foreign_key_differences(inspector, table))
        errors.extend(_unique_differences(inspector, table))
        errors.extend(_index_differences(engine_or_connection, inspector, table))
        errors.extend(_check_constraint_differences(inspector, table))
    return errors


def _column_differences(
    *,
    engine: Engine,
    table: sa.Table,
    actual_columns: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    for column in table.columns:
        actual = actual_columns[column.name]
        reflected_type = actual["type"]
        if not isinstance(reflected_type, TypeEngine):
            errors.append(f"{table.name}.{column.name}: reflected type is invalid")
            continue
        actual_type = _normalize_type(
            str(reflected_type.compile(dialect=engine.dialect))
        )
        expected_type = _normalize_type(
            str(column.type.compile(dialect=engine.dialect))
        )
        if actual_type != expected_type:
            errors.append(
                f"{table.name}.{column.name}: type {actual_type} != {expected_type}"
            )
        actual_nullable = bool(actual["nullable"])
        if actual_nullable != bool(column.nullable):
            errors.append(
                f"{table.name}.{column.name}: nullable "
                f"{actual_nullable} != {bool(column.nullable)}"
            )
    return errors


def _primary_key_differences(inspector, table: sa.Table) -> list[str]:
    actual = tuple(
        inspector.get_pk_constraint(table.name).get("constrained_columns") or ()
    )
    expected = tuple(column.name for column in table.primary_key.columns)
    if actual != expected:
        return [f"{table.name}: primary key {actual} != {expected}"]
    return []


def _foreign_key_differences(inspector, table: sa.Table) -> list[str]:
    actual = {
        (
            item.get("name") or "",
            tuple(item.get("constrained_columns") or ()),
            item.get("referred_table") or "",
            tuple(item.get("referred_columns") or ()),
        )
        for item in inspector.get_foreign_keys(table.name)
    }
    expected = {
        (
            constraint.name or "",
            tuple(column.name for column in constraint.columns),
            next(iter(constraint.elements)).column.table.name,
            tuple(element.column.name for element in constraint.elements),
        )
        for constraint in table.foreign_key_constraints
    }
    return _set_difference_errors(table.name, "foreign keys", actual, expected)


def _unique_differences(inspector, table: sa.Table) -> list[str]:
    actual = {
        tuple(item.get("column_names") or ())
        for item in inspector.get_unique_constraints(table.name)
    } | {
        tuple(item.get("column_names") or ())
        for item in inspector.get_indexes(table.name)
        if item.get("unique")
    }
    expected = (
        {(column.name,) for column in table.columns if column.unique}
        | {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, sa.UniqueConstraint)
        }
        | {
            tuple(column.name for column in index.columns)
            for index in table.indexes
            if index.unique
        }
    )
    return _set_difference_errors(table.name, "unique constraints", actual, expected)


def _index_differences(
    engine: Engine,
    inspector,
    table: sa.Table,
) -> list[str]:
    sqlite_predicates = _sqlite_index_predicates(engine, table.name)
    actual = {
        str(item["name"]): (
            tuple(item.get("column_names") or ()),
            bool(item.get("unique")),
            _normalize_predicate_sql(
                _strip_table_prefix(
                    str(
                        sqlite_predicates.get(str(item["name"]))
                        or item.get("dialect_options", {}).get("sqlite_where")
                        or item.get("sqlite_where")
                        or ""
                    ),
                    table_name=table.name,
                )
            ),
        )
        for item in inspector.get_indexes(table.name)
        if item.get("name")
    }
    expected = {
        str(index.name): (
            tuple(column.name for column in index.columns),
            bool(index.unique),
            _normalize_predicate_sql(
                _strip_table_prefix(
                    _compile_index_where(index, engine),
                    table_name=table.name,
                )
            ),
        )
        for index in table.indexes
        if index.name
    }
    errors: list[str] = []
    for name, expected_shape in sorted(expected.items()):
        actual_shape = actual.get(name)
        if actual_shape is None:
            errors.append(f"{table.name}: missing index {name}")
        elif actual_shape != expected_shape:
            errors.append(
                f"{table.name}: index {name} {actual_shape} != {expected_shape}"
            )
    extra = sorted(set(actual) - set(expected))
    if extra:
        errors.append(f"{table.name}: extra indexes {', '.join(extra)}")
    return errors


def _check_constraint_differences(inspector, table: sa.Table) -> list[str]:
    actual = {
        str(item.get("name") or ""): _normalize_sql(str(item.get("sqltext") or ""))
        for item in inspector.get_check_constraints(table.name)
        if item.get("name")
    }
    expected = {
        str(constraint.name): _normalize_sql(
            _strip_table_prefix(
                str(
                    constraint.sqltext.compile(
                        dialect=inspector.bind.dialect,
                        compile_kwargs={"literal_binds": True},
                    )
                ),
                table_name=table.name,
            )
        )
        for constraint in table.constraints
        if isinstance(constraint, sa.CheckConstraint) and constraint.name
    }
    errors: list[str] = []
    for name, expected_sql in sorted(expected.items()):
        actual_sql = actual.get(name)
        if actual_sql is None:
            errors.append(f"{table.name}: missing check constraint {name}")
        elif actual_sql != expected_sql:
            errors.append(
                f"{table.name}: check constraint {name} {actual_sql} != {expected_sql}"
            )
    extra = sorted(set(actual) - set(expected))
    if extra:
        errors.append(f"{table.name}: extra check constraints {', '.join(extra)}")
    return errors


_DifferenceTuple = TypeVar("_DifferenceTuple", bound=tuple[object, ...])


def _set_difference_errors(
    table_name: str,
    label: str,
    actual: set[_DifferenceTuple],
    expected: set[_DifferenceTuple],
) -> list[str]:
    errors: list[str] = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{table_name}: missing {label} {missing}")
    if extra:
        errors.append(f"{table_name}: extra {label} {extra}")
    return errors


def _normalize_type(value: str) -> str:
    return " ".join(value.upper().split())


def _normalize_sql(value: str) -> str:
    return " ".join(value.replace('"', "").split())


def _normalize_predicate_sql(value: str) -> str:
    normalized = _normalize_sql(value)
    if normalized.upper().startswith("WHERE "):
        normalized = normalized[6:].strip()
    while (
        normalized.startswith("(")
        and normalized.endswith(")")
        and _balanced_outer(normalized)
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def _balanced_outer(value: str) -> bool:
    depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
    return depth == 0


def _strip_table_prefix(value: str, *, table_name: str) -> str:
    return re.sub(rf"\b{re.escape(table_name)}\.", "", value)


def _sqlite_index_predicates(engine: Engine, table_name: str) -> dict[str, str]:
    if engine.dialect.name != "sqlite":
        return {}
    if hasattr(engine, "exec_driver_sql"):
        rows = engine.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (table_name,),
        ).all()
    else:
        with engine.connect() as connection:
            rows = connection.exec_driver_sql(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
                (table_name,),
            ).all()
    predicates: dict[str, str] = {}
    for name, sql in rows:
        if not sql:
            continue
        match = re.search(r"\bWHERE\b(.+)$", str(sql), flags=re.IGNORECASE)
        if match:
            predicates[str(name)] = match.group(1).strip()
    return predicates


def _compile_index_where(index: sa.Index, engine: Engine) -> str:
    where = index.dialect_options[engine.dialect.name].get("where")
    if where is None and engine.dialect.name == "sqlite":
        where = index.dialect_options["sqlite"].get("where")
    if where is None:
        return ""
    return str(
        where.compile(
            dialect=engine.dialect,
            compile_kwargs={"literal_binds": True},
        )
    )


def _schema_error_message(errors: list[str]) -> str:
    if not errors:
        return "Fervis persistence tables are present."
    return "Fervis schema does not match package schema: " + "; ".join(errors)
