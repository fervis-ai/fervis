"""Resolve SQL storage engines from Fervis project configuration."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine

from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.integration import (
    DatabaseUrlPersistence,
    DjangoDatabasePersistence,
    SQLitePersistence,
)
from fervis.project.persistence.database_url import (
    DatabaseUrlPersistenceBackend,
)
from fervis.project.persistence.sql_backend import SqlPersistenceBackend
from fervis.project.persistence.sqlite import SQLitePersistenceBackend


@dataclass(frozen=True)
class SQLStorageTarget:
    engine: Engine
    kind: str
    location: str


class FervisPersistenceNotReady(RuntimeError):
    def __init__(self, *, check_id: str, message: str) -> None:
        super().__init__(f"Fervis persistence is not ready: {check_id}: {message}")
        self.check_id = check_id
        self.message = message


def resolve_sql_storage_target(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> SQLStorageTarget:
    persistence = loaded_config.config.persistence
    backend: SqlPersistenceBackend
    if isinstance(persistence, SQLitePersistence):
        backend = SQLitePersistenceBackend(
            project_root=project.root_path,
            config=persistence,
        )
    elif isinstance(persistence, DatabaseUrlPersistence):
        backend = DatabaseUrlPersistenceBackend(config=persistence)
    elif isinstance(persistence, DjangoDatabasePersistence):
        raise RuntimeError(
            "DjangoDatabasePersistence is not implemented for SQL runtime storage."
        )
    else:
        raise RuntimeError("Unknown Fervis persistence target.")

    checks = backend.inspect()
    failed = [check for check in checks if not check.passed]
    if failed:
        first = failed[0]
        raise FervisPersistenceNotReady(
            check_id=first.id,
            message=first.message,
        )
    engine = backend.engine()
    if engine is None:
        raise RuntimeError("Fervis persistence target is not available.")
    return SQLStorageTarget(
        engine=engine,
        kind=backend.target.kind,
        location=backend.target.location,
    )
