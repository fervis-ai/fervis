"""Fervis project persistence inspection and migration."""

from __future__ import annotations

from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.integration import (
    DatabaseUrlPersistence,
    DjangoDatabasePersistence,
    SQLitePersistence,
)
from fervis.interfaces.agent.actions import edit_config_action

from .contracts import (
    MigrationResult,
    MigrationStatus,
    PersistenceCheck,
    ResolvedPersistenceTarget,
)
from .database_url import DatabaseUrlPersistenceBackend
from .revisions import TARGET_REVISION
from .sqlite import SQLitePersistenceBackend

__all__ = [
    "MigrationResult",
    "MigrationStatus",
    "PersistenceCheck",
    "ResolvedPersistenceTarget",
    "inspect_persistence",
    "migrate_persistence",
]


def inspect_persistence(
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> list[PersistenceCheck]:
    backend = _backend(project, loaded_config)
    return backend.inspect()


def migrate_persistence(
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
) -> MigrationResult:
    backend = _backend(project, loaded_config)
    return backend.migrate()


def _backend(
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
):
    persistence = loaded_config.config.persistence
    if isinstance(persistence, SQLitePersistence):
        return SQLitePersistenceBackend(
            project_root=project.root_path,
            config=persistence,
        )
    if isinstance(persistence, DjangoDatabasePersistence):
        return _BlockedPersistenceBackend(
            target=ResolvedPersistenceTarget(
                kind="django",
                location=persistence.database,
            ),
            message="Django database persistence is explicit but not implemented in this slice.",
        )
    if isinstance(persistence, DatabaseUrlPersistence):
        return DatabaseUrlPersistenceBackend(config=persistence)
    return _BlockedPersistenceBackend(
        target=ResolvedPersistenceTarget(kind="unknown", location="config/fervis.json"),
        message="Unknown persistence target.",
    )


class _BlockedPersistenceBackend:
    target_revision = TARGET_REVISION

    def __init__(self, *, target: ResolvedPersistenceTarget, message: str) -> None:
        self.target = target
        self.message = message

    def inspect(self) -> list[PersistenceCheck]:
        return [
            PersistenceCheck(
                id="persistence.target",
                passed=False,
                message=self.message,
                fix=edit_config_action(),
            )
        ]

    def migrate(self) -> MigrationResult:
        return MigrationResult(
            target=self.target,
            status=MigrationStatus.BLOCKED,
            current_revision=None,
            target_revision=TARGET_REVISION,
            error=self.message,
        )
