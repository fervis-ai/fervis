"""Default local SQLite persistence backend."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.engine import Engine

from fervis.interfaces.agent.actions import chmod_action, edit_config_action
from fervis.interfaces.agent.commands import commands, render_command
from fervis.project.integration import SQLitePersistence

from .contracts import PersistenceCheck, ResolvedPersistenceTarget
from .sql_backend import SqlPersistenceBackend
from .sqlite_engine import create_sqlite_engine


class SQLitePersistenceBackend(SqlPersistenceBackend):
    def __init__(self, *, project_root: Path, config: SQLitePersistence) -> None:
        self.path = _sqlite_path(project_root, config.path)
        self.project_root = project_root
        super().__init__(
            target=ResolvedPersistenceTarget(
                kind="sqlite",
                location=_display_path(self.path, project_root),
            )
        )

    def engine(self, *, create: bool = False) -> Engine | None:
        if not self.path.exists() and create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return create_sqlite_engine(f"sqlite:///{self.path}")
        if not self.path.exists():
            return None
        return create_sqlite_engine(f"sqlite:///{self.path}")

    def target_check(self) -> PersistenceCheck:
        if self.path.exists() and not self.path.is_file():
            return PersistenceCheck(
                id="persistence.target",
                passed=False,
                message=f"SQLite path {self.target.location} is not a file.",
                fix=edit_config_action(),
            )
        parent = self.path.parent
        writable_probe_path = _nearest_existing_path(parent)
        if not os.access(writable_probe_path, os.W_OK | os.X_OK):
            return PersistenceCheck(
                id="persistence.target",
                passed=False,
                message=(
                    f"SQLite parent directory for {self.target.location} "
                    "is not writable."
                ),
                fix=chmod_action(str(writable_probe_path)),
            )
        return PersistenceCheck(
            id="persistence.target",
            passed=True,
            message=f"Using local Fervis SQLite store at {self.target.location}.",
        )

    def connection_check(self) -> PersistenceCheck:
        if not self.path.exists():
            return PersistenceCheck(
                id="persistence.connection",
                passed=True,
                message=(
                    "Local Fervis SQLite store can be created by "
                    f"{render_command(commands.migrate())}."
                ),
            )
        return super().connection_check()


def _sqlite_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current
