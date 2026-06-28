"""Explicit SQLAlchemy database URL persistence backend."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.exc import ArgumentError
from sqlalchemy.engine import make_url
from sqlalchemy.engine import Engine

from fervis.interfaces.agent.actions import edit_config_action, set_env_action
from fervis.interfaces.agent.commands import commands, render_command
from fervis.project.integration import DatabaseUrlPersistence

from .contracts import PersistenceCheck, ResolvedPersistenceTarget
from .sql_backend import SqlPersistenceBackend
from .sqlite_engine import create_sqlite_engine


class DatabaseUrlPersistenceBackend(SqlPersistenceBackend):
    def __init__(self, *, config: DatabaseUrlPersistence) -> None:
        self.url_env = config.url_env
        super().__init__(
            target=ResolvedPersistenceTarget(
                kind="database_url",
                location=config.url_env,
            )
        )

    def engine(self, *, create: bool = False) -> Engine | None:
        parsed = self._parsed_url()
        if parsed is None or parsed.get_backend_name() != "sqlite":
            return None
        target = _sqlite_file_target(str(parsed))
        if target is not None and not target.exists() and not create:
            return None
        return create_sqlite_engine(str(parsed))

    def target_check(self) -> PersistenceCheck:
        url = self._url()
        if not url:
            return PersistenceCheck(
                id="persistence.target",
                passed=False,
                message=f"{self.url_env} is not set.",
                fix=self.connection_fix(),
            )
        try:
            parsed = make_url(url)
        except ArgumentError as exc:
            return PersistenceCheck(
                id="persistence.target",
                passed=False,
                message=str(exc) or "Invalid database URL.",
                fix=self.connection_fix(),
            )
        if parsed.get_backend_name() != "sqlite":
            return PersistenceCheck(
                id="persistence.target",
                passed=False,
                message=(
                    "DatabaseUrlPersistence supports only sqlite URLs in this slice."
                ),
                fix=edit_config_action(),
            )
        return PersistenceCheck(
            id="persistence.target",
            passed=True,
            message=f"Using sqlite database URL from {self.url_env}.",
        )

    def connection_fix(self) -> dict[str, object]:
        return set_env_action(self.url_env)

    def target_unavailable_message(self) -> str:
        parsed = self._parsed_url()
        if parsed is None:
            return f"{self.url_env} is not set."
        if parsed.get_backend_name() != "sqlite":
            return "DatabaseUrlPersistence supports only sqlite URLs in this slice."
        return (
            "Database URL SQLite store can be created by "
            f"{render_command(commands.migrate())}."
        )

    def connection_check(self) -> PersistenceCheck:
        try:
            parsed = self._parsed_url()
            target = _sqlite_file_target(str(parsed)) if parsed is not None else None
        except ArgumentError as exc:
            return PersistenceCheck(
                id="persistence.connection",
                passed=False,
                message=str(exc) or "Invalid database URL.",
                fix=self.connection_fix(),
            )
        if target is not None and not target.exists():
            return PersistenceCheck(
                id="persistence.connection",
                passed=True,
                message=(
                    "Database URL SQLite store can be created by "
                    f"{render_command(commands.migrate())}."
                ),
            )
        return super().connection_check()

    def _url(self) -> str:
        return os.environ.get(self.url_env, "")

    def _parsed_url(self):
        url = self._url()
        return make_url(url) if url else None


def _sqlite_file_target(url: str) -> Path | None:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return None
    database = parsed.database
    if not database or database == ":memory:":
        return None
    return Path(database).expanduser()
