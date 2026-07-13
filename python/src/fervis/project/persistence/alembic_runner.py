"""Package-owned Alembic runner for Fervis SQL targets."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.engine import Connection

from .revisions import (
    ALEMBIC_REVISION,
    ALEMBIC_VERSION_TABLE,
    PUBLIC_REVISIONS,
    TARGET_REVISION,
)


def current_public_revision(connection: Connection) -> str | None:
    return _public_revision(current_alembic_revision(connection))


def pending_public_revisions(connection: Connection) -> list[str]:
    current = current_alembic_revision(connection)
    revisions = tuple(PUBLIC_REVISIONS)
    if current is None:
        return [PUBLIC_REVISIONS[revision] for revision in revisions]
    try:
        current_index = revisions.index(current)
    except ValueError:
        return [TARGET_REVISION]
    return [
        PUBLIC_REVISIONS[revision] for revision in revisions[current_index + 1 :]
    ]


def upgrade_to_head(connection: Connection) -> None:
    command.upgrade(alembic_config(connection), "head")


def migrations_applied(connection: Connection) -> bool:
    return current_alembic_revision(connection) == ALEMBIC_REVISION


def current_alembic_revision(connection: Connection) -> str | None:
    context = MigrationContext.configure(
        connection,
        opts={"version_table": ALEMBIC_VERSION_TABLE},
    )
    return context.get_current_revision()


def alembic_config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", _script_location())
    config.set_main_option("version_table", ALEMBIC_VERSION_TABLE)
    config.attributes["connection"] = connection
    return config


def _public_revision(current_revision: str | None) -> str | None:
    if current_revision is None:
        return None
    return PUBLIC_REVISIONS.get(current_revision, current_revision)


def _script_location() -> str:
    return "fervis.project:persistence_migrations"
