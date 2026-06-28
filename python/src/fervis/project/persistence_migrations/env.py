"""Package-owned Alembic environment for Fervis persistence."""

from __future__ import annotations

from alembic import context

from fervis.project.persistence.schema import metadata
from fervis.project.persistence.revisions import ALEMBIC_VERSION_TABLE


def run_migrations_online() -> None:
    connection = context.config.attributes["connection"]
    context.configure(
        connection=connection,
        target_metadata=metadata,
        version_table=ALEMBIC_VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
