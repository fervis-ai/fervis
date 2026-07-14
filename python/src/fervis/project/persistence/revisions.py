"""Fervis persistence revision identifiers."""

from __future__ import annotations

TARGET_REVISION = "fervis.0003"
ALEMBIC_REVISION = "0003_clarification_successor_runs"
ALEMBIC_VERSION_TABLE = "fervis_schema_migration"
PUBLIC_REVISIONS = {
    "0001_initial": "fervis.0001",
    "0002_same_run_clarification_and_idempotency": "fervis.0002",
    ALEMBIC_REVISION: TARGET_REVISION,
}
