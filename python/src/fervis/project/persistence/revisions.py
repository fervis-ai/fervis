"""Fervis persistence revision identifiers."""

from __future__ import annotations

TARGET_REVISION = "fervis.0004"
ALEMBIC_REVISION = "0004_semantic_requested_facts"
ALEMBIC_VERSION_TABLE = "fervis_schema_migration"
PUBLIC_REVISIONS = {
    "0001_initial": "fervis.0001",
    "0002_same_run_clarification_and_idempotency": "fervis.0002",
    "0003_clarification_successor_runs": "fervis.0003",
    ALEMBIC_REVISION: TARGET_REVISION,
}
