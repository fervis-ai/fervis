from __future__ import annotations

from pathlib import Path

from fervis.project.persistence.revisions import ALEMBIC_REVISION, TARGET_REVISION


SOURCE = Path(__file__).resolve().parents[2] / "src" / "fervis"


def test_persistence_keeps_one_frozen_snapshot_per_public_revision() -> None:
    assert _snapshot_files(SOURCE / "project" / "persistence" / "schema_snapshots") == (
        "v0001.py",
        "v0002.py",
    )


def test_public_and_alembic_revisions_name_the_current_head() -> None:
    assert TARGET_REVISION == "fervis.0002"
    assert ALEMBIC_REVISION == "0002_same_run_clarification_and_idempotency"


def _snapshot_files(path: Path) -> tuple[str, ...]:
    return tuple(sorted(item.name for item in path.glob("v[0-9]*.py")))
