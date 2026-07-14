from __future__ import annotations

from pathlib import Path

from fervis.project.persistence.revisions import ALEMBIC_REVISION, TARGET_REVISION


SOURCE = Path(__file__).resolve().parents[2] / "src" / "fervis"


def test_persistence_keeps_one_frozen_snapshot_per_public_revision() -> None:
    assert _snapshot_files(SOURCE / "project" / "persistence" / "schema_snapshots") == (
        "v0001.py",
        "v0002.py",
        "v0003.py",
    )


def test_public_and_alembic_revisions_name_the_current_head() -> None:
    assert TARGET_REVISION == "fervis.0003"
    assert ALEMBIC_REVISION == "0003_clarification_successor_runs"


def _snapshot_files(path: Path) -> tuple[str, ...]:
    return tuple(sorted(item.name for item in path.glob("v[0-9]*.py")))
