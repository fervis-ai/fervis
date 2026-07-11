from __future__ import annotations

from pathlib import Path

from fervis.project.persistence.revisions import ALEMBIC_REVISION, TARGET_REVISION


SOURCE = Path(__file__).resolve().parents[2] / "src" / "fervis"


def test_greenfield_persistence_has_one_schema_snapshot() -> None:
    assert _snapshot_files(
        SOURCE / "project" / "persistence" / "schema_snapshots"
    ) == ("v0001.py",)


def test_public_and_alembic_revisions_name_the_fresh_baseline() -> None:
    assert TARGET_REVISION == "fervis.0001"
    assert ALEMBIC_REVISION == "0001_initial"


def _snapshot_files(path: Path) -> tuple[str, ...]:
    return tuple(sorted(item.name for item in path.glob("v[0-9]*.py")))
