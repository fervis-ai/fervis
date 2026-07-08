from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
LINEAGE_MIGRATIONS = "python/src/fervis/lineage/migrations"


def test_existing_lineage_migrations_are_not_rewritten() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("migration history check requires a git checkout")

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            f"{LINEAGE_MIGRATIONS}/000*.py",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip().splitlines() == []
