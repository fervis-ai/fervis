from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_project_mounting_imports_without_fastapi_installed() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    script = """
import importlib.abc
import sys

class MissingFastAPI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'fastapi' or fullname.startswith('fastapi.'):
            raise ModuleNotFoundError("No module named 'fastapi'")
        return None

sys.meta_path.insert(0, MissingFastAPI())
import fervis.project.mounting
"""

    safe_path = ["-P"] if sys.version_info >= (3, 11) else []
    completed = subprocess.run(
        [sys.executable, *safe_path, "-c", script],
        cwd=source_root,
        env={"PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
