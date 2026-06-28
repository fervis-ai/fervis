from __future__ import annotations

import json
from pathlib import Path

from fervis.project import discover_project
from fervis.project.init import initialize_fervis_project


def test_init_writes_json_config_without_python_projection(tmp_path: Path) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
        encoding="utf-8",
    )

    result = initialize_fervis_project(
        discover_project(root),
        requested_framework="fastapi",
        yes=True,
        path_prefixes=("/",),
    )

    assert not result.is_blocked
    assert "config/fervis.json" in result.changed_files
    assert not (root / "config" / "fervis.py").exists()
    assert not (root / "config" / "__init__.py").exists()
    schema = json.loads((root / "config" / "fervis.json").read_text(encoding="utf-8"))
    assert schema["schema_version"] == "v0.1"
    assert schema["default_environment"] in schema["environments"]
    assert "cli" not in schema
    assert "execution" not in schema
    assert "model" not in schema
    assert "models" in schema
