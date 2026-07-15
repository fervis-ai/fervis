from __future__ import annotations

from pathlib import Path


def configured_fervis(*, root: Path | str | None = None):
    """Load the integration configured for the current host project."""
    from fervis.project.configuration import (
        ConfigProblem,
        load_fervis_project_config,
    )
    from fervis.project.discovery import discover_project

    project = discover_project(Path(root) if root is not None else Path.cwd())
    loaded = load_fervis_project_config(project)
    if isinstance(loaded, ConfigProblem):
        raise RuntimeError(loaded.message)
    return loaded.integration
