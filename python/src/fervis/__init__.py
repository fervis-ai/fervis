from __future__ import annotations

from pathlib import Path

from fervis.project import (
    DatabaseUrlPersistence,
    DjangoAppSource,
    DjangoDatabasePersistence,
    DjangoIntegration,
    FastAPIAppSource,
    FastAPIIntegration,
    FervisConfig,
    FlaskAppSource,
    FlaskIntegration,
    HostConfig,
    ModelConfig,
    PersistenceTarget,
    ProviderConfig,
    RuntimeRoutes,
    SQLitePersistence,
    discover_project,
)
from fervis.project.configuration import ConfigProblem, load_fervis_project_config


def configured_fervis(*, root: Path | str | None = None):
    project = discover_project(Path(root) if root is not None else Path.cwd())
    loaded = load_fervis_project_config(project)
    if isinstance(loaded, ConfigProblem):
        raise RuntimeError(loaded.message)
    return loaded.integration


__all__ = [
    "DatabaseUrlPersistence",
    "DjangoAppSource",
    "DjangoDatabasePersistence",
    "DjangoIntegration",
    "FastAPIAppSource",
    "FastAPIIntegration",
    "FervisConfig",
    "FlaskAppSource",
    "FlaskIntegration",
    "HostConfig",
    "ModelConfig",
    "PersistenceTarget",
    "ProviderConfig",
    "RuntimeRoutes",
    "SQLitePersistence",
    "configured_fervis",
]
