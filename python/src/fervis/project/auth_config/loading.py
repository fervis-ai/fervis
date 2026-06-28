"""Load Fervis host-auth JSON config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from fervis.project.configuration import ConfigProblem
from fervis.project.config_io import (
    ActiveEnvironment,
    ConfigIOError,
    load_auth_json_config,
)
from fervis.project.discovery import ProjectInspection


@dataclass(frozen=True)
class LoadedAuthSchema:
    schema: dict[str, object]
    config_path: Path
    active_environment: ActiveEnvironment


def load_auth_project_schema(
    project: ProjectInspection,
    *,
    active_environment: ActiveEnvironment,
) -> LoadedAuthSchema | ConfigProblem:
    try:
        loaded = load_auth_json_config(
            project.root_path,
            active_environment=active_environment,
        )
    except ConfigIOError as exc:
        return ConfigProblem(
            code="auth_config_invalid",
            message=str(exc) or exc.__class__.__name__,
        )
    return LoadedAuthSchema(
        schema=loaded.active_schema,
        config_path=loaded.config_path,
        active_environment=loaded.active_environment,
    )
