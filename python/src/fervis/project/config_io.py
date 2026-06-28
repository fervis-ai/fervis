"""Read, validate, write, and resolve Fervis JSON config."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config_versions.auth import active_auth_schema, normalize_auth_schema
from .config_versions.main import active_project_schema, normalize_project_schema

PROJECT_CONFIG_PATH = Path("config") / "fervis.json"
AUTH_CONFIG_PATH = Path("config") / "fervis_auth.json"

EnvironmentSource = Literal["explicit", "FERVIS_ENV", "default_environment"]


@dataclass(frozen=True)
class ConfigIOError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class ActiveEnvironment:
    name: str
    source: EnvironmentSource


@dataclass(frozen=True)
class ResolvedProjectConfig:
    raw_schema: dict[str, object]
    active_schema: dict[str, object]
    active_environment: ActiveEnvironment
    config_path: Path
    upgraded_from: str | None
    needs_write: bool


@dataclass(frozen=True)
class ResolvedAuthConfig:
    raw_schema: dict[str, object]
    active_schema: dict[str, object]
    active_environment: ActiveEnvironment
    config_path: Path
    upgraded_from: str | None
    needs_write: bool


def load_project_json_config(
    root_path: Path,
    *,
    config_path: Path = PROJECT_CONFIG_PATH,
    explicit_env: str | None = None,
) -> ResolvedProjectConfig:
    raw = load_json_schema(root_path / config_path)
    try:
        schema, upgraded_from = normalize_project_schema(raw)
        active_environment = resolve_environment_schema(
            schema,
            explicit_env=explicit_env,
        )
        active = active_project_schema(
            schema,
            environment_name=active_environment.name,
        )
    except ValueError as exc:
        raise ConfigIOError(str(exc)) from exc
    return ResolvedProjectConfig(
        raw_schema=schema,
        active_schema=active,
        active_environment=active_environment,
        config_path=config_path,
        upgraded_from=upgraded_from,
        needs_write=upgraded_from is not None,
    )


def load_auth_json_config(
    root_path: Path,
    *,
    active_environment: ActiveEnvironment,
) -> ResolvedAuthConfig:
    config_path = AUTH_CONFIG_PATH
    raw = load_json_schema(root_path / config_path)
    try:
        schema, upgraded_from = normalize_auth_schema(raw)
        active = active_auth_schema(
            schema,
            environment_name=active_environment.name,
        )
    except ValueError as exc:
        raise ConfigIOError(str(exc)) from exc
    return ResolvedAuthConfig(
        raw_schema=schema,
        active_schema=active,
        active_environment=active_environment,
        config_path=config_path,
        upgraded_from=upgraded_from,
        needs_write=upgraded_from is not None,
    )


def load_json_schema(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigIOError(f"Fervis config was not found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise ConfigIOError(f"{path} is not valid JSON: {exc.msg}.") from exc
    except OSError as exc:
        raise ConfigIOError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ConfigIOError(f"{path} must contain a JSON object.")
    if "schema_version" not in payload:
        raise ConfigIOError(f"{path} must declare schema_version.")
    return dict(payload)


def write_json_schema(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_environment_schema(
    schema: Mapping[str, object],
    *,
    explicit_env: str | None = None,
) -> ActiveEnvironment:
    if explicit_env:
        name = explicit_env
        source: EnvironmentSource = "explicit"
    else:
        env_name = os.getenv("FERVIS_ENV")
        if env_name:
            name = env_name
            source = "FERVIS_ENV"
        else:
            default = schema.get("default_environment")
            if not isinstance(default, str) or not default.strip():
                raise ValueError("default_environment must be a non-empty string.")
            name = default
            source = "default_environment"
    environments = schema.get("environments")
    if not isinstance(environments, Mapping):
        raise ValueError("environments must be an object.")
    if name not in environments:
        raise ValueError(f"Fervis environment {name!r} is not declared.")
    return ActiveEnvironment(name=name, source=source)
