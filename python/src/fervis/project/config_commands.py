"""Public config/source command services."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fervis.interfaces.agent.commands import (
    Placeholder,
    commands,
    render_command,
)

from .config_io import (
    AUTH_CONFIG_PATH,
    ConfigIOError,
    PROJECT_CONFIG_PATH,
    ResolvedProjectConfig,
    load_json_schema,
    load_project_json_config,
    write_json_schema,
)
from .config_versions.auth import normalize_auth_schema
from .config_versions.main import normalize_project_schema
from .config_schema import (
    add_source_schema,
    set_schema_value,
)
from .discovery import ProjectInspection
from .edit_result import BlockedEdit, ProjectEditResult, blocked_edit


@dataclass(frozen=True)
class ConfigCommandResult:
    payload: dict[str, object]
    blocked_edits: list[BlockedEdit] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_edits)


@dataclass
class ConfigUpgradeResult:
    changed_files: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    blocked_edits: list[BlockedEdit] = field(default_factory=list)
    upgraded_configs: list[dict[str, str]] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_edits)

    def merge(self, other: ConfigUpgradeResult) -> None:
        self.changed_files.extend(other.changed_files)
        self.skipped_existing.extend(other.skipped_existing)
        self.blocked_edits.extend(other.blocked_edits)
        self.upgraded_configs.extend(other.upgraded_configs)

    def to_payload(self) -> dict[str, object]:
        return {
            "changed_files": self.changed_files,
            "skipped_existing": self.skipped_existing,
            "blocked_edits": [
                {"file": edit.file, "reason": edit.reason}
                for edit in self.blocked_edits
            ],
            "upgraded_configs": self.upgraded_configs,
        }


def config_show(
    project: ProjectInspection,
    *,
    explicit_env: str | None = None,
) -> ConfigCommandResult:
    loaded = _load_or_problem(project, explicit_env=explicit_env)
    if isinstance(loaded, ConfigCommandResult):
        return loaded
    return ConfigCommandResult(
        payload={
            "config_path": str(loaded.config_path),
            "active_environment": {
                "name": loaded.active_environment.name,
                "source": loaded.active_environment.source,
            },
            **loaded.active_schema,
        }
    )


def config_get(
    project: ProjectInspection,
    path: str,
    *,
    explicit_env: str | None = None,
) -> ConfigCommandResult:
    try:
        if path in _CONFIG_SCALAR_PATHS:
            result = _load_raw_schema(project)
            if isinstance(result, ProjectEditResult):
                return _config_read_blocked(result)
            value = _nested_value(result, path)
        else:
            loaded = _load_or_edit_problem(project, explicit_env=explicit_env)
            if isinstance(loaded, ProjectEditResult):
                return _config_read_blocked(loaded)
            value = _schema_value(loaded, path)
    except KeyError:
        return _blocked_config_result(f"Unsupported config path {path!r}.")
    return ConfigCommandResult(payload={"path": path, "value": value})


def config_set(
    project: ProjectInspection,
    path: str,
    value: str,
    *,
    explicit_env: str | None = None,
) -> ProjectEditResult:
    if _is_model_policy_path(path):
        allow_command = render_command(commands.model_allow(Placeholder("model-ref")))
        use_command = render_command(commands.model_use(Placeholder("model-ref")))
        return _blocked_edit(
            f"Model policy must be changed with `{allow_command}` or `{use_command}`."
        )
    if path not in _CONFIG_SCALAR_PATHS and path not in _ENV_CONFIG_SCALAR_PATHS:
        return _blocked_edit(f"Unsupported config path {path!r}.")
    try:
        if path in _ENV_CONFIG_SCALAR_PATHS:
            loaded = _load_or_edit_problem(project, explicit_env=explicit_env)
            if isinstance(loaded, ProjectEditResult):
                return loaded
            updated = _set_active_environment_value(loaded, path, value)
        else:
            schema = _load_raw_schema(project)
            if isinstance(schema, ProjectEditResult):
                return schema
            updated = set_schema_value(schema, path, value)
    except KeyError:
        return _blocked_edit(f"Unsupported config path {path!r}.")
    return write_schema(project, updated)


def config_upgrade(project: ProjectInspection) -> ConfigUpgradeResult:
    if project.config_path is None:
        message = "Fervis config was not found at config/fervis.json."
        return ConfigUpgradeResult(
            blocked_edits=[BlockedEdit(file=str(PROJECT_CONFIG_PATH), reason=message)]
        )
    try:
        raw_schema = load_json_schema(project.root_path / _project_config_path(project))
        schema, upgraded_from = normalize_project_schema(raw_schema)
    except (ConfigIOError, ValueError) as exc:
        return ConfigUpgradeResult(
            blocked_edits=[
                BlockedEdit(file=str(_project_config_path(project)), reason=str(exc))
            ]
        )
    result = _upgrade_loaded_schema(
        project.root_path / _project_config_path(project),
        schema,
        relative_path=_project_config_path(project),
        upgraded_from=upgraded_from,
        needs_write=upgraded_from is not None,
    )
    auth_path = project.root_path / AUTH_CONFIG_PATH
    if auth_path.is_file():
        try:
            raw_auth_schema = load_json_schema(auth_path)
            auth_schema, auth_upgraded_from = normalize_auth_schema(raw_auth_schema)
        except (ConfigIOError, ValueError) as exc:
            result.blocked_edits.append(
                BlockedEdit(file=str(AUTH_CONFIG_PATH), reason=str(exc))
            )
            return result
        result.merge(
            _upgrade_loaded_schema(
                auth_path,
                auth_schema,
                relative_path=AUTH_CONFIG_PATH,
                upgraded_from=auth_upgraded_from,
                needs_write=auth_upgraded_from is not None,
            )
        )
    return result


def add_django_source(
    project: ProjectInspection,
    *,
    name: str,
    app_modules: tuple[str, ...],
    path_prefixes: tuple[str, ...],
) -> ProjectEditResult:
    return _add_source_schema(
        project,
        {
            "kind": "django_app",
            "name": name,
            "app_modules": list(app_modules),
            "path_prefixes": list(path_prefixes),
        },
    )


def add_fastapi_source(
    project: ProjectInspection,
    *,
    name: str,
    import_paths: tuple[str, ...],
    path_prefixes: tuple[str, ...],
) -> ProjectEditResult:
    return _add_source_schema(
        project,
        {
            "kind": "fastapi_app",
            "name": name,
            "import_paths": list(import_paths),
            "path_prefixes": list(path_prefixes),
        },
    )


def add_flask_source(
    project: ProjectInspection,
    *,
    name: str,
    app: str,
    path_prefixes: tuple[str, ...],
    blueprints: tuple[str, ...],
) -> ProjectEditResult:
    return _add_source_schema(
        project,
        {
            "kind": "flask_app",
            "name": name,
            "app": app,
            "app_args": [],
            "app_kwargs": {},
            "path_prefixes": list(path_prefixes),
            "blueprints": list(blueprints),
        },
    )


_CONFIG_SCALAR_PATHS = frozenset(
    {
        "host.organization_name",
        "host.about_api",
        "routes.prefix",
    }
)


_ENV_CONFIG_SCALAR_PATHS = frozenset(
    {
        "persistence.path",
        "persistence.database",
        "persistence.url_env",
    }
)


def _load_or_problem(
    project: ProjectInspection,
    *,
    explicit_env: str | None = None,
) -> ResolvedProjectConfig | ConfigCommandResult:
    try:
        loaded = load_project_json_config(
            project.root_path,
            config_path=_project_config_path(project),
            explicit_env=explicit_env,
        )
    except ConfigIOError as exc:
        return ConfigCommandResult(
            payload={"error": {"code": "config_schema_invalid", "message": str(exc)}},
            blocked_edits=[BlockedEdit(file="config/fervis.json", reason=str(exc))],
        )
    return loaded


def _load_or_edit_problem(
    project: ProjectInspection,
    *,
    explicit_env: str | None = None,
) -> ResolvedProjectConfig | ProjectEditResult:
    try:
        return load_project_json_config(
            project.root_path,
            config_path=_project_config_path(project),
            explicit_env=explicit_env,
        )
    except ConfigIOError as exc:
        return _blocked_edit(str(exc))


def _load_schema(project: ProjectInspection) -> dict[str, object] | ProjectEditResult:
    return _load_raw_schema(project)


def _load_raw_schema(
    project: ProjectInspection,
) -> dict[str, object] | ProjectEditResult:
    try:
        raw_schema = load_json_schema(project.root_path / _project_config_path(project))
        schema, _ = normalize_project_schema(raw_schema)
    except (ConfigIOError, ValueError) as exc:
        return _blocked_edit(str(exc))
    return schema


def write_schema(
    project: ProjectInspection,
    schema: dict[str, object],
) -> ProjectEditResult:
    try:
        normalize_project_schema(schema)
    except ValueError as exc:
        return _blocked_edit(str(exc))
    relative_path = _project_config_path(project)
    config_path = project.root_path / relative_path
    original = config_path.read_text(encoding="utf-8")
    write_json_schema(config_path, schema)
    updated = config_path.read_text(encoding="utf-8")
    if updated == original:
        return ProjectEditResult(skipped_existing=[str(relative_path)])
    try:
        raw_schema = load_json_schema(config_path)
        normalize_project_schema(raw_schema)
    except (ConfigIOError, ValueError) as exc:
        config_path.write_text(original, encoding="utf-8")
        return _blocked_edit(str(exc))
    return ProjectEditResult(changed_files=[str(relative_path)])


def _add_source_schema(
    project: ProjectInspection,
    source: dict[str, object],
) -> ProjectEditResult:
    if not _source_matches_project(source, project.framework):
        return _blocked_edit(
            f"Cannot add {source.get('kind')} source to {project.framework} project."
        )
    result = _load_schema(project)
    if isinstance(result, ProjectEditResult):
        return result
    updated, changed = add_source_schema(result, source)
    if not changed:
        try:
            loaded = load_project_json_config(
                project.root_path,
                config_path=_project_config_path(project),
            )
        except ConfigIOError as exc:
            return _blocked_edit(str(exc))
        return ProjectEditResult(skipped_existing=[str(loaded.config_path)])
    return write_schema(project, updated)


def _schema_value(loaded: ResolvedProjectConfig, path: str) -> object:
    if path in _CONFIG_SCALAR_PATHS:
        return _nested_value(loaded.raw_schema, path)
    if path in _ENV_CONFIG_SCALAR_PATHS or _is_model_policy_path(path):
        return _nested_value(loaded.active_schema, path)
    raise KeyError(path)


def _nested_value(schema: dict[str, object], path: str) -> object:
    value: Any = schema
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _set_active_environment_value(
    loaded: ResolvedProjectConfig,
    path: str,
    value: str,
) -> dict[str, object]:
    updated = copy.deepcopy(loaded.raw_schema)
    environments = _dict(updated.get("environments"))
    environment = _dict(environments.get(loaded.active_environment.name))
    target: Any = environment
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            raise KeyError(path)
        target = target[part]
    if not isinstance(target, dict) or parts[-1] not in target:
        raise KeyError(path)
    target[parts[-1]] = value
    environments[loaded.active_environment.name] = environment
    updated["environments"] = environments
    return updated


def _upgrade_loaded_schema(
    absolute_path: Path,
    schema: dict[str, object],
    *,
    relative_path: Path,
    upgraded_from: str | None,
    needs_write: bool,
) -> ConfigUpgradeResult:
    if not needs_write:
        return ConfigUpgradeResult(skipped_existing=[str(relative_path)])
    write_json_schema(absolute_path, schema)
    return ConfigUpgradeResult(
        changed_files=[str(relative_path)],
        upgraded_configs=[
            {
                "path": str(relative_path),
                "from": str(upgraded_from or ""),
                "to": str(schema.get("schema_version") or ""),
            }
        ],
    )


def _source_matches_project(source: dict[str, object], framework: str) -> bool:
    return (
        (framework == "django" and source.get("kind") == "django_app")
        or (framework == "fastapi" and source.get("kind") == "fastapi_app")
        or (framework == "flask" and source.get("kind") == "flask_app")
    )


def _blocked_config_result(reason: str) -> ConfigCommandResult:
    return ConfigCommandResult(
        payload={"error": {"code": "config_edit_blocked", "message": reason}},
        blocked_edits=[BlockedEdit(file="config/fervis.json", reason=reason)],
    )


def _config_read_blocked(result: ProjectEditResult) -> ConfigCommandResult:
    return ConfigCommandResult(
        payload={
            "error": {
                "code": "config_read_blocked",
                "message": result.blocked_edits[0].reason,
            }
        },
        blocked_edits=result.blocked_edits,
    )


def _blocked_edit(reason: str) -> ProjectEditResult:
    return blocked_edit(file="config/fervis.json", reason=reason)


def _project_config_path(project: ProjectInspection) -> Path:
    return project.config_path or PROJECT_CONFIG_PATH


def _is_model_policy_path(path: str) -> bool:
    return (
        path == "models.providers"
        or path.startswith("models.")
        or ".models.default." in path
    )


def _dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}
