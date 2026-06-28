"""Public model command services."""

from __future__ import annotations

import copy
from pathlib import Path

from fervis.interfaces.agent.commands import commands, render_command
from fervis.model_io.models import ModelRef
from fervis.model_io.providers.specs import (
    ProviderSpec,
    supported_provider_spec,
    supported_provider_specs,
)

from .config_commands import ConfigCommandResult, write_schema
from .config_io import ConfigIOError, load_json_schema, load_project_json_config
from .config_versions.main import normalize_project_schema
from .discovery import ProjectInspection
from .edit_result import BlockedEdit, ProjectEditResult


def models_view(project: ProjectInspection) -> ConfigCommandResult:
    if project.config_path is None:
        message = "Fervis config was not found at config/fervis.json."
        return ConfigCommandResult(
            payload={"error": {"code": "config_missing", "message": message}},
            blocked_edits=[BlockedEdit(file="config/fervis.json", reason=message)],
        )
    try:
        loaded = load_project_json_config(
            project.root_path,
            config_path=_project_config_path(project),
        )
    except ConfigIOError as exc:
        return ConfigCommandResult(
            payload={"error": {"code": "config_schema_invalid", "message": str(exc)}},
            blocked_edits=[BlockedEdit(file="config/fervis.json", reason=str(exc))],
        )
    model = loaded.active_schema.get("models")
    active_model = _active_model_ref(model)
    return ConfigCommandResult(
        payload={
            "active_model": active_model,
            "active_environment": {
                "name": loaded.active_environment.name,
                "source": loaded.active_environment.source,
            },
            "providers": [_provider_payload(spec) for spec in _provider_specs()],
        }
    )


def allow_model(project: ProjectInspection, model_ref: str) -> ProjectEditResult:
    try:
        ref = ModelRef.parse(model_ref)
        supported_provider_spec(ref.provider)
    except ValueError as error:
        return _blocked(str(error))
    try:
        if project.config_path is None:
            return _blocked("Fervis config was not found at config/fervis.json.")
        raw_schema = load_json_schema(project.root_path / _project_config_path(project))
        schema, _ = normalize_project_schema(raw_schema)
    except (ConfigIOError, ValueError) as exc:
        return _blocked(str(exc))
    updated = _allow_model(schema, ref)
    if updated == schema:
        return ProjectEditResult(skipped_existing=[str(_project_config_path(project))])
    return write_schema(project, updated)


def use_model(
    project: ProjectInspection,
    model_ref: str,
    *,
    explicit_env: str | None = None,
) -> ProjectEditResult:
    try:
        ref = ModelRef.parse(model_ref)
        supported_provider_spec(ref.provider)
    except ValueError as error:
        return _blocked(str(error))
    try:
        if project.config_path is None:
            return _blocked("Fervis config was not found at config/fervis.json.")
        loaded = load_project_json_config(
            project.root_path,
            config_path=_project_config_path(project),
            explicit_env=explicit_env,
        )
    except ConfigIOError as exc:
        return _blocked(str(exc))
    if not _model_allowed(loaded.raw_schema, ref):
        allow_command = render_command(commands.model_allow(str(ref)))
        return _blocked(f"Model {ref} is not allowed. Run `{allow_command}` first.")
    updated = _use_model(
        loaded.raw_schema,
        ref,
        environment=loaded.active_environment.name,
    )
    return write_schema(project, updated)


def _provider_specs() -> tuple[ProviderSpec, ...]:
    specs = supported_provider_specs()
    return tuple(specs[name] for name in sorted(specs))


def _provider_payload(spec: ProviderSpec) -> dict[str, object]:
    model_ref = f"{spec.name}:{spec.default_model}"
    return {
        "name": spec.name,
        "transport": spec.transport,
        "default_model": spec.default_model,
        "default_model_ref": model_ref,
        "strict_tools": spec.strict_tool_certified,
        "use_command": render_command(commands.model_use(model_ref)),
    }


def _active_model_ref(model: object) -> str:
    if not isinstance(model, dict):
        return ""
    default = model.get("default")
    if not isinstance(default, dict):
        return ""
    provider = str(default.get("provider") or "")
    model_key = str(default.get("model_key") or "")
    if not provider or not model_key:
        return ""
    return str(ModelRef(provider=provider, model_id=model_key))


def _allow_model(schema: dict[str, object], ref: ModelRef) -> dict[str, object]:
    updated = _copy_schema(schema)
    models = _dict(updated.setdefault("models", {}))
    providers = _provider_entries(models.setdefault("providers", []))
    for provider in providers:
        if provider.get("name") != ref.provider:
            continue
        allowed = _string_list(provider.setdefault("allowed_model_keys", []))
        if ref.model_id not in allowed:
            allowed.append(ref.model_id)
        provider["allowed_model_keys"] = allowed
        models["providers"] = providers
        updated["models"] = models
        return updated
    providers.append({"name": ref.provider, "allowed_model_keys": [ref.model_id]})
    models["providers"] = providers
    updated["models"] = models
    return updated


def _use_model(
    schema: dict[str, object],
    ref: ModelRef,
    *,
    environment: str,
) -> dict[str, object]:
    updated = _copy_schema(schema)
    environments = _dict(updated["environments"])
    target = _dict(environments[environment])
    target["models"] = {
        "default": {"provider": ref.provider, "model_key": ref.model_id}
    }
    environments[environment] = target
    updated["environments"] = environments
    return updated


def _model_allowed(schema: dict[str, object], ref: ModelRef) -> bool:
    models = _dict(schema.get("models"))
    for provider in _provider_entries(models.get("providers")):
        if provider.get("name") != ref.provider:
            continue
        return ref.model_id in _string_list(provider.get("allowed_model_keys"))
    return False


def _copy_schema(schema: dict[str, object]) -> dict[str, object]:
    return copy.deepcopy(schema)


def _provider_entries(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _blocked(reason: str) -> ProjectEditResult:
    return ProjectEditResult(
        blocked_edits=[BlockedEdit(file="config/fervis.json", reason=reason)]
    )


def _project_config_path(project: ProjectInspection) -> Path:
    return project.config_path or Path("config/fervis.json")
