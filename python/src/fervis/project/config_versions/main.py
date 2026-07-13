"""Current Fervis project JSON schema contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from fervis.model_io.providers.specs import supported_provider_specs
from fervis.project.config_versions.common import (
    reject_unknown_keys,
    require_list,
    require_mapping,
    require_string,
    require_string_list,
)

PROJECT_CONFIG_SCHEMA_VERSION = "v0.1"


def normalize_project_schema(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    """Return the current project schema plus the source version, if upgraded."""

    version = payload.get("schema_version")
    if version != PROJECT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Fervis config schema_version must be {PROJECT_CONFIG_SCHEMA_VERSION}."
        )
    schema = deepcopy(dict(payload))
    validate_project_schema(schema)
    return schema, None


def validate_project_schema(schema: Mapping[str, object]) -> None:
    reject_unknown_keys(
        schema,
        allowed={
            "schema_version",
            "framework",
            "default_environment",
            "host",
            "routes",
            "sources",
            "models",
            "environments",
        },
    )
    if schema.get("schema_version") != PROJECT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Fervis config schema_version must be {PROJECT_CONFIG_SCHEMA_VERSION}."
        )
    framework = require_string(schema, "framework")
    if framework not in {"django", "fastapi", "flask"}:
        raise ValueError("framework must be django, fastapi, or flask.")
    require_string(schema, "default_environment")
    _validate_host(require_mapping(schema, "host"))
    _validate_routes(require_mapping(schema, "routes"))
    _validate_sources(require_list(schema, "sources"), framework=framework)
    global_models = require_mapping(schema, "models")
    _validate_models(global_models)
    environments = require_mapping(schema, "environments")
    default_environment = str(schema["default_environment"])
    if default_environment not in environments:
        raise ValueError(
            f"default_environment {default_environment!r} is not declared "
            "in environments."
        )
    for name, environment in environments.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("environment names must be non-empty strings.")
        if not isinstance(environment, Mapping):
            raise ValueError(f"environment {name!r} must be an object.")
        _validate_environment(environment, name=name, global_models=global_models)


def active_project_schema(
    schema: Mapping[str, object],
    *,
    environment_name: str,
) -> dict[str, object]:
    environments = require_mapping(schema, "environments")
    environment = environments.get(environment_name)
    if not isinstance(environment, Mapping):
        raise ValueError(f"Fervis environment {environment_name!r} is not declared.")
    models = require_mapping(schema, "models")
    environment_models = require_mapping(environment, "models")
    return {
        "schema_version": schema["schema_version"],
        "framework": schema["framework"],
        "host": deepcopy(schema["host"]),
        "routes": deepcopy(schema["routes"]),
        "sources": deepcopy(schema["sources"]),
        "models": {
            "default": deepcopy(environment_models["default"]),
            "providers": deepcopy(models["providers"]),
        },
        "persistence": deepcopy(environment["persistence"]),
    }


def _validate_host(host: Mapping[str, object]) -> None:
    reject_unknown_keys(
        host,
        allowed={"organization_name", "about_api", "timezone"},
        prefix="host",
    )
    require_string(host, "organization_name", allow_blank=True)
    require_string(host, "about_api", allow_blank=True)
    require_string(host, "timezone")


def _validate_routes(routes: Mapping[str, object]) -> None:
    reject_unknown_keys(routes, allowed={"prefix"}, prefix="routes")
    require_string(routes, "prefix")


def _validate_sources(sources: list[object], *, framework: str) -> None:
    if not sources:
        raise ValueError("sources must contain at least one source.")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"sources[{index}] must be an object.")
        kind = require_string(source, "kind")
        if kind == "django_app":
            if framework != "django":
                raise ValueError("django_app sources require framework django.")
            reject_unknown_keys(
                source,
                allowed={"kind", "name", "app_modules", "path_prefixes"},
                prefix=f"sources[{index}]",
            )
            require_string(source, "name")
            require_string_list(source, "app_modules")
            require_string_list(source, "path_prefixes")
        elif kind == "fastapi_app":
            if framework != "fastapi":
                raise ValueError("fastapi_app sources require framework fastapi.")
            reject_unknown_keys(
                source,
                allowed={"kind", "name", "import_paths", "path_prefixes"},
                prefix=f"sources[{index}]",
            )
            require_string(source, "name")
            require_string_list(source, "import_paths")
            require_string_list(source, "path_prefixes")
        elif kind == "flask_app":
            if framework != "flask":
                raise ValueError("flask_app sources require framework flask.")
            reject_unknown_keys(
                source,
                allowed={
                    "kind",
                    "name",
                    "app",
                    "app_args",
                    "app_kwargs",
                    "path_prefixes",
                    "blueprints",
                },
                prefix=f"sources[{index}]",
            )
            require_string(source, "name")
            require_string(source, "app")
            app_args = source.get("app_args", [])
            if not isinstance(app_args, list):
                raise ValueError(f"sources[{index}].app_args must be a list.")
            app_kwargs = source.get("app_kwargs", {})
            if not isinstance(app_kwargs, Mapping):
                raise ValueError(f"sources[{index}].app_kwargs must be an object.")
            require_string_list(source, "path_prefixes")
            blueprints = source.get("blueprints", [])
            if not isinstance(blueprints, list) or any(
                not isinstance(item, str) for item in blueprints
            ):
                raise ValueError(
                    f"sources[{index}].blueprints must be a list of strings."
                )
        else:
            raise ValueError(f"Unsupported source kind {kind!r}.")


def _validate_models(models: Mapping[str, object]) -> None:
    reject_unknown_keys(models, allowed={"providers"}, prefix="models")
    providers = require_list(models, "providers")
    if not providers:
        raise ValueError("models.providers must contain at least one provider.")
    supported = supported_provider_specs()
    seen: set[str] = set()
    for index, provider in enumerate(providers):
        if not isinstance(provider, Mapping):
            raise ValueError(f"models.providers[{index}] must be an object.")
        reject_unknown_keys(
            provider,
            allowed={"name", "allowed_model_keys"},
            prefix=f"models.providers[{index}]",
        )
        name = require_string(provider, "name")
        if name not in supported:
            supported_names = ", ".join(sorted(supported))
            raise ValueError(
                f"Unsupported Fervis provider {name!r}. Supported: {supported_names}."
            )
        if name in seen:
            raise ValueError(f"models.providers contains duplicate provider {name!r}.")
        seen.add(name)
        require_string_list(provider, "allowed_model_keys")


def _validate_environment(
    environment: Mapping[str, object],
    *,
    name: str,
    global_models: Mapping[str, object],
) -> None:
    reject_unknown_keys(
        environment,
        allowed={"models", "persistence"},
        prefix=f"environments.{name}",
    )
    models = require_mapping(environment, "models")
    reject_unknown_keys(
        models,
        allowed={"default"},
        prefix=f"environments.{name}.models",
    )
    default = require_mapping(models, "default")
    reject_unknown_keys(
        default,
        allowed={"provider", "model_key"},
        prefix=f"environments.{name}.models.default",
    )
    provider = require_string(default, "provider")
    model_key = require_string(default, "model_key")
    allowed = _allowed_models_by_provider(global_models.get("providers"))
    if provider not in allowed:
        raise ValueError(
            f"environments.{name}.models.default provider {provider!r} "
            "is not declared in models.providers."
        )
    if model_key not in allowed[provider]:
        raise ValueError(
            f"environments.{name}.models.default model_key {model_key!r} "
            f"is not allowed for provider {provider!r}."
        )
    _validate_persistence(require_mapping(environment, "persistence"), name=name)


def _allowed_models_by_provider(providers: object) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    if not isinstance(providers, list):
        raise ValueError("models.providers must be a list.")
    for provider in providers:
        if isinstance(provider, Mapping):
            name = str(provider.get("name") or "")
            keys = provider.get("allowed_model_keys")
            if isinstance(keys, list):
                allowed[name] = {str(key) for key in keys if isinstance(key, str)}
    return allowed


def _validate_persistence(persistence: Mapping[str, object], *, name: str) -> None:
    kind = require_string(persistence, "kind")
    if kind == "sqlite":
        reject_unknown_keys(
            persistence,
            allowed={"kind", "path"},
            prefix=f"environments.{name}.persistence",
        )
        require_string(persistence, "path")
        return
    if kind == "django_database":
        reject_unknown_keys(
            persistence,
            allowed={"kind", "database"},
            prefix=f"environments.{name}.persistence",
        )
        require_string(persistence, "database")
        return
    if kind == "database_url":
        reject_unknown_keys(
            persistence,
            allowed={"kind", "url_env"},
            prefix=f"environments.{name}.persistence",
        )
        require_string(persistence, "url_env")
        return
    raise ValueError(f"Unsupported persistence kind {kind!r}.")
