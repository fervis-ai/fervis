"""Versioned public Fervis config schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .integration import (
    DatabaseUrlPersistence,
    DjangoAppSource,
    DjangoDatabasePersistence,
    FastAPIAppSource,
    FlaskAppSource,
    FervisConfig,
    HostConfig,
    ModelConfig,
    PersistenceTarget,
    ProviderConfig,
    RuntimeRoutes,
    SQLitePersistence,
)
from .config_versions.main import PROJECT_CONFIG_SCHEMA_VERSION
from .source_paths import normalize_source_path_prefixes


FERVIS_CONFIG_SCHEMA_VERSION = PROJECT_CONFIG_SCHEMA_VERSION


def config_from_schema(payload: dict[str, object]) -> FervisConfig:
    validate_config_schema(payload)
    version = str(payload["schema_version"])
    framework = str(payload.get("framework") or "")
    return FervisConfig(
        host=_host_config(_dict(payload.get("host"))),
        routes=RuntimeRoutes(
            prefix=str(_dict(payload.get("routes")).get("prefix") or "")
        ),
        model=_model_config(_dict(payload.get("models"))),
        sources=_sources(tuple(_list(payload.get("sources"))), framework=framework),
        schema_version=version,
        persistence=_persistence(_dict(payload.get("persistence"))),
    )


def validate_config_schema(payload: dict[str, object]) -> None:
    unknown = _unsupported_schema_keys(payload)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"Fervis config contains unsupported keys: {keys}.")
    version = str(payload.get("schema_version") or "")
    if version != FERVIS_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Fervis config schema_version must be {FERVIS_CONFIG_SCHEMA_VERSION}."
        )
    framework = str(payload.get("framework") or "")
    if framework not in {"django", "fastapi", "flask"}:
        raise ValueError("Fervis config framework must be django, fastapi, or flask.")
    host = _require_mapping(payload, "host")
    for key in (
        "organization_name",
        "about_api",
    ):
        _require_string(host, key, allow_blank=True)
    _require_string(host, "timezone")
    routes = _require_mapping(payload, "routes")
    _require_string(routes, "prefix")
    model = _require_mapping(payload, "models")
    default = _require_mapping(model, "default")
    _require_string(default, "provider")
    _require_string(default, "model_key")
    for provider in _require_list(model, "providers"):
        if not isinstance(provider, dict):
            raise ValueError("Fervis config models.providers must contain objects.")
        _require_string(provider, "name")
        _require_string_list(provider, "allowed_model_keys")
    _require_list(payload, "sources")
    for source in _list(payload["sources"]):
        if not isinstance(source, dict):
            raise ValueError("Fervis config sources must contain objects.")
        _validate_source_schema(source, framework=framework)
    _validate_persistence_schema(_require_mapping(payload, "persistence"))


def _validate_source_schema(source: dict[str, object], *, framework: str) -> None:
    _require_string(source, "kind")
    _require_string(source, "name", allow_blank=True)
    _require_string_list(source, "path_prefixes")
    normalize_source_path_prefixes(tuple(_list(source.get("path_prefixes"))))
    if framework == "django":
        if source["kind"] != "django_app":
            raise ValueError("Django Fervis sources must use kind django_app.")
        _require_string_list(source, "app_modules")
        return
    if framework == "fastapi":
        if source["kind"] != "fastapi_app":
            raise ValueError("FastAPI Fervis sources must use kind fastapi_app.")
        _require_string_list(source, "import_paths")
        return
    if source["kind"] != "flask_app":
        raise ValueError("Flask Fervis sources must use kind flask_app.")
    _require_string(source, "app")
    app_args = source.get("app_args", [])
    if not isinstance(app_args, list):
        raise ValueError("Fervis config app_args must be a list.")
    app_kwargs = source.get("app_kwargs", {})
    if not isinstance(app_kwargs, dict):
        raise ValueError("Fervis config app_kwargs must be an object.")
    blueprints = source.get("blueprints", [])
    if not isinstance(blueprints, list) or any(
        not isinstance(item, str) for item in blueprints
    ):
        raise ValueError("Fervis config blueprints must be a list of strings.")


def _validate_persistence_schema(payload: dict[str, object]) -> None:
    kind = str(payload.get("kind") or "")
    if kind == "sqlite":
        _require_string(payload, "path")
        return
    if kind == "django_database":
        _require_string(payload, "database")
        return
    if kind == "database_url":
        _require_string(payload, "url_env")
        return
    raise ValueError(
        "Fervis config persistence.kind must be sqlite, django_database, "
        "or database_url."
    )


def _unsupported_schema_keys(payload: dict[str, object]) -> tuple[str, ...]:
    paths: list[str] = []
    _collect_unknown_keys(
        payload,
        allowed={
            "schema_version",
            "framework",
            "host",
            "routes",
            "models",
            "sources",
            "persistence",
        },
        label="",
        paths=paths,
    )
    host = payload.get("host")
    if isinstance(host, dict):
        _collect_unknown_keys(
            host,
            allowed={"organization_name", "about_api", "timezone"},
            label="host",
            paths=paths,
        )
    routes = payload.get("routes")
    if isinstance(routes, dict):
        _collect_unknown_keys(routes, allowed={"prefix"}, label="routes", paths=paths)
    model = payload.get("models")
    if isinstance(model, dict):
        _collect_unknown_keys(
            model,
            allowed={"default", "providers"},
            label="models",
            paths=paths,
        )
        default = model.get("default")
        if isinstance(default, dict):
            _collect_unknown_keys(
                default,
                allowed={"provider", "model_key"},
                label="models.default",
                paths=paths,
            )
        for index, provider in enumerate(_list(model.get("providers"))):
            if isinstance(provider, dict):
                _collect_unknown_keys(
                    provider,
                    allowed={"name", "allowed_model_keys"},
                    label=f"models.providers[{index}]",
                    paths=paths,
                )
    for index, source in enumerate(_list(payload.get("sources"))):
        if not isinstance(source, dict):
            continue
        kind = str(source.get("kind") or "")
        allowed = {"kind", "name", "path_prefixes"}
        if kind == "django_app":
            allowed.add("app_modules")
        if kind == "fastapi_app":
            allowed.add("import_paths")
        if kind == "flask_app":
            allowed.update({"app", "app_args", "app_kwargs", "blueprints"})
        _collect_unknown_keys(
            source,
            allowed=allowed,
            label=f"sources[{index}]",
            paths=paths,
        )
    persistence = payload.get("persistence")
    if isinstance(persistence, dict):
        kind = str(persistence.get("kind") or "")
        allowed = {"kind"}
        if kind == "sqlite":
            allowed.add("path")
        if kind == "django_database":
            allowed.add("database")
        if kind == "database_url":
            allowed.add("url_env")
        _collect_unknown_keys(
            persistence,
            allowed=allowed,
            label="persistence",
            paths=paths,
        )
    return tuple(sorted(paths))


def _collect_unknown_keys(
    payload: dict[str, object],
    *,
    allowed: set[str],
    label: str,
    paths: list[str],
) -> None:
    for key in payload:
        if key in allowed:
            continue
        paths.append(f"{label}.{key}" if label else str(key))


def config_to_schema(config: FervisConfig, *, framework: str) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "framework": framework,
        "host": {
            "organization_name": config.host.organization_name,
            "about_api": config.host.about_api,
            "timezone": config.host.timezone,
        },
        "routes": {"prefix": config.routes.prefix},
        "models": {
            "default": {
                "provider": config.model.default_provider,
                "model_key": config.model.default_model_key,
            },
            "providers": [
                {
                    "name": provider.name,
                    "allowed_model_keys": list(provider.allowed_model_keys),
                }
                for provider in config.model.providers
            ],
        },
        "sources": [_source_schema(source) for source in config.sources],
        "persistence": _persistence_schema(config.persistence),
    }


def set_schema_value(
    payload: dict[str, object],
    path: str,
    value: object,
) -> dict[str, object]:
    updated = deepcopy(payload)
    parts = path.split(".")
    if not parts:
        raise KeyError(path)
    target: Any = updated
    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            raise KeyError(path)
        target = target[part]
    if not isinstance(target, dict) or parts[-1] not in target:
        raise KeyError(path)
    target[parts[-1]] = value
    return updated


def add_source_schema(
    payload: dict[str, object],
    source: dict[str, object],
) -> tuple[dict[str, object], bool]:
    updated = deepcopy(payload)
    sources = _list(updated.get("sources"))
    source_name = str(source.get("name") or "")
    if any(
        str(item.get("name") or "") == source_name
        for item in sources
        if isinstance(item, dict)
    ):
        return updated, False
    sources.append(dict(source))
    updated["sources"] = sources
    return updated, True


def _host_config(payload: dict[str, object]) -> HostConfig:
    return HostConfig(
        timezone=str(payload.get("timezone") or ""),
        organization_name=str(payload.get("organization_name") or ""),
        about_api=str(payload.get("about_api") or ""),
    )


def _model_config(payload: dict[str, object]) -> ModelConfig:
    default = _dict(payload.get("default"))
    return ModelConfig(
        default_provider=str(default.get("provider") or ""),
        default_model_key=str(default.get("model_key") or ""),
        providers=[
            ProviderConfig(
                name=str(item.get("name") or ""),
                allowed_model_keys=[
                    str(model_key)
                    for model_key in _list(item.get("allowed_model_keys"))
                ],
            )
            for item in _list(payload.get("providers"))
            if isinstance(item, dict)
        ],
    )


def _sources(
    values: tuple[object, ...],
    *,
    framework: str,
) -> list[DjangoAppSource | FastAPIAppSource | FlaskAppSource]:
    sources: list[DjangoAppSource | FastAPIAppSource | FlaskAppSource] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        kind = str(value.get("kind") or "")
        if framework == "django" or kind == "django_app":
            sources.append(
                DjangoAppSource(
                    name=str(value.get("name") or ""),
                    app_modules=[str(item) for item in _list(value.get("app_modules"))],
                    path_prefixes=[
                        str(item) for item in _list(value.get("path_prefixes"))
                    ],
                )
            )
            continue
        if framework == "fastapi" or kind == "fastapi_app":
            sources.append(
                FastAPIAppSource(
                    name=str(value.get("name") or ""),
                    import_paths=[
                        str(item) for item in _list(value.get("import_paths"))
                    ],
                    path_prefixes=[
                        str(item) for item in _list(value.get("path_prefixes"))
                    ],
                )
            )
            continue
        sources.append(
            FlaskAppSource(
                name=str(value.get("name") or ""),
                app=str(value.get("app") or ""),
                app_args=list(_list(value.get("app_args"))),
                app_kwargs=dict(value.get("app_kwargs") or {}),
                path_prefixes=[str(item) for item in _list(value.get("path_prefixes"))],
                blueprints=[str(item) for item in _list(value.get("blueprints"))],
            )
        )
    return sources


def _persistence(payload: dict[str, object]) -> PersistenceTarget:
    kind = str(payload.get("kind") or "sqlite")
    if kind == "django_database":
        return DjangoDatabasePersistence(
            database=str(payload.get("database") or "default")
        )
    if kind == "database_url":
        return DatabaseUrlPersistence(
            url_env=str(payload.get("url_env") or "FERVIS_DATABASE_URL")
        )
    return SQLitePersistence(path=str(payload.get("path") or ".fervis/fervis.sqlite3"))


def _source_schema(
    source: DjangoAppSource | FastAPIAppSource | FlaskAppSource,
) -> dict[str, object]:
    if isinstance(source, DjangoAppSource):
        return {
            "kind": "django_app",
            "name": source.name,
            "app_modules": list(source.app_modules),
            "path_prefixes": list(source.path_prefixes),
        }
    if isinstance(source, FastAPIAppSource):
        return {
            "kind": "fastapi_app",
            "name": source.name,
            "import_paths": list(source.import_paths),
            "path_prefixes": list(source.path_prefixes),
        }
    return {
        "kind": "flask_app",
        "name": source.name,
        "app": source.app,
        "app_args": list(source.app_args),
        "app_kwargs": dict(source.app_kwargs),
        "path_prefixes": list(source.path_prefixes),
        "blueprints": list(source.blueprints),
    }


def _persistence_schema(persistence: PersistenceTarget) -> dict[str, object]:
    if isinstance(persistence, SQLitePersistence):
        return {"kind": "sqlite", "path": persistence.path}
    if isinstance(persistence, DjangoDatabasePersistence):
        return {"kind": "django_database", "database": persistence.database}
    return {"kind": "database_url", "url_env": persistence.url_env}


def _dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _require_mapping(
    payload: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Fervis config {key} must be an object.")
    return value


def _require_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Fervis config {key} must be a list.")
    return value


def _require_string_list(payload: dict[str, object], key: str) -> None:
    values = _require_list(payload, key)
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"Fervis config {key} must be a list of strings.")


def _require_string(
    payload: dict[str, object],
    key: str,
    *,
    allow_blank: bool = False,
) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        requirement = "a string" if allow_blank else "a non-empty string"
        raise ValueError(f"Fervis config {key} must be {requirement}.")
