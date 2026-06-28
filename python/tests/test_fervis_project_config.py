from __future__ import annotations

import json
from pathlib import Path

from fervis.project import discover_project
from fervis.project.configuration import (
    ConfigProblem,
    LoadedFervisConfig,
    load_fervis_project_config,
)
from fervis.project.integration import FlaskAppSource
from fervis.project.source_scope import configured_django_source_scopes


def test_load_fervis_project_config_accepts_versioned_json_config(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    _write_schema_config(root, _valid_django_schema())

    loaded = load_fervis_project_config(discover_project(root))

    assert isinstance(loaded, LoadedFervisConfig)
    assert loaded.schema["schema_version"] == "v0.1"
    assert loaded.integration.framework == "django"
    assert loaded.config.model.default_model_ref == "openai:gpt-5.4-mini"
    assert loaded.active_environment.name == "local"


def test_load_fervis_project_config_accepts_flask_json_config(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_schema_config(root, _valid_flask_schema())

    loaded = load_fervis_project_config(discover_project(root))

    assert isinstance(loaded, LoadedFervisConfig)
    assert loaded.integration.framework == "flask"
    assert loaded.config.sources == [
        FlaskAppSource(
            name="api",
            app="app:create_app",
            app_args=[],
            app_kwargs={},
            path_prefixes=["/api/"],
            blueprints=[],
        )
    ]


def test_load_fervis_project_config_rejects_cached_flask_routes(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    schema = _valid_flask_schema()
    source = dict(schema["sources"][0])
    source["routes"] = [{"path": "/api/orders/", "method": "GET"}]
    schema["sources"] = [source]
    _write_schema_config(root, schema)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_schema_invalid",
        message="unsupported keys: sources[0].routes",
    )


def test_configured_django_source_scopes_use_json_config(
    settings,
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    _write_schema_config(root, _valid_django_schema())
    settings.FERVIS_CONFIG_PATH = "config/fervis.json"
    settings.BASE_DIR = root

    scopes = configured_django_source_scopes()

    assert [scope.name for scope in scopes] == ["sales"]
    assert scopes[0].app_modules == ("apps.sales",)
    assert scopes[0].path_prefixes == ("/api/",)


def test_load_fervis_project_config_rejects_missing_json_config(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_missing",
        message="Fervis config was not found at config/fervis.json.",
    )


def test_load_fervis_project_config_rejects_invalid_schema_version(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    schema = _valid_django_schema()
    schema["schema_version"] = "v9"
    _write_schema_config(root, schema)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_schema_invalid",
        message="Fervis config schema_version must be v0.1.",
    )


def test_load_fervis_project_config_rejects_unknown_schema_keys(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    schema = _valid_django_schema()
    schema["system_prompt"] = "ignored"
    host = dict(schema["host"])
    host["timezone"] = "UTC"
    schema["host"] = host
    _write_schema_config(root, schema)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_schema_invalid",
        message="unsupported keys: system_prompt",
    )


def test_load_fervis_project_config_rejects_wrong_framework(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    schema = {**_valid_fastapi_schema(), "framework": "fastapi"}
    _write_schema_config(root, schema)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_framework_mismatch",
        message="config framework fastapi does not match detected project framework django.",
    )


def test_load_fervis_project_config_rejects_empty_source_details(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    schema = _valid_django_schema()
    schema["sources"] = [
        {"kind": "django_app", "name": "sales", "app_modules": [], "path_prefixes": ["/api/"]}
    ]
    _write_schema_config(root, schema)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_schema_invalid",
        message="app_modules must contain at least one value.",
    )


def test_load_fervis_project_config_rejects_undeclared_default_provider(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    schema = _valid_django_schema()
    environments = schema["environments"]
    assert isinstance(environments, dict)
    local = environments["local"]
    assert isinstance(local, dict)
    local["models"] = {
        "default": {"provider": "anthropic", "model_key": "claude"}
    }
    _write_schema_config(root, schema)

    problem = load_fervis_project_config(discover_project(root))

    assert problem == ConfigProblem(
        code="config_schema_invalid",
        message=(
            "environments.local.models.default provider 'anthropic' "
            "is not declared in models.providers."
        ),
    )


def _valid_django_schema() -> dict[str, object]:
    return {
        "schema_version": "v0.1",
        "framework": "django",
        "default_environment": "local",
        "host": {
            "organization_name": "",
            "about_api": "",
        },
        "routes": {"prefix": "/fervis/"},
        "models": {
            "providers": [
                {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
            ],
        },
        "sources": [
            {
                "kind": "django_app",
                "name": "sales",
                "app_modules": ["apps.sales"],
                "path_prefixes": ["/api/"],
            }
        ],
        "environments": {
            "local": {
                "models": {
                    "default": {"provider": "openai", "model_key": "gpt-5.4-mini"}
                },
                "persistence": {"kind": "sqlite", "path": ".fervis/fervis.sqlite3"},
            }
        },
    }


def _valid_fastapi_schema() -> dict[str, object]:
    schema = _valid_django_schema()
    schema["framework"] = "fastapi"
    schema["routes"] = {"prefix": "/fervis"}
    schema["sources"] = [
        {
            "kind": "fastapi_app",
            "name": "api",
            "import_paths": ["app.main:app"],
            "path_prefixes": ["/api/"],
        }
    ]
    return schema


def _valid_flask_schema() -> dict[str, object]:
    schema = _valid_django_schema()
    schema["framework"] = "flask"
    schema["routes"] = {"prefix": "/fervis/"}
    schema["sources"] = [
        {
            "kind": "flask_app",
            "name": "api",
            "app": "app:create_app",
            "app_args": [],
            "app_kwargs": {},
            "path_prefixes": ["/api/"],
            "blueprints": [],
        }
    ]
    return schema


def _django_project(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    root.mkdir()
    (root / "manage.py").write_text(
        "#!/usr/bin/env python\n"
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
    return root


def _flask_project(tmp_path: Path) -> Path:
    root = tmp_path / "flask_api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'flask-api'\ndependencies = ['flask>=3.0']\n",
        encoding="utf-8",
    )
    return root


def _write_schema_config(root: Path, schema: dict[str, object]) -> None:
    config_dir = root / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "fervis.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
