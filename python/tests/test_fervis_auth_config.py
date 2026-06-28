from __future__ import annotations

import json
from pathlib import Path

from fervis.project import discover_project
from fervis.project.auth_config.commands import configure_auth


def test_fervis_auth_configure_django_writes_json_schema(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    _write_project_config(root, framework="django")

    result = configure_auth(
        discover_project(root),
        framework="django-drf",
        security_mode="principal_reauthorization",
        transport_mode="in_process",
    )

    assert not result.is_blocked
    assert result.to_payload()["changed_files"] == ["config/fervis_auth.json"]
    assert _read_json(root / "config" / "fervis_auth.json") == {
        "schema_version": "v0.1",
        "framework": "django",
        "security": {"mode": "principal_reauthorization"},
        "principal": {
            "source": "django_request_user",
            "id_attr": "pk",
        },
        "environments": {
            "local": {
                "transport": {"mode": "in_process"},
            }
        },
    }


def test_fervis_auth_configure_fastapi_writes_env_scoped_transport(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    _write_project_config(root, framework="fastapi", extra_environment="ci")
    monkeypatch.setenv("FERVIS_ENV", "ci")
    monkeypatch.syspath_prepend(str(root))

    result = configure_auth(
        discover_project(root),
        framework="fastapi",
        security_mode="principal_reauthorization",
        transport_mode="in_process",
        principal_dependency="app.api.deps:get_current_user",
        principal_id_attr="id",
        principal_resolver="app.users:get_user_by_id",
    )

    assert not result.is_blocked
    assert _read_json(root / "config" / "fervis_auth.json") == {
        "schema_version": "v0.1",
        "framework": "fastapi",
        "security": {"mode": "principal_reauthorization"},
        "principal": {
            "source": "fastapi_dependency",
            "dependency": "app.api.deps:get_current_user",
            "id_attr": "id",
            "resolver": "app.users:get_user_by_id",
        },
        "environments": {
            "ci": {
                "transport": {"mode": "in_process"},
            }
        },
    }


def test_fervis_auth_configure_flask_login_writes_json_schema(tmp_path: Path) -> None:
    root = _flask_project(tmp_path)
    _write_project_config(root, framework="flask")

    result = configure_auth(
        discover_project(root),
        framework="flask",
        security_mode="principal_reauthorization",
        transport_mode="in_process",
        principal_id_attr="get_id",
        principal_resolver="app.users:get_user_by_id",
    )

    assert not result.is_blocked
    assert _read_json(root / "config" / "fervis_auth.json") == {
        "schema_version": "v0.1",
        "framework": "flask",
        "security": {"mode": "principal_reauthorization"},
        "principal": {
            "source": "flask_login_current_user",
            "id_attr": "get_id",
            "resolver": "app.users:get_user_by_id",
        },
        "environments": {
            "local": {
                "transport": {"mode": "in_process"},
            }
        },
    }


def test_fervis_auth_configure_http_writes_transport_under_selected_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    _write_project_config(root, framework="fastapi")
    _write_overlay(root)
    monkeypatch.syspath_prepend(str(root))

    result = configure_auth(
        discover_project(root),
        framework="fastapi",
        security_mode="principal_reauthorization",
        transport_mode="http",
        principal_dependency="app.api.deps:get_current_user",
        principal_id_attr="id",
        base_url_env="FERVIS_HOST_API_BASE_URL",
        request_overlay_source="app.fervis_host_hooks:http_request_overlay",
        auth_query_params=("tenant",),
    )

    assert not result.is_blocked
    schema = _read_json(root / "config" / "fervis_auth.json")
    assert schema["environments"] == {
        "local": {
            "transport": {
                "mode": "http",
                "base_url_env": "FERVIS_HOST_API_BASE_URL",
                "request_overlay_source": "app.fervis_host_hooks:http_request_overlay",
                "auth_query_params": ["tenant"],
            }
        }
    }


def test_fervis_auth_configure_writes_env_scoped_captured_headers(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_project_config(root, framework="flask")

    result = configure_auth(
        discover_project(root),
        framework="flask",
        security_mode="principal_reauthorization",
        transport_mode="in_process",
        principal_id_attr="get_id",
        principal_resolver="app.users:get_user_by_id",
        credential_headers=("Authorization", "API-TOKEN"),
        credential_key_env="FERVIS_TEST_CREDENTIAL_KEY",
        credential_ttl_seconds=120,
    )

    assert not result.is_blocked
    schema = _read_json(root / "config" / "fervis_auth.json")
    assert schema["environments"] == {
        "local": {
            "transport": {"mode": "in_process"},
            "credentials": {
                "source": "captured_request_headers",
                "headers": ["Authorization", "API-TOKEN"],
                "ttl_seconds": 120,
                "encryption_key_env": "FERVIS_TEST_CREDENTIAL_KEY",
            },
        }
    }
    assert {
        "id": "auth.delegated_credentials_configured",
        "status": "passed",
    } in result.to_payload()["checks"]


def test_fervis_auth_configure_rejects_invalid_fastapi_resolver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    _write_project_config(root, framework="fastapi")
    monkeypatch.syspath_prepend(str(root))

    result = configure_auth(
        discover_project(root),
        framework="fastapi",
        security_mode="principal_reauthorization",
        transport_mode="in_process",
        principal_dependency="app.api.deps:get_current_user",
        principal_id_attr="id",
        principal_resolver="app.users:missing",
    )

    assert result.is_blocked
    assert result.to_payload()["blocked_edits"] == [
        {
            "file": "config/fervis_auth.json",
            "reason": (
                "FastAPI auth principal resolver could not be imported: "
                "module 'app.users' has no attribute 'missing'"
            ),
        }
    ]
    assert not (root / "config" / "fervis_auth.json").exists()


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


def _fastapi_project(tmp_path: Path) -> Path:
    root = tmp_path / "api"
    (root / "app" / "api").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "api" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "api" / "deps.py").write_text(
        "def get_current_user():\n"
        "    return object()\n",
        encoding="utf-8",
    )
    (root / "app" / "users.py").write_text(
        "def get_user_by_id(subject_key):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    return root


def _flask_project(tmp_path: Path) -> Path:
    root = tmp_path / "flask_api"
    (root / "app").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'flask-api'\ndependencies = ['flask>=3.0']\n",
        encoding="utf-8",
    )
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "users.py").write_text(
        "def get_user_by_id(subject_key):\n"
        "    return object()\n",
        encoding="utf-8",
    )
    return root


def _write_overlay(root: Path) -> None:
    (root / "app" / "fervis_host_hooks.py").write_text(
        "def http_request_overlay(read_context_ref, invocation):\n"
        "    return {'headers': {'X-Subject': read_context_ref.key or ''}}\n",
        encoding="utf-8",
    )


def _write_project_config(
    root: Path,
    *,
    framework: str,
    extra_environment: str | None = None,
) -> None:
    source: dict[str, object]
    if framework == "django":
        source = {
            "kind": "django_app",
            "name": "sales",
            "app_modules": ["apps.sales"],
            "path_prefixes": ["/api/"],
        }
    elif framework == "fastapi":
        source = {
            "kind": "fastapi_app",
            "name": "api",
            "import_paths": ["app.main:app"],
            "path_prefixes": ["/api/"],
        }
    else:
        source = {
            "kind": "flask_app",
            "name": "api",
            "app": "app:create_app",
            "app_args": [],
            "app_kwargs": {},
            "path_prefixes": ["/api/"],
            "blueprints": [],
        }
    environments = {
        "local": {
            "models": {
                "default": {"provider": "openai", "model_key": "gpt-5.4-mini"}
            },
            "persistence": {"kind": "sqlite", "path": ".fervis/fervis.sqlite3"},
        }
    }
    if extra_environment is not None:
        environments[extra_environment] = {
            "models": {
                "default": {"provider": "openai", "model_key": "gpt-5.4-mini"}
            },
            "persistence": {
                "kind": "sqlite",
                "path": f".fervis/{extra_environment}.sqlite3",
            },
        }
    schema = {
        "schema_version": "v0.1",
        "framework": framework,
        "default_environment": "local",
        "host": {"organization_name": "", "about_api": ""},
        "routes": {"prefix": "/fervis/" if framework == "django" else "/fervis"},
        "sources": [source],
        "models": {
            "providers": [
                {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
            ]
        },
        "environments": environments,
    }
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "fervis.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
