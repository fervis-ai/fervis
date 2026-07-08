from __future__ import annotations

import json
from pathlib import Path

import pytest

from fervis.project.config_io import (
    ActiveEnvironment,
    ConfigIOError,
    load_auth_json_config,
    load_project_json_config,
    write_json_schema,
)


def test_project_json_config_resolves_default_environment(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    _write_json(root / "config" / "fervis.json", _project_schema())

    loaded = load_project_json_config(root)

    assert loaded.config_path == Path("config/fervis.json")
    assert loaded.active_environment == ActiveEnvironment(
        name="dev",
        source="default_environment",
    )
    assert loaded.upgraded_from is None
    assert loaded.needs_write is False
    assert loaded.active_schema["models"] == {
        "default": {"provider": "openai", "model_key": "gpt-5.4-mini"},
        "providers": [
            {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]},
            {"name": "opencode", "allowed_model_keys": ["deepseek-v4-pro"]},
        ],
    }
    assert loaded.active_schema["persistence"] == {
        "kind": "sqlite",
        "path": ".fervis/dev.sqlite3",
    }
    assert "environments" not in loaded.active_schema


def test_project_json_config_prefers_explicit_env_over_fervis_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project_root(tmp_path)
    schema = _project_schema()
    environments = schema["environments"]
    assert isinstance(environments, dict)
    environments["staging"] = {
        "models": {"default": {"provider": "opencode", "model_key": "deepseek-v4-pro"}},
        "persistence": {"kind": "sqlite", "path": ".fervis/staging.sqlite3"},
    }
    environments["ci"] = {
        "models": {"default": {"provider": "openai", "model_key": "gpt-5.4-mini"}},
        "persistence": {"kind": "sqlite", "path": ".fervis/ci.sqlite3"},
    }
    _write_json(root / "config" / "fervis.json", schema)
    monkeypatch.setenv("FERVIS_ENV", "staging")

    loaded = load_project_json_config(root, explicit_env="ci")

    assert loaded.active_environment == ActiveEnvironment(name="ci", source="explicit")
    assert loaded.active_schema["persistence"] == {
        "kind": "sqlite",
        "path": ".fervis/ci.sqlite3",
    }


def test_project_json_config_uses_fervis_env_before_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project_root(tmp_path)
    schema = _project_schema()
    environments = schema["environments"]
    assert isinstance(environments, dict)
    environments["staging"] = {
        "models": {"default": {"provider": "opencode", "model_key": "deepseek-v4-pro"}},
        "persistence": {"kind": "sqlite", "path": ".fervis/staging.sqlite3"},
    }
    _write_json(root / "config" / "fervis.json", schema)
    monkeypatch.setenv("FERVIS_ENV", "staging")

    loaded = load_project_json_config(root)

    assert loaded.active_environment == ActiveEnvironment(
        name="staging",
        source="FERVIS_ENV",
    )
    assert loaded.active_schema["models"]["default"] == {
        "provider": "opencode",
        "model_key": "deepseek-v4-pro",
    }


def test_auth_json_config_uses_project_active_environment(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    _write_json(root / "config" / "fervis.json", _project_schema())
    _write_json(root / "config" / "fervis_auth.json", _auth_schema())
    project = load_project_json_config(root)

    loaded = load_auth_json_config(root, active_environment=project.active_environment)

    assert loaded.config_path == Path("config/fervis_auth.json")
    assert loaded.active_environment == project.active_environment
    assert loaded.active_schema["transport"] == {"mode": "in_process"}
    assert "environments" not in loaded.active_schema


def test_auth_json_config_rejects_top_level_transport(tmp_path: Path) -> None:
    root = _project_root(tmp_path)
    schema = _auth_schema()
    schema["transport"] = {"mode": "in_process"}
    _write_json(root / "config" / "fervis_auth.json", schema)

    with pytest.raises(ConfigIOError, match="unsupported keys: transport"):
        load_auth_json_config(
            root,
            active_environment=ActiveEnvironment(
                name="dev",
                source="default_environment",
            ),
        )


def test_write_json_schema_writes_canonical_json(tmp_path: Path) -> None:
    path = tmp_path / "config" / "fervis.json"

    write_json_schema(path, {"schema_version": "v0.1", "framework": "django"})

    assert path.read_text(encoding="utf-8") == (
        '{\n'
        '  "framework": "django",\n'
        '  "schema_version": "v0.1"\n'
        "}\n"
    )


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    (root / "config").mkdir(parents=True)
    return root


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_schema() -> dict[str, object]:
    return {
        "schema_version": "v0.1",
        "framework": "django",
        "default_environment": "dev",
        "host": {
            "organization_name": "Shop",
            "about_api": "The Shop API exposes orders and inventory records.",
            "timezone": "UTC",
        },
        "routes": {"prefix": "/fervis/"},
        "sources": [
            {
                "kind": "django_app",
                "name": "sales",
                "app_modules": ["apps.sales"],
                "path_prefixes": ["/api/"],
            }
        ],
        "models": {
            "providers": [
                {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]},
                {"name": "opencode", "allowed_model_keys": ["deepseek-v4-pro"]},
            ],
        },
        "environments": {
            "dev": {
                "models": {
                    "default": {
                        "provider": "openai",
                        "model_key": "gpt-5.4-mini",
                    }
                },
                "persistence": {"kind": "sqlite", "path": ".fervis/dev.sqlite3"},
            }
        },
    }


def _auth_schema() -> dict[str, object]:
    return {
        "schema_version": "v0.1",
        "framework": "django",
        "security": {"mode": "principal_reauthorization"},
        "principal": {"source": "django_request_user", "id_attr": "pk"},
        "environments": {
            "dev": {
                "transport": {"mode": "in_process"},
            }
        },
    }
