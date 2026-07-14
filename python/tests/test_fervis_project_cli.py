from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from fervis.interfaces.agent.actions import (
    choose_framework_init_action,
    configure_auth_action,
    edit_config_action,
    install_dependencies_action,
    resolve_blocked_edits_action,
    run_doctor_action,
    run_migrate_action,
    set_env_action,
)
from fervis.interfaces.agent.commands import (
    Placeholder,
    commands,
    render_command,
)
from fervis.interfaces.cli.dispatch import (
    run_auth_command,
    run_config_command,
    run_doctor_command,
    run_init_command,
    run_migrate_command,
    run_models_command,
    run_sources_command,
)
from fervis.project import discover_project
from fervis.project.persistence.sqlite_engine import create_sqlite_engine


API_DIR = Path(__file__).resolve().parents[2]


def test_fervis_init_creates_config_and_patches_django_hooks(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["schema"] == "fervis-command-result.v0.1"
    assert envelope["command"] == "init"
    assert envelope["status"] == "succeeded"
    assert envelope["project"] == {
        "framework": "django",
        "config_path": "config/fervis.json",
    }
    assert envelope["payload_schema"] == "fervis-init-result.v0.1"
    assert envelope["payload"] == {
        "changed_files": [
            "config/fervis.json",
            "config/settings.py",
            "config/urls.py",
        ],
        "skipped_existing": [],
        "blocked_edits": [],
    }
    assert (root / "config" / "__init__.py").exists()
    assert (root / "config" / "fervis.json").is_file()
    assert '"rest_framework",' in (root / "config" / "settings.py").read_text(
        encoding="utf-8"
    )
    assert '"fervis.django",' in (root / "config" / "settings.py").read_text(
        encoding="utf-8"
    )
    settings_text = (root / "config" / "settings.py").read_text(encoding="utf-8")
    assert '"fervis.lineage",' in settings_text
    assert '"fervis.run_work.queue.django",' in settings_text
    assert '"fervis.interfaces.django",' in settings_text
    assert 'FERVIS_CONFIG_PATH = "config/fervis.json"' in (
        root / "config" / "settings.py"
    ).read_text(encoding="utf-8")
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in (root / "config" / "urls.py").read_text(encoding="utf-8")
    )
    assert _config_schema(root)["schema_version"] == "v0.1"
    assert _config_schema(root)["environments"]["local"]["persistence"] == {
        "kind": "sqlite",
        "path": ".fervis/fervis.sqlite3",
    }


def test_fervis_init_derives_django_source_from_project_modules(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    app_dir = root / "quickstart"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "apps.py").write_text("", encoding="utf-8")

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"] == [
        {
            "kind": "django_app",
            "name": "default",
            "app_modules": ["orders", "quickstart"],
            "path_prefixes": ["/"],
        }
    ]


def test_fervis_init_blocks_django_without_local_application_modules(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty_django"
    root.mkdir()
    (root / "manage.py").write_text(
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "__init__.py").write_text("", encoding="utf-8")
    (config_dir / "settings.py").write_text(
        "INSTALLED_APPS = []\nROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )
    (config_dir / "urls.py").write_text(
        "from django.urls import path\n\nurlpatterns = []\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": (
                "Could not identify local Django API modules. Add a source "
                "explicitly with `fervis sources add django-app`."
            ),
        }
    ]
    assert envelope["next_actions"] == [resolve_blocked_edits_action()]


def test_fervis_doctor_returns_structured_result_for_django_cli_without_global_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _django_project(tmp_path)
    init = _run_fervis(root, "init", "--framework", "django", "--yes")
    migrate = _run_fervis(root, "migrate")

    doctor = _run_fervis(root, "doctor")

    assert init.returncode == 0
    assert migrate.returncode == 0
    assert doctor.stderr == ""
    envelope = json.loads(doctor.stdout)
    assert envelope["command"] == "doctor"
    assert envelope["payload_schema"] == "fervis-doctor-report.v0.1"
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert "fervis" not in checks["source.catalog"]["message"]


def test_fervis_doctor_unknown_project_does_not_guess_framework(
    tmp_path: Path,
) -> None:
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(tmp_path),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert envelope["next_actions"] == [choose_framework_init_action()]


def test_fervis_config_show_exposes_public_contract(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_config_command(
        ("config", "show"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["command"] == "config.show"
    assert envelope["payload_schema"] == "fervis-config-view.v0.1"
    payload = envelope["payload"]
    assert payload["config_path"] == "config/fervis.json"
    assert payload["schema_version"] == "v0.1"
    assert payload["active_environment"] == {
        "name": "local",
        "source": "default_environment",
    }
    assert payload["host"] == {
        "organization_name": "",
        "about_api": "",
        "timezone": "UTC",
    }
    assert payload["models"]["default"] == {
        "provider": "openai",
        "model_key": "gpt-5.4-mini",
    }
    assert payload["routes"]["prefix"] == "/fervis/"
    assert payload["persistence"] == {
        "kind": "sqlite",
        "path": ".fervis/fervis.sqlite3",
    }
    assert payload["sources"] == [
        {
            "name": "default",
            "kind": "django_app",
            "app_modules": ["orders"],
            "path_prefixes": ["/"],
        }
    ]
    assert "environments" not in payload


def test_fervis_config_get_and_set_patch_generated_scalar(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    set_stdout = StringIO()
    set_exit = run_config_command(
        ("config", "set", "host.about_api", "A factual commerce API."),
        project=discover_project(root),
        stdout=set_stdout,
    )
    get_stdout = StringIO()
    get_exit = run_config_command(
        ("config", "get", "host.about_api"),
        project=discover_project(root),
        stdout=get_stdout,
    )

    set_envelope = json.loads(set_stdout.getvalue())
    get_envelope = json.loads(get_stdout.getvalue())
    assert set_exit == 0
    assert set_envelope["command"] == "config.set"
    assert set_envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert set_envelope["payload"]["blocked_edits"] == []
    assert get_exit == 0
    assert get_envelope["payload"] == {
        "path": "host.about_api",
        "value": "A factual commerce API.",
    }


def test_fervis_config_set_global_scalar_ignores_unrelated_fervis_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    monkeypatch.setenv("FERVIS_ENV", "not-declared")

    exit_code = run_config_command(
        ("config", "set", "host.organization_name", "Global Shop"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["host"]["organization_name"] == "Global Shop"


def test_fervis_config_get_and_set_patch_active_environment_scalar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    schema = _config_schema(root)
    schema["environments"]["qa-branch"] = {
        "models": {"default": {"provider": "openai", "model_key": "gpt-5.4-mini"}},
        "persistence": {"kind": "sqlite", "path": ".fervis/qa.sqlite3"},
    }
    _write_schema_config(root, schema)
    monkeypatch.setenv("FERVIS_ENV", "qa-branch")

    set_stdout = StringIO()
    set_exit = run_config_command(
        ("config", "set", "persistence.path", ".fervis/qa-updated.sqlite3"),
        project=discover_project(root),
        stdout=set_stdout,
    )
    get_stdout = StringIO()
    get_exit = run_config_command(
        ("config", "get", "persistence.path"),
        project=discover_project(root),
        stdout=get_stdout,
    )

    set_envelope = json.loads(set_stdout.getvalue())
    get_envelope = json.loads(get_stdout.getvalue())
    schema = _config_schema(root)
    assert set_exit == 0
    assert set_envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert get_exit == 0
    assert get_envelope["payload"] == {
        "path": "persistence.path",
        "value": ".fervis/qa-updated.sqlite3",
    }
    assert schema["environments"]["local"]["persistence"]["path"] == (
        ".fervis/fervis.sqlite3"
    )
    assert schema["environments"]["qa-branch"]["persistence"]["path"] == (
        ".fervis/qa-updated.sqlite3"
    )


def test_fervis_config_set_uses_discovered_config_path(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    custom_path = root / "custom" / "fervis.json"
    custom_path.parent.mkdir()
    custom_path.write_text(
        json.dumps(_valid_schema("django"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = []\n"
        'FERVIS_CONFIG_PATH = "custom/fervis.json"\n'
        "ROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )

    exit_code = run_config_command(
        ("config", "set", "host.organization_name", "Custom Shop"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert not (root / "config" / "fervis.json").exists()
    assert (
        json.loads(custom_path.read_text(encoding="utf-8"))["host"]["organization_name"]
        == "Custom Shop"
    )


def test_fervis_config_set_blocks_model_policy_scalar_edits(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_config_command(
        ("config", "set", "environments.local.models.default.model_key", "gpt-5.4"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": (
                "Model policy must be changed with "
                f"`{render_command(commands.model_allow(Placeholder('model-ref')))}` "
                "or "
                f"`{render_command(commands.model_use(Placeholder('model-ref')))}`."
            ),
        }
    ]
    assert _config_schema(root)["models"] == {
        "providers": [{"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}],
    }


def test_fervis_config_upgrade_reports_noop_for_current_schema(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_config_command(
        ("config", "upgrade"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["command"] == "config.upgrade"
    assert envelope["payload"] == {
        "changed_files": [],
        "skipped_existing": ["config/fervis.json"],
        "blocked_edits": [],
        "upgraded_configs": [],
    }


def test_fervis_config_upgrade_blocks_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    schema = _config_schema(root)
    schema["schema_version"] = "v99.0"
    _write_schema_config(root, schema)
    stdout = StringIO()

    exit_code = run_config_command(
        ("config", "upgrade"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["command"] == "config.upgrade"
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": "Fervis config schema_version must be v0.1.",
        }
    ]


def test_fervis_models_lists_strict_tool_certified_providers(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_models_command(
        ("models",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["command"] == "models"
    assert envelope["payload_schema"] == "fervis-models-view.v0.1"
    assert envelope["payload"]["active_model"] == "openai:gpt-5.4-mini"
    assert envelope["payload"]["providers"] == [
        {
            "name": "anthropic",
            "transport": "anthropic_messages",
            "default_model": "claude-haiku-4-5-20251001",
            "default_model_ref": "anthropic:claude-haiku-4-5-20251001",
            "strict_tools": True,
            "use_command": ("fervis model use anthropic:claude-haiku-4-5-20251001"),
        },
        {
            "name": "baseten",
            "transport": "openai_chat_completions",
            "default_model": "deepseek-ai/DeepSeek-V4-Pro",
            "default_model_ref": "baseten:deepseek-ai/DeepSeek-V4-Pro",
            "strict_tools": True,
            "use_command": ("fervis model use baseten:deepseek-ai/DeepSeek-V4-Pro"),
        },
        {
            "name": "fireworks",
            "transport": "openai_chat_completions",
            "default_model": "accounts/fireworks/models/kimi-k2-instruct-0905",
            "default_model_ref": (
                "fireworks:accounts/fireworks/models/kimi-k2-instruct-0905"
            ),
            "strict_tools": True,
            "use_command": (
                "fervis model use "
                "fireworks:accounts/fireworks/models/kimi-k2-instruct-0905"
            ),
        },
        {
            "name": "openai",
            "transport": "openai_chat_completions",
            "default_model": "gpt-5.4-mini",
            "default_model_ref": "openai:gpt-5.4-mini",
            "strict_tools": True,
            "use_command": "fervis model use openai:gpt-5.4-mini",
        },
        {
            "name": "opencode",
            "transport": "openai_chat_completions",
            "default_model": "deepseek-v4-pro",
            "default_model_ref": "opencode:deepseek-v4-pro",
            "strict_tools": True,
            "use_command": "fervis model use opencode:deepseek-v4-pro",
        },
    ]


def test_fervis_model_use_updates_default_model_and_provider_declaration(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    allow_exit_code = run_models_command(
        ("model", "allow", "baseten:deepseek-ai/DeepSeek-V4-Pro"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    exit_code = run_models_command(
        ("model", "use", "baseten:deepseek-ai/DeepSeek-V4-Pro"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    schema = _config_schema(root)
    assert allow_exit_code == 0
    assert exit_code == 0
    assert envelope["command"] == "model.use"
    assert envelope["payload_schema"] == "fervis-config-edit-result.v0.1"
    assert envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert schema["environments"]["local"]["models"] == {
        "default": {
            "provider": "baseten",
            "model_key": "deepseek-ai/DeepSeek-V4-Pro",
        },
    }
    assert schema["models"] == {
        "providers": [
            {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]},
            {
                "name": "baseten",
                "allowed_model_keys": ["deepseek-ai/DeepSeek-V4-Pro"],
            },
        ],
    }


def test_fervis_model_use_can_target_explicit_environment(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    schema = _config_schema(root)
    schema["environments"]["ci"] = {
        "models": {"default": {"provider": "openai", "model_key": "gpt-5.4-mini"}},
        "persistence": {"kind": "sqlite", "path": ".fervis/ci.sqlite3"},
    }
    _write_schema_config(root, schema)

    run_models_command(
        ("model", "allow", "opencode:deepseek-v4-pro"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    exit_code = run_models_command(
        ("model", "use", "opencode:deepseek-v4-pro", "--env", "ci"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    schema = _config_schema(root)
    assert exit_code == 0
    assert schema["environments"]["local"]["models"] == {
        "default": {"provider": "openai", "model_key": "gpt-5.4-mini"}
    }
    assert schema["environments"]["ci"]["models"] == {
        "default": {"provider": "opencode", "model_key": "deepseek-v4-pro"}
    }


def test_fervis_model_allow_ignores_unrelated_fervis_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    monkeypatch.setenv("FERVIS_ENV", "not-declared")

    exit_code = run_models_command(
        ("model", "allow", "opencode:deepseek-v4-pro"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["models"]["providers"] == [
        {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]},
        {"name": "opencode", "allowed_model_keys": ["deepseek-v4-pro"]},
    ]


def test_fervis_model_use_blocks_unsupported_model_ref(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_models_command(
        ("model", "use", "unknown:model"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": (
                "Unsupported Fervis provider 'unknown'. Supported: "
                "anthropic, baseten, fireworks, openai, opencode"
            ),
        }
    ]


def test_fervis_models_blocks_without_config(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    stdout = StringIO()

    exit_code = run_models_command(
        ("models",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert envelope["payload"]["error"]["code"] == "config_missing"


def test_fervis_doctor_reports_missing_active_model_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    root = _django_project(tmp_path)
    _write_valid_django_config(root)
    schema = _config_schema(root)
    schema["models"] = {
        "providers": [
            {
                "name": "fireworks",
                "allowed_model_keys": [
                    "accounts/fireworks/models/kimi-k2-instruct-0905"
                ],
            }
        ],
    }
    schema["environments"]["local"]["models"] = {
        "default": {
            "provider": "fireworks",
            "model_key": "accounts/fireworks/models/kimi-k2-instruct-0905",
        },
    }
    _write_schema_config(root, schema)
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["model.active_api_key"]["status"] == "failed"
    assert checks["model.active_api_key"]["fix"] == set_env_action("FIREWORKS_API_KEY")


def test_fervis_doctor_reports_missing_delegated_credential_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("FERVIS_READ_CREDENTIAL_KEY", raising=False)
    root = _fastapi_project(tmp_path)
    _write_valid_fastapi_config(root)
    _write_fastapi_auth_helpers(root)
    _write_fastapi_auth_config(
        root,
        credentials={
            "source": "captured_request_headers",
            "headers": ["Authorization", "API-TOKEN"],
            "ttl_seconds": 900,
            "encryption_key_env": "FERVIS_READ_CREDENTIAL_KEY",
        },
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["auth.delegated_credential_key"]["status"] == "failed"
    assert checks["auth.delegated_credential_key"]["fix"] == set_env_action(
        "FERVIS_READ_CREDENTIAL_KEY"
    )


def test_fervis_doctor_does_not_require_credential_key_without_header_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("FERVIS_READ_CREDENTIAL_KEY", raising=False)
    root = _fastapi_project(tmp_path)
    _write_valid_fastapi_config(root)
    _write_fastapi_auth_helpers(root)
    _write_fastapi_auth_config(root)
    stdout = StringIO()

    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert "auth.delegated_credential_key" not in checks


def test_fervis_doctor_reports_missing_host_dependency_for_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _fastapi_project(tmp_path)
    _write_valid_fastapi_config(root)

    def raise_missing_dependency(*args, **kwargs):
        del args, kwargs
        raise ModuleNotFoundError(
            "No module named 'sentry_sdk'",
            name="sentry_sdk",
        )

    monkeypatch.setattr(
        "fervis.host_api.adapters.fastapi.catalog.get_fastapi_endpoint_contracts",
        raise_missing_dependency,
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["source.catalog"]["status"] == "failed"
    assert checks["source.catalog"]["fix"] == install_dependencies_action("sentry_sdk")


def test_fervis_doctor_reports_config_fix_for_missing_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _fastapi_project(tmp_path)
    _write_valid_fastapi_config(root)
    schema = _config_schema(root)
    schema["sources"][0]["import_paths"] = ["missing_app.main:app"]
    _write_schema_config(root, schema)

    def raise_missing_source(*args, **kwargs):
        del args, kwargs
        raise ModuleNotFoundError(
            "No module named 'missing_app'",
            name="missing_app",
        )

    monkeypatch.setattr(
        "fervis.host_api.adapters.fastapi.catalog.get_fastapi_endpoint_contracts",
        raise_missing_source,
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["source.catalog"]["status"] == "failed"
    assert checks["source.catalog"]["fix"] == edit_config_action()


def test_fervis_doctor_reports_migrate_fix_when_fastapi_catalog_needs_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _fastapi_project(tmp_path)
    _write_valid_fastapi_config(root)

    def raise_persistence_not_ready(*args, **kwargs):
        del args, kwargs
        from fervis.storage.sql.engine import FervisPersistenceNotReady

        raise FervisPersistenceNotReady(
            check_id="persistence.migrations",
            message="Fervis migrations are not applied.",
        )

    monkeypatch.setattr(
        "fervis.host_api.adapters.fastapi.catalog.get_fastapi_endpoint_contracts",
        raise_persistence_not_ready,
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["source.catalog"]["status"] == "failed"
    assert checks["source.catalog"]["fix"] == run_migrate_action()


def test_fervis_sources_add_django_app_patches_generated_sources(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_sources_command(
        (
            "sources",
            "add",
            "django-app",
            "commerce",
            "--app-modules",
            "apps.sales,apps.inventory",
            "--path-prefixes",
            "/api/v1/",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    schema = _config_schema(root)
    assert exit_code == 0
    assert envelope["command"] == "sources.add"
    assert envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert {
        "kind": "django_app",
        "name": "commerce",
        "app_modules": ["apps.sales", "apps.inventory"],
        "path_prefixes": ["/api/v1/"],
    } in schema["sources"]


def test_fervis_sources_add_fastapi_app_uses_import_paths(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_sources_command(
        (
            "sources",
            "add",
            "fastapi-app",
            "commerce",
            "--import-paths",
            "app.sales.main:app,app.inventory.main:app",
            "--path-prefixes",
            "/api/v1/",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    schema = _config_schema(root)
    config_text = (root / "config" / "fervis.json").read_text(encoding="utf-8")
    assert exit_code == 0
    assert envelope["command"] == "sources.add"
    assert envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert {
        "kind": "fastapi_app",
        "name": "commerce",
        "import_paths": [
            "app.sales.main:app",
            "app.inventory.main:app",
        ],
        "path_prefixes": ["/api/v1/"],
    } in schema["sources"]
    assert "router_imports" not in config_text


def test_fervis_sources_add_flask_app_uses_app_target_and_source_prefixes(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--app",
            "app:app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_sources_command(
        (
            "sources",
            "add",
            "flask-app",
            "crm",
            "--app",
            "server:connex_app",
            "--source-prefix",
            "/api/v1/",
            "--blueprint",
            "people",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    schema = _config_schema(root)
    assert exit_code == 0
    assert envelope["command"] == "sources.add"
    assert envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert {
        "kind": "flask_app",
        "name": "crm",
        "app": "server:connex_app",
        "app_args": [],
        "app_kwargs": {},
        "path_prefixes": ["/api/v1/"],
        "blueprints": ["people"],
    } in schema["sources"]


def test_fervis_django_integration_urls_are_importable() -> None:
    from fervis import (
        FervisConfig,
        HostConfig,
        ModelConfig,
        ProviderConfig,
        RuntimeRoutes,
    )
    from fervis.django import DjangoIntegration

    integration = DjangoIntegration(
        config=FervisConfig(
            host=HostConfig(timezone="UTC"),
            routes=RuntimeRoutes(prefix="/fervis/"),
            model=ModelConfig(
                default_provider="openai",
                default_model_key="gpt-5.4-mini",
                providers=[
                    ProviderConfig(
                        name="openai",
                        allowed_model_keys=["gpt-5.4-mini"],
                    )
                ],
            ),
            sources=[],
        )
    )

    urls = importlib.import_module(integration.urls)

    assert urls.urlpatterns


def test_fervis_fastapi_integration_mounts_router() -> None:
    from fastapi import FastAPI
    from fervis import (
        FervisConfig,
        HostConfig,
        ModelConfig,
        ProviderConfig,
        RuntimeRoutes,
    )
    from fervis.fastapi import FastAPIIntegration

    integration = FastAPIIntegration(
        config=FervisConfig(
            host=HostConfig(timezone="UTC"),
            routes=RuntimeRoutes(prefix="/fervis/"),
            model=ModelConfig(
                default_provider="openai",
                default_model_key="gpt-5.4-mini",
                providers=[
                    ProviderConfig(
                        name="openai",
                        allowed_model_keys=["gpt-5.4-mini"],
                    )
                ],
            ),
            sources=[],
        )
    )
    app = FastAPI()

    assert integration.mount(app, question_interface=object()) is app
    assert "/fervis/" not in set(app.openapi()["paths"])


def test_configured_fervis_fastapi_mount_uses_configured_question_interface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from fervis.interfaces.common.questions import QuestionInterfaceResponse
    from fervis.project import configuration
    from fervis import configured_fervis

    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    _configure_fastapi_auth(root)
    (root / "app" / "api" / "deps.py").write_text(
        "from app.users import User\n\n"
        "def get_current_user():\n"
        "    return User('user-1')\n",
        encoding="utf-8",
    )
    calls = []

    class FakeQuestionInterface:
        def create_question(self, payload, *, principal, idempotency_key=None):
            calls.append(
                {
                    "project": str(root),
                    "payload": payload,
                    "principal_id": principal.principal_id,
                    "tenant_id": principal.tenant_id,
                    "idempotency_key": idempotency_key,
                }
            )
            return QuestionInterfaceResponse(
                status_code=202,
                payload={
                    "questionId": "question-1",
                    "latestRunId": "run-1",
                    "status": "RUNNING",
                },
            )

        def get_question(self, question_id, *, principal):
            return QuestionInterfaceResponse(
                status_code=200,
                payload={"questionId": question_id, "status": "RUNNING"},
            )

        def list_question_runs(self, question_id, *, principal):
            return QuestionInterfaceResponse(
                status_code=200,
                payload={"questionId": question_id, "runs": []},
            )

        def get_question_run(self, question_id, run_id, *, principal):
            return QuestionInterfaceResponse(
                status_code=200,
                payload={"questionId": question_id, "runId": run_id},
            )

    def fake_question_interface(*, project, loaded_config):
        assert project.root_path == root
        assert loaded_config.config_path == Path("config/fervis.json")
        return FakeQuestionInterface()

    monkeypatch.setattr(
        configuration,
        "_fastapi_question_interface",
        fake_question_interface,
    )
    monkeypatch.chdir(root)

    app = FastAPI()
    configured_fervis().mount(app)

    response = TestClient(app).post(
        "/fervis/questions/",
        json={"question": "How many orders?"},
        headers={
            "Idempotency-Key": "idem-1",
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "questionId": "question-1",
        "latestRunId": "run-1",
        "status": "RUNNING",
    }
    assert calls == [
        {
            "project": str(root),
            "payload": {"question": "How many orders?"},
            "principal_id": "user-1",
            "tenant_id": "default",
            "idempotency_key": "idem-1",
        }
    ]


def test_fervis_init_is_idempotent(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert envelope["payload"]["changed_files"] == []
    assert envelope["payload"]["skipped_existing"] == [
        "config/fervis.json",
        "config/settings.py",
        "config/urls.py",
    ]


def test_fervis_init_patches_fastapi_mount(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["project"] == {
        "framework": "fastapi",
        "config_path": "config/fervis.json",
    }
    assert envelope["payload"]["changed_files"] == [
        "config/fervis.json",
        "app/main.py",
    ]
    assert envelope["payload"]["blocked_edits"] == []
    main_py = (root / "app" / "main.py").read_text(encoding="utf-8")
    assert "from fervis import configured_fervis" in main_py
    assert "configured_fervis().mount(app)" in main_py
    assert _config_schema(root)["sources"] == [
        {
            "kind": "fastapi_app",
            "name": "default",
            "import_paths": ["app.main:app"],
            "path_prefixes": ["/health/"],
        }
    ]
    assert _config_schema(root)["environments"]["local"]["persistence"] == {
        "kind": "sqlite",
        "path": ".fervis/fervis.sqlite3",
    }


def test_fervis_init_mounts_after_module_level_route_registration(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import APIRouter, FastAPI\n\n"
        "app = FastAPI()\n"
        "router = APIRouter()\n\n"
        "@router.get('/records/')\n"
        "def list_records():\n"
        "    return []\n\n"
        "app.include_router(router)\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--path-prefixes",
            "/records/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    updated = main_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert updated.index("app.include_router(router)") < updated.index(
        "configured_fervis().mount(app)"
    )


def test_fervis_init_derives_fastapi_source_prefixes_from_runtime_app_routes(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/users/')\n"
        "def list_users():\n"
        "    return []\n\n"
        "@app.get('/items/{item_id}')\n"
        "def get_item(item_id: str):\n"
        "    return {'id': item_id}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["path_prefixes"] == ["/items/", "/users/"]


def test_fervis_init_derives_fastapi_source_prefixes_from_runtime_router_includes(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    routers = root / "app" / "routers"
    routers.mkdir()
    (routers / "__init__.py").write_text("", encoding="utf-8")
    (routers / "users.py").write_text(
        "from fastapi import APIRouter\n\n"
        "router = APIRouter(prefix='/users')\n\n"
        "@router.get('/')\n"
        "def list_users():\n"
        "    return []\n",
        encoding="utf-8",
    )
    (routers / "items.py").write_text(
        "from fastapi import APIRouter\n\n"
        "router = APIRouter(prefix='/items')\n\n"
        "@router.get('/{item_id}')\n"
        "def get_item(item_id: str):\n"
        "    return {'id': item_id}\n",
        encoding="utf-8",
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from .routers import items, users\n\n"
        "app = FastAPI()\n"
        "app.include_router(users.router)\n"
        "app.include_router(items.router)\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["path_prefixes"] == ["/items/", "/users/"]


def test_fervis_init_derives_fastapi_source_prefixes_from_runtime_settings(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    api_dir = root / "app" / "api"
    routes_dir = api_dir / "routes"
    core_dir = root / "app" / "core"
    routes_dir.mkdir(parents=True)
    core_dir.mkdir()
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (routes_dir / "__init__.py").write_text("", encoding="utf-8")
    (core_dir / "__init__.py").write_text("", encoding="utf-8")
    (core_dir / "config.py").write_text(
        "class Settings:\n    API_V1_STR: str = '/api/v1'\n\nsettings = Settings()\n",
        encoding="utf-8",
    )
    (routes_dir / "items.py").write_text(
        "from fastapi import APIRouter\n\n"
        "router = APIRouter(prefix='/items')\n\n"
        "@router.get('/')\n"
        "def list_items():\n"
        "    return []\n",
        encoding="utf-8",
    )
    (api_dir / "main.py").write_text(
        "from fastapi import APIRouter\n"
        "from app.api.routes import items\n\n"
        "api_router = APIRouter()\n"
        "api_router.include_router(items.router)\n",
        encoding="utf-8",
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from app.api.main import api_router\n"
        "from app.core.config import settings\n\n"
        "app = FastAPI()\n"
        "app.include_router(api_router, prefix=settings.API_V1_STR)\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["path_prefixes"] == ["/api/v1/items/"]


def test_fervis_init_patches_fastapi_factory_app(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "\n"
        "    @app.get('/orders/')\n"
        "    def list_orders():\n"
        "        return []\n"
        "\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "app.main:create_app",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = main_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "from fervis import configured_fervis" in text
    assert "    configured_fervis().mount(app)\n    return app" in text
    assert _config_schema(root)["sources"] == [
        {
            "kind": "fastapi_app",
            "name": "default",
            "import_paths": ["app.main:create_app"],
            "path_prefixes": ["/orders/"],
        }
    ]


def test_fervis_init_derives_factory_source_prefixes_from_runtime_openapi(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    api_path = root / "app" / "api.py"
    api_path.write_text(
        "from fastapi import APIRouter\n\n"
        "router = APIRouter(prefix='/orders')\n\n"
        "@router.get('/')\n"
        "def list_orders():\n"
        "    return []\n",
        encoding="utf-8",
    )
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n"
        "from app.api import router\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    app.include_router(router, prefix='/api')\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "app.main:create_app",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0] == {
        "kind": "fastapi_app",
        "name": "default",
        "import_paths": ["app.main:create_app"],
        "path_prefixes": ["/api/orders/"],
    }


def test_fervis_doctor_accepts_fastapi_factory_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n\n"
        "class Order(BaseModel):\n"
        "    id: str\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "\n"
        "    @app.get('/orders/', response_model=list[Order])\n"
        "    def list_orders():\n"
        "        return []\n"
        "\n"
        "    return app\n",
        encoding="utf-8",
    )
    run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "app.main:create_app",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "passed"
    assert checks["source.fastapi.entrypoint"]["status"] == "passed"
    assert checks["auth.config"]["status"] == "failed"


def test_fervis_doctor_does_not_discover_static_entrypoint_for_configured_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _fastapi_project(tmp_path)
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
        encoding="utf-8",
    )
    package = root / "service"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "factory.py").write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n"
        "from pydantic import BaseModel\n\n"
        "class Order(BaseModel):\n"
        "    id: str\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    configured_fervis().mount(app)\n"
        "\n"
        "    @app.get('/orders/', response_model=list[Order])\n"
        "    def list_orders():\n"
        "        return []\n"
        "\n"
        "    return app\n",
        encoding="utf-8",
    )
    schema = _valid_schema("fastapi")
    schema["sources"][0]["import_paths"] = ["service.factory:create_app"]
    schema["sources"][0]["path_prefixes"] = ["/orders/"]
    _write_schema_config(root, schema)
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )

    def fail_static_entrypoint_scan(*args, **kwargs):
        del args, kwargs
        raise AssertionError("doctor should validate configured factory directly")

    monkeypatch.setattr(
        "fervis.project.mounting.fastapi.fastapi_entrypoint",
        fail_static_entrypoint_scan,
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "passed"
    assert checks["source.fastapi.entrypoint"]["status"] == "passed"
    assert checks["source.catalog"]["status"] == "passed"
    assert checks["auth.config"]["status"] == "failed"


def test_fervis_doctor_validates_fastapi_entrypoint_independent_of_source_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n"
        "from pydantic import BaseModel\n\n"
            "class Order(BaseModel):\n"
            "    id: str\n\n"
            "app = FastAPI()\n"
            "@app.get('/orders/', response_model=list[Order])\n"
            "def list_orders():\n"
            "    return []\n\n"
            "configured_fervis().mount(app)\n",
        encoding="utf-8",
    )
    schema = _valid_schema("fastapi")
    schema["sources"][0]["path_prefixes"] = ["/api/"]
    _write_schema_config(root, schema)
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "passed"
    assert checks["source.fastapi.entrypoint"]["status"] == "passed"
    assert checks["source.catalog"]["status"] == "failed"


def test_fervis_init_accepts_explicit_fastapi_factory_path_prefixes(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n\n"
        "def include_routes(app):\n"
        "    return app\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    include_routes(app)\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "app.main:create_app",
            "--path-prefixes",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0] == {
        "kind": "fastapi_app",
        "name": "default",
        "import_paths": ["app.main:create_app"],
        "path_prefixes": ["/api/"],
    }


def test_fervis_init_accepts_explicit_fastapi_app_path_prefixes_without_import(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n"
        "raise RuntimeError('init must not import app when prefixes are explicit')\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--path-prefixes",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0] == {
        "kind": "fastapi_app",
        "name": "default",
        "import_paths": ["app.main:app"],
        "path_prefixes": ["/api/"],
    }
    assert "configured_fervis().mount(app)" in main_path.read_text(encoding="utf-8")


def test_fervis_init_resolves_fastapi_factory_from_hatch_package_source_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "langflow_style"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'langflow-style'\n"
        "dependencies = ['fastapi']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "packages = ['src/backend/base/langflow']\n",
        encoding="utf-8",
    )
    package_dir = root / "src" / "backend" / "base" / "langflow"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "langflow.main:create_app",
            "--path-prefixes",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0] == {
        "kind": "fastapi_app",
        "name": "default",
        "import_paths": ["langflow.main:create_app"],
        "path_prefixes": ["/api/"],
    }
    assert "configured_fervis().mount(app)" in (package_dir / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_uses_explicit_fastapi_factory_when_framework_not_detected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace_app"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'workspace-app'\n"
        "dependencies = ['workspace-api']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "packages = ['src/backend/base/workspace_api']\n",
        encoding="utf-8",
    )
    package_dir = root / "src" / "backend" / "base" / "workspace_api"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    return app\n",
        encoding="utf-8",
    )

    initial_project = discover_project(root)
    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "workspace_api.main:create_app",
            "--path-prefixes",
            "/api/",
            "--yes",
        ),
        project=initial_project,
        stdout=StringIO(),
    )

    discovered = discover_project(root)
    assert initial_project.framework == "unknown"
    assert exit_code == 0
    assert discovered.framework == "fastapi"
    assert discovered.config_path == Path("config/fervis.json")
    assert _config_schema(root)["sources"][0]["import_paths"] == [
        "workspace_api.main:create_app"
    ]


def test_fervis_init_resolves_fastapi_factory_from_uv_workspace_member(
    tmp_path: Path,
) -> None:
    root = tmp_path / "uv_workspace"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'workspace-root'\n"
        "dependencies = ['workspace-api']\n\n"
        "[tool.uv.workspace]\n"
        "members = ['src/backend/base']\n",
        encoding="utf-8",
    )
    member_dir = root / "src" / "backend" / "base"
    member_dir.mkdir(parents=True)
    (member_dir / "pyproject.toml").write_text(
        "[project]\nname = 'workspace-api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = member_dir / "workspace_api"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "workspace_api.main:create_app",
            "--path-prefixes",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert "configured_fervis().mount(app)" in (package_dir / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_existing_config_does_not_execute_fastapi_factory(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    _write_schema_config(
        root,
        {
            "schema_version": "v0.1",
            "framework": "fastapi",
            "default_environment": "local",
            "host": {
                "organization_name": "",
                "about_api": "",
                "timezone": "UTC",
            },
            "routes": {"prefix": "/fervis"},
            "models": {
                "providers": [
                    {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]},
                ],
            },
            "sources": [
                {
                    "kind": "fastapi_app",
                    "name": "default",
                    "import_paths": ["app.main:create_app"],
                    "path_prefixes": ["/api/"],
                }
            ],
            "environments": {
                "local": {
                    "models": {
                        "default": {
                            "provider": "openai",
                            "model_key": "gpt-5.4-mini",
                        }
                    },
                    "persistence": {
                        "kind": "sqlite",
                        "path": ".fervis/fervis.sqlite3",
                    },
                }
            },
        },
    )
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n\n"
        "def create_app():\n"
        "    app = FastAPI()\n"
        "    raise RuntimeError('init must not execute host app factories')\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "fastapi",
            "--app-factory",
            "app.main:create_app",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert "configured_fervis().mount(app)" in main_path.read_text(encoding="utf-8")


def test_fervis_init_config_uses_root_fastapi_entrypoint(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    app_dir = root / "app"
    for path in app_dir.iterdir():
        path.unlink()
    app_dir.rmdir()
    (root / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["import_paths"] == ["main:app"]
    assert "configured_fervis().mount(app)" in (root / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_uses_declared_fastapi_source_root(tmp_path: Path) -> None:
    root = tmp_path / "open-webui"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'open-webui'\n"
        "dependencies = ['fastapi']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "sources = ['backend']\n",
        encoding="utf-8",
    )
    package_dir = root / "backend" / "open_webui"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI(title='Open WebUI')\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    main_text = (package_dir / "main.py").read_text(encoding="utf-8")
    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["import_paths"] == ["open_webui.main:app"]
    assert "configured_fervis().mount(app)" in main_text


def test_fervis_init_uses_default_fastapi_src_layout(tmp_path: Path) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = root / "src" / "service"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["import_paths"] == ["service.main:app"]
    assert "configured_fervis().mount(app)" in (package_dir / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_fastapi_source_root_ignores_project_ancestor_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tests" / "api"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'api'\n"
        "dependencies = ['fastapi']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "sources = ['backend']\n",
        encoding="utf-8",
    )
    package_dir = root / "backend" / "service"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert "configured_fervis().mount(app)" in (package_dir / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_fastapi_source_root_ignores_unrelated_broken_python(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'api'\n"
        "dependencies = ['fastapi']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "sources = ['backend']\n",
        encoding="utf-8",
    )
    package_dir = root / "backend" / "service"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )
    scratch_dir = root / "backend" / "scratch"
    scratch_dir.mkdir()
    (scratch_dir / "broken.py").write_text("this is not python", encoding="utf-8")

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert "configured_fervis().mount(app)" in (package_dir / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_uses_declared_fastapi_source_root_with_placeholder_main(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'api'\n"
        "dependencies = ['fastapi']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "sources = ['backend']\n",
        encoding="utf-8",
    )
    (root / "main.py").write_text("print('maintenance entrypoint')\n", encoding="utf-8")
    package_dir = root / "backend" / "service"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert _config_schema(root)["sources"][0]["import_paths"] == ["service.main:app"]
    assert "configured_fervis().mount(app)" in (package_dir / "main.py").read_text(
        encoding="utf-8"
    )
    assert "configured_fervis().mount(app)" not in (root / "main.py").read_text(
        encoding="utf-8"
    )


def test_fervis_init_patches_typed_fastapi_app_assignment(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "from fastapi import FastAPI\n\n"
        "app: FastAPI = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = main_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "from fervis import configured_fervis" in text
    assert "configured_fervis().mount(app)" in text


def test_fervis_init_suppresses_host_source_syntax_warnings(
    tmp_path: Path,
    capsys,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        'from fastapi import FastAPI\n\nPATTERN = "\\d+"\n'
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_fervis_init_preserves_existing_multiline_django_urlpatterns(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.urls import path\n\n"
        "urlpatterns = [\n"
        '    path("health/", health_view),\n'
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "from django.urls import path, include" in text
    assert "from fervis import configured_fervis" in text
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in text
    )
    assert 'path("health/", health_view),' in text


def test_fervis_init_patches_django_urlpatterns_without_existing_trailing_comma(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.urls import path\n\n"
        "urlpatterns = [\n"
        '    path("health/", health_view)\n'
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert 'path("health/", health_view),' in text
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in text
    )


def test_fervis_init_patches_django_installed_apps_without_existing_trailing_comma(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    settings_path.write_text(
        "INSTALLED_APPS = [\n"
        '    "django.contrib.auth"\n'
        "]\n"
        "ROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = settings_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert '"django.contrib.auth",' in text
    assert '"rest_framework",' in text
    assert '"fervis.django",' in text


def test_fervis_init_patches_django_installed_apps_with_starred_extension(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    settings_path.write_text(
        "extra_apps = load_extra_apps()\n"
        "INSTALLED_APPS = [\n"
        '    "django.contrib.auth",\n'
        "    *extra_apps,\n"
        "]\n"
        "ROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = settings_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "    *extra_apps,\n" in text
    assert '"rest_framework",' in text
    assert '"fervis.django",' in text


def test_fervis_init_blocks_nonempty_single_line_urlpatterns(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    original = (
        'from django.urls import path\n\nurlpatterns = [path("health/", health_view)]\n'
    )
    urls_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "config/urls.py",
        "reason": "Non-empty single-line urlpatterns must be edited manually.",
    } in envelope["payload"]["blocked_edits"]
    assert urls_path.read_text(encoding="utf-8") == original


def test_fervis_init_does_not_partially_patch_django_hooks_when_blocked(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    urls_path = root / "config" / "urls.py"
    original_settings = settings_path.read_text(encoding="utf-8")
    urls_path.write_text(
        "from django.urls import path\n\n"
        'urlpatterns = [path("health/", health_view)]\n',
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 2
    assert settings_path.read_text(encoding="utf-8") == original_settings


def test_fervis_init_does_not_treat_commented_django_mount_as_installed(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.urls import path\n\n"
        "# path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "urlpatterns = []\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert (
        text.count(
            "path(configured_fervis().routes.django_path, include(configured_fervis().urls)),"
        )
        == 2
    )
    assert "from fervis import configured_fervis" in text


def test_fervis_init_blocks_dynamic_installed_apps_without_rewriting_settings(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    original = (
        'BASE_APPS = ["django.contrib.auth"]\n'
        "INSTALLED_APPS = BASE_APPS + []\n"
        "ROOT_URLCONF = 'config.urls'\n"
    )
    settings_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "config/settings.py",
        "reason": (
            "INSTALLED_APPS must be a literal list or tuple before Fervis can patch it."
        ),
    } in envelope["payload"]["blocked_edits"]
    assert settings_path.read_text(encoding="utf-8") == original


def test_fervis_init_allows_safe_installed_apps_mutations(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    settings_path.write_text(
        "INSTALLED_APPS = [\n"
        '    "django.contrib.auth",\n'
        "]\n"
        "INSTALLED_APPS.append('local.app')\n"
        "ROOT_URLCONF = 'config.urls'\n"
        "INSTALLED_APPS.remove('debug_toolbar')\n",
        encoding="utf-8",
    )
    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = settings_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert '"rest_framework",' in text
    assert '"fervis.django",' in text
    assert "INSTALLED_APPS.append('local.app')" in text
    assert "INSTALLED_APPS.remove('debug_toolbar')" in text


def test_fervis_init_blocks_destructive_installed_apps_reassignment(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    original = (
        "INSTALLED_APPS = [\n"
        '    "django.contrib.auth",\n'
        "]\n"
        "INSTALLED_APPS = []\n"
        "ROOT_URLCONF = 'config.urls'\n"
    )
    settings_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "config/settings.py",
        "reason": (
            "INSTALLED_APPS is overwritten or may remove Fervis after assignment; "
            "mount manually."
        ),
    } in envelope["payload"]["blocked_edits"]
    assert settings_path.read_text(encoding="utf-8") == original


def test_fervis_init_blocks_conflicting_fervis_config_setting(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    original = (
        "INSTALLED_APPS = []\n"
        'FERVIS_CONFIG_PATH = "custom/fervis.json"\n'
        "ROOT_URLCONF = 'config.urls'\n"
    )
    settings_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "config/settings.py",
        "reason": "FERVIS_CONFIG_PATH already exists with a different value.",
    } in envelope["payload"]["blocked_edits"]


def test_fervis_init_blocks_annotated_conflicting_fervis_config_setting(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    original = (
        "INSTALLED_APPS = []\n"
        'FERVIS_CONFIG_PATH: str = "custom/fervis.json"\n'
        "ROOT_URLCONF = 'config.urls'\n"
    )
    settings_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "config/settings.py",
        "reason": "FERVIS_CONFIG_PATH already exists with a different value.",
    } in envelope["payload"]["blocked_edits"]
    assert settings_path.read_text(encoding="utf-8") == original


def test_fervis_init_patches_active_django_root_urlconf(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    settings_path.write_text(
        "INSTALLED_APPS = []\nROOT_URLCONF = 'shop.urls'\n",
        encoding="utf-8",
    )
    shop_dir = root / "shop"
    shop_dir.mkdir()
    (shop_dir / "__init__.py").write_text("", encoding="utf-8")
    (shop_dir / "urls.py").write_text(
        "from django.urls import path\n\nurlpatterns = []\n",
        encoding="utf-8",
    )
    original_config_urls = (root / "config" / "urls.py").read_text(encoding="utf-8")

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in (shop_dir / "urls.py").read_text(encoding="utf-8")
    )
    assert (root / "config" / "urls.py").read_text(
        encoding="utf-8"
    ) == original_config_urls


def test_fervis_init_patches_active_django_settings_module(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    (root / "manage.py").write_text(
        "import os\nos.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shop.settings')\n",
        encoding="utf-8",
    )
    shop_dir = root / "shop"
    shop_dir.mkdir()
    (shop_dir / "__init__.py").write_text("", encoding="utf-8")
    (shop_dir / "settings.py").write_text(
        "INSTALLED_APPS = []\nROOT_URLCONF = 'shop.urls'\n",
        encoding="utf-8",
    )
    (shop_dir / "urls.py").write_text(
        "from django.urls import path\n\nurlpatterns = []\n",
        encoding="utf-8",
    )
    original_config_settings = (root / "config" / "settings.py").read_text(
        encoding="utf-8"
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert '"fervis.django",' in (shop_dir / "settings.py").read_text(encoding="utf-8")
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in (shop_dir / "urls.py").read_text(encoding="utf-8")
    )
    assert (root / "config" / "settings.py").read_text(
        encoding="utf-8"
    ) == original_config_settings


def test_fervis_init_patches_package_django_settings_module(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    (root / "manage.py").write_text(
        "import os\nos.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shop.settings')\n",
        encoding="utf-8",
    )
    shop_dir = root / "shop"
    settings_dir = shop_dir / "settings"
    settings_dir.mkdir(parents=True)
    (shop_dir / "__init__.py").write_text("", encoding="utf-8")
    (settings_dir / "__init__.py").write_text(
        "INSTALLED_APPS = []\nROOT_URLCONF = 'shop.urls'\n",
        encoding="utf-8",
    )
    (shop_dir / "urls.py").write_text(
        "from django.urls import path\n\nurlpatterns = []\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert '"fervis.django",' in (settings_dir / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in (shop_dir / "urls.py").read_text(encoding="utf-8")
    )


def test_fervis_init_blocks_ambiguous_django_settings_module(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    (root / "manage.py").write_text(
        "import os\n"
        "if TESTING:\n"
        "    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.settings')\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "manage.py",
            "reason": (
                "manage.py must declare exactly one literal DJANGO_SETTINGS_MODULE "
                "so Fervis can patch the active settings file."
            ),
        }
    ]


def test_fervis_init_reports_blocked_active_django_urlconf_path(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = []\nROOT_URLCONF = 'shop.urls'\n",
        encoding="utf-8",
    )
    shop_dir = root / "shop"
    shop_dir.mkdir()
    (shop_dir / "__init__.py").write_text("", encoding="utf-8")
    (shop_dir / "urls.py").write_text(
        "from django.urls import *\n\nurlpatterns = []\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "shop/urls.py",
        "reason": "Aliased, relative, or wildcard django.urls imports must be edited manually.",
    } in envelope["payload"]["blocked_edits"]


def test_fervis_init_accepts_django_conf_urls_include(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.conf.urls import include\n"
        "from django.urls import path\n\n"
        "urlpatterns = [\n"
        "    path('api/', include('api.urls')),\n"
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "from django.conf.urls import include" in text
    assert "from django.urls import path" in text
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in text
    )


def test_fervis_init_accepts_split_django_url_imports(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.conf.urls import include\n"
        "from django.urls import path\n"
        "from django.urls import re_path\n\n"
        "urlpatterns = [\n"
        "    re_path(r'^api/', include('api.urls')),\n"
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "from django.urls import path" in text
    assert "from django.urls import re_path" in text
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in text
    )


def test_fervis_init_patches_wrapped_django_urlpattern_list(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    settings_path.write_text(
        "INSTALLED_APPS = [\n"
        '    "django.contrib.auth",\n'
        "]\n"
        "INSTALLED_APPS.extend(plugin_apps)\n"
        "sorted_apps = reversed(list(dict.fromkeys(reversed(INSTALLED_APPS))))\n"
        "INSTALLED_APPS = list(sorted_apps)\n"
        "ROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.conf import settings\n"
        "from django.conf.urls import include\n"
        "from django.urls import path\n\n"
        "_patterns = [\n"
        "    path('api/', include('api.urls')),\n"
        "]\n\n"
        "if settings.DEBUG:\n"
        "    _patterns.append(path('__debug__/', include(debug_toolbar.urls)))\n\n"
        "urlpatterns = [\n"
        "    path(settings.BASE_PATH, include(_patterns))\n"
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    settings_text = settings_path.read_text(encoding="utf-8")
    urls_text = urls_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert '"rest_framework",' in settings_text
    assert '"fervis.django",' in settings_text
    assert "from django.conf.urls import include" in urls_text
    assert "from django.urls import path" in urls_text
    assert "_patterns = [\n" in urls_text
    assert (
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n]\n"
        in urls_text
    )
    assert (
        "urlpatterns = [\n    path(settings.BASE_PATH, include(_patterns))\n]"
        in urls_text
    )


def test_fervis_init_patches_urlpatterns_for_non_site_wrapper(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from django.urls import include, path\n\n"
        "api_patterns = [\n"
        "    path('orders/', orders_view),\n"
        "]\n\n"
        "urlpatterns = [\n"
        "    path('api/', include(api_patterns)),\n"
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    api_patterns, urlpatterns = text.split("urlpatterns = [", 1)
    assert exit_code == 0
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        not in api_patterns
    )
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in urlpatterns
    )


def test_fervis_init_does_not_treat_local_settings_base_path_as_site_wrapper(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    urls_path.write_text(
        "from types import SimpleNamespace\n"
        "from django.urls import include, path\n\n"
        "settings = SimpleNamespace(BASE_PATH='api/')\n"
        "api_patterns = [\n"
        "    path('orders/', orders_view),\n"
        "]\n\n"
        "urlpatterns = [\n"
        "    path(settings.BASE_PATH, include(api_patterns)),\n"
        "]\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    text = urls_path.read_text(encoding="utf-8")
    api_patterns, urlpatterns = text.split("urlpatterns = [", 1)
    assert exit_code == 0
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        not in api_patterns
    )
    assert (
        "path(configured_fervis().routes.django_path, include(configured_fervis().urls))"
        in urlpatterns
    )


def test_fervis_init_blocks_django_url_name_imported_from_unexpected_module(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    urls_path = root / "config" / "urls.py"
    original = (
        "from custom.urls import include\n"
        "from django.urls import path\n\n"
        "urlpatterns = [\n"
        "    path('api/', include('api.urls')),\n"
        "]\n"
    )
    urls_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "config/urls.py",
        "reason": "`include` is already imported from another module; mount manually.",
    } in envelope["payload"]["blocked_edits"]
    assert urls_path.read_text(encoding="utf-8") == original


def test_fervis_doctor_rejects_inactive_django_urlconf_mount(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    settings_path = root / "config" / "settings.py"
    settings_path.write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        'ROOT_URLCONF = "shop.urls"\n'
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    shop_dir = root / "shop"
    shop_dir.mkdir()
    (shop_dir / "__init__.py").write_text("", encoding="utf-8")
    (shop_dir / "urls.py").write_text(
        "from django.urls import path\n\nurlpatterns = []\n",
        encoding="utf-8",
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n",
        encoding="utf-8",
    )
    _write_valid_django_config(root)
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_mutated_django_installed_apps(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "INSTALLED_APPS.remove('fervis.django')\n"
        "ROOT_URLCONF = 'config.urls'\n"
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_aliased_django_installed_apps_mutation(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "apps = INSTALLED_APPS\n"
        "apps.clear()\n"
        "ROOT_URLCONF = 'config.urls'\n"
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_destructured_django_installed_apps_alias(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "apps, = (INSTALLED_APPS,)\n"
        "apps.clear()\n"
        "ROOT_URLCONF = 'config.urls'\n"
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_dynamic_namespace_django_settings_mutation(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "globals().update(INSTALLED_APPS=[])\n"
        "ROOT_URLCONF = 'config.urls'\n"
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_django_installed_apps_subscript_write(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "INSTALLED_APPS[:] = []\n"
        "ROOT_URLCONF = 'config.urls'\n"
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_django_installed_apps_global_subscript_write(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "globals()['INSTALLED_APPS'] = []\n"
        "ROOT_URLCONF = 'config.urls'\n"
        'FERVIS_CONFIG = "config.fervis:fervis"\n',
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_mutated_django_urlpatterns(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n"
        "urlpatterns.clear()\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_django_urlpatterns_subscript_write(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n"
        "urlpatterns[:] = []\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_django_urlpatterns_global_subscript_write(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n"
        "globals()['urlpatterns'] = []\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_aliased_django_urlpatterns_mutation(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n"
        "patterns = urlpatterns\n"
        "patterns.clear()\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_destructured_django_urlpatterns_alias(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n"
        "patterns, = (urlpatterns,)\n"
        "patterns.clear()\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_shadowed_django_url_wrapper_include(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "_patterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n\n"
        "def include(value):\n"
        "    return value\n\n"
        "urlpatterns = [\n"
        "    path('', include(_patterns)),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_local_settings_base_path_wrapper(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from types import SimpleNamespace\n"
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "settings = SimpleNamespace(BASE_PATH='api/')\n"
        "api_patterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n\n"
        "urlpatterns = [\n"
        "    path(settings.BASE_PATH, include(api_patterns)),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_missing_django_fervis_config(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "settings.py").write_text(
        "INSTALLED_APPS = [\n"
        '    "rest_framework",\n'
        '    "fervis.django",\n'
        "]\n"
        "ROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.installed_apps"]["status"] == "failed"


def test_fervis_doctor_rejects_noncanonical_django_fervis_path(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls), name='wrong'),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_shadowed_django_include(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "def include(value):\n"
        "    return value\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_nested_shadowed_django_include(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "if True:\n"
        "    include = lambda value: value\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_deleted_django_include(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "del include\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_doctor_rejects_walrus_shadowed_django_include(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "from django.urls import path, include\n"
        "from fervis import configured_fervis\n\n"
        "if include := (lambda value: value):\n"
        "    pass\n\n"
        "urlpatterns = [\n"
        "    path(configured_fervis().routes.django_path, include(configured_fervis().urls)),\n"
        "]\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_init_places_fastapi_import_after_future_import(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        '"""Application entrypoint."""\n'
        "from __future__ import annotations\n"
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n"
        "OTHER = 1\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    lines = main_path.read_text(encoding="utf-8").splitlines()
    assert exit_code == 0
    assert lines.index("from fervis import configured_fervis") > lines.index(
        "from __future__ import annotations"
    )
    assert lines.index("configured_fervis().mount(app)") > lines.index(
        "def get_health():"
    )
    assert lines.index("configured_fervis().mount(app)") > lines.index("OTHER = 1")


def test_fervis_init_preserves_existing_fastapi_routes(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    original = main_path.read_text(encoding="utf-8")

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    content = main_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "@app.get('/health/', response_model=HealthResponse)" in content
    assert "def get_health() -> HealthResponse:" in content
    assert "return HealthResponse(status='ok')" in content
    assert "configured_fervis().mount(app)" in content
    assert len(content) > len(original)


def test_fervis_init_preserves_fastapi_shebang_and_encoding(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "#!/usr/bin/env python\n"
        "# -*- coding: utf-8 -*-\n"
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    lines = main_path.read_text(encoding="utf-8").splitlines()
    assert exit_code == 0
    assert lines[:2] == ["#!/usr/bin/env python", "# -*- coding: utf-8 -*-"]
    assert lines.index("from fervis import configured_fervis") == 2


def test_fervis_init_preserves_crlf_line_endings(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_bytes(
        b"from fastapi import FastAPI\r\n\r\n"
        b"app = FastAPI()\r\n\r\n"
        b"@app.get('/health/')\r\n"
        b"def get_health():\r\n"
        b"    return {'status': 'ok'}\r\n"
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    content = main_path.read_bytes()
    assert exit_code == 0
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")


def test_fervis_init_blocks_fastapi_app_reassignment_after_mount(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    original = (
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n"
        "configured_fervis().mount(app)\n"
        "app = build_application()\n"
    )
    main_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "app/main.py",
        "reason": "`app` is reassigned after `app = FastAPI(...)`; mount manually.",
    } in envelope["payload"]["blocked_edits"]
    assert main_path.read_text(encoding="utf-8") == original


def test_fervis_doctor_rejects_nested_fastapi_app_reassignment(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "configured_fervis().mount(app)\n"
        "if True:\n"
        "    app = FastAPI()\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_doctor_rejects_global_fastapi_app_reassignment(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "configured_fervis().mount(app)\n"
        "globals()['app'] = FastAPI()\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_doctor_rejects_subscript_fastapi_app_assignment(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = []\n"
        "app[0] = FastAPI()\n"
        "configured_fervis().mount(app)\n",
        encoding="utf-8",
    )
    _write_valid_fastapi_config(root)
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_doctor_rejects_deleted_fastapi_app_after_mount(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "configured_fervis().mount(app)\n"
        "del app\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_init_blocks_nested_fastapi_mount(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    original = (
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n"
        "if ENABLE_FERVIS:\n"
        "    configured_fervis().mount(app)\n"
    )
    main_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "app/main.py",
        "reason": (
            "Existing Fervis mount must be exactly one top-level "
            "`configured_fervis().mount(app)` after `app = FastAPI(...)`."
        ),
    } in envelope["payload"]["blocked_edits"]
    assert main_path.read_text(encoding="utf-8") == original


def test_fervis_doctor_rejects_unproven_fastapi_constructor(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "app" / "main.py").write_text(
        "from fervis import configured_fervis\n\n"
        "def FastAPI():\n"
        "    return object()\n\n"
        "app = FastAPI()\n"
        "configured_fervis().mount(app)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_init_blocks_existing_fastapi_mount_on_wrong_object(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    original = (
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n"
        "configured_fervis().mount(other_app)\n"
    )
    main_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "app/main.py",
        "reason": (
            "Existing Fervis mount must be exactly one top-level "
            "`configured_fervis().mount(app)` after "
            "`app = FastAPI(...)`."
        ),
    } in envelope["payload"]["blocked_edits"]
    assert main_path.read_text(encoding="utf-8") == original


def test_fervis_init_blocks_aliased_fastapi_fervis_import_without_rewriting(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    original = (
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis as runtime\n\n"
        "app = FastAPI()\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n"
    )
    main_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert {
        "file": "app/main.py",
        "reason": (
            "Aliased or wildcard imports of configured_fervis must be edited manually."
        ),
    } in envelope["payload"]["blocked_edits"]
    assert main_path.read_text(encoding="utf-8") == original


def test_fervis_init_blocks_ambiguous_fastapi_entrypoints(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    (root / "main.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "application entrypoint",
            "reason": "Multiple FastAPI app entrypoints were found; mount manually.",
        }
    ]


def test_fervis_init_blocks_fastapi_source_root_outside_project(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'api'\n"
        "dependencies = ['fastapi']\n\n"
        "[tool.hatch.build.targets.wheel]\n"
        "sources = ['../backend']\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "pyproject.toml",
            "reason": (
                "Python source roots must be relative paths inside the project."
            ),
        }
    ]


def test_fervis_doctor_rejects_fastapi_mount_on_wrong_object(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fervis import configured_fervis\n\n"
        "app = FastAPI()\n"
        "configured_fervis().mount(other_app)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_doctor_rejects_fastapi_import_after_mount(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "app" / "main.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n"
        "configured_fervis().mount(app)\n"
        "from fervis import configured_fervis\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.fastapi.mount"]["status"] == "failed"


def test_fervis_doctor_rejects_django_imports_after_urlpatterns(
    tmp_path: Path,
) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    (root / "config" / "urls.py").write_text(
        "urlpatterns = [path(configured_fervis().routes.django_path, include(configured_fervis().urls))]\n"
        "from django.urls import include, path\n"
        "from fervis import configured_fervis\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["framework.django.urls"]["status"] == "failed"


def test_fervis_init_requires_yes_for_writes(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["changed_files"] == []
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": "Pass --yes to write generated Fervis config files.",
        }
    ]
    assert not (root / "config" / "fervis.json").exists()


def test_fervis_init_does_not_write_without_project_marker(tmp_path: Path) -> None:
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(tmp_path),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["changed_files"] == []
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": "No Django, FastAPI, or Flask project root marker was found.",
        }
    ]
    assert not (tmp_path / "config" / "fervis.json").exists()


def test_fervis_doctor_validates_generated_config(tmp_path: Path) -> None:
    root = _django_project(tmp_path)
    run_init_command(
        ("init", "--framework", "django", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert envelope["command"] == "doctor"
    assert envelope["status"] == "blocked"
    assert envelope["payload_schema"] == "fervis-doctor-report.v0.1"
    assert checks["project.detected"]["status"] == "passed"
    assert checks["config.exists"]["status"] == "passed"
    assert checks["config.imports"]["status"] == "passed"
    assert checks["model.default_ref_syntax"]["status"] == "passed"
    assert checks["model.provider_declared"]["status"] == "passed"
    assert checks["source.explicit"]["status"] == "passed"
    assert checks["routes.prefix_valid"]["status"] == "passed"
    assert checks["framework.django.installed_apps"]["status"] == "passed"
    assert checks["framework.django.urls"]["status"] == "passed"
    assert checks["persistence.target"]["status"] == "passed"
    assert checks["persistence.connection"]["status"] == "passed"
    assert checks["persistence.migrations"]["status"] == "failed"
    assert checks["persistence.migrations"]["fix"] == run_migrate_action()
    assert checks["persistence.tables"]["status"] == "failed"
    assert envelope["next_actions"] == [
        edit_config_action(),
        configure_auth_action(framework="django"),
        run_migrate_action(),
    ]
    assert not (root / ".fervis" / "fervis.sqlite3").exists()


def test_fervis_doctor_requires_host_api_domain_context(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    missing_stdout = StringIO()
    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=missing_stdout,
    )
    missing_checks = {
        check["id"]: check
        for check in json.loads(missing_stdout.getvalue())["payload"]["checks"]
    }

    assert missing_checks["host.organization_name"] == {
        "id": "host.organization_name",
        "status": "failed",
        "message": "host.organization_name must identify the host organization.",
        "fix": edit_config_action(),
    }
    assert missing_checks["host.about_api"] == {
        "id": "host.about_api",
        "status": "failed",
        "message": (
            "host.about_api must describe the API domain and the factual questions "
            "it supports."
        ),
        "fix": edit_config_action(),
    }

    run_config_command(
        ("config", "set", "host.organization_name", "Example"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_config_command(
        (
            "config",
            "set",
            "host.about_api",
            "The Example API helps operators work with orders and inventory.",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )
    configured_stdout = StringIO()
    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=configured_stdout,
    )
    configured_checks = {
        check["id"]: check
        for check in json.loads(configured_stdout.getvalue())["payload"]["checks"]
    }

    assert configured_checks["host.organization_name"] == {
        "id": "host.organization_name",
        "status": "passed",
        "message": "Host organization is named.",
    }
    assert configured_checks["host.about_api"] == {
        "id": "host.about_api",
        "status": "passed",
        "message": "Host API domain and factual-question scope are described.",
    }


def test_fervis_doctor_validates_fastapi_mount(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert checks["framework.fastapi.mount"]["status"] == "passed"
    assert "persistence.database_url" not in checks
    assert checks["persistence.target"]["status"] == "passed"
    assert checks["persistence.migrations"]["status"] == "failed"
    assert checks["persistence.migrations"]["fix"] == run_migrate_action()
    assert envelope["next_actions"] == [
        edit_config_action(),
        run_migrate_action(),
        configure_auth_action(framework="fastapi"),
    ]
    assert not (root / ".fervis" / "fervis.sqlite3").exists()


def test_fervis_migrate_creates_sqlite_store_and_unblocks_doctor(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    migrate_stdout = StringIO()

    migrate_exit_code = run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=migrate_stdout,
    )

    migrate_envelope = json.loads(migrate_stdout.getvalue())
    assert migrate_exit_code == 0
    assert migrate_envelope["command"] == "migrate"
    assert migrate_envelope["payload_schema"] == "fervis-migration-result.v0.1"
    assert migrate_envelope["payload"]["target"] == "sqlite"
    assert migrate_envelope["payload"]["location"] == ".fervis/fervis.sqlite3"
    assert migrate_envelope["payload"]["status"] == "applied"
    assert migrate_envelope["payload"]["target_revision"] == "fervis.0003"
    assert migrate_envelope["next_actions"] == [run_doctor_action()]
    assert (root / ".fervis" / "fervis.sqlite3").is_file()
    with sqlite3.connect(root / ".fervis" / "fervis.sqlite3") as connection:
        created_indexes = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='index'"
            )
        }
    assert "fervis_work_idempotency_unique" in created_indexes
    assert "fervis_work_active_conversation_unique" in created_indexes
    assert "fervis_work_claim_idx" in created_indexes

    second_stdout = StringIO()
    second_exit_code = run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=second_stdout,
    )
    second_envelope = json.loads(second_stdout.getvalue())
    assert second_exit_code == 0
    assert second_envelope["payload"]["status"] == "up_to_date"
    assert second_envelope["payload"]["already_applied"] is True

    _configure_host_context(root)
    _configure_fastapi_auth(root)
    doctor_stdout = StringIO()
    doctor_exit_code = run_doctor_command(
        ("doctor", "--probe-read-context-key", "user_7"),
        project=discover_project(root),
        stdout=doctor_stdout,
    )
    doctor_envelope = json.loads(doctor_stdout.getvalue())
    checks = {check["id"]: check for check in doctor_envelope["payload"]["checks"]}
    assert doctor_exit_code == 0, json.dumps(doctor_envelope, indent=2)
    assert checks["persistence.migrations"]["status"] == "passed"
    assert checks["persistence.tables"]["status"] == "passed"


def test_fervis_migrate_cli_outcome_unblocks_doctor(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)

    init = _run_fervis(root, "init", "--framework", "fastapi", "--yes")
    migrate = _run_fervis(root, "migrate")
    _write_fastapi_auth_helpers(root)
    auth = _run_fervis(
        root,
        "auth",
        "configure",
        "--principal-dependency",
        "app.api.deps:get_current_user",
        "--principal-id-attr",
        "id",
        "--principal-resolver",
        "app.users:get_user_by_id",
        "--transport-mode",
        "in_process",
    )

    assert init.returncode == 0
    assert migrate.returncode == 0
    assert auth.returncode == 0
    _configure_host_context(root)
    doctor = _run_fervis(root, "doctor", "--probe-read-context-key", "user_7")
    assert doctor.returncode == 0
    migrate_envelope = json.loads(migrate.stdout)
    doctor_envelope = json.loads(doctor.stdout)
    checks = {check["id"]: check for check in doctor_envelope["payload"]["checks"]}

    assert migrate_envelope["command"] == "migrate"
    assert migrate_envelope["payload_schema"] == "fervis-migration-result.v0.1"
    assert migrate_envelope["payload"]["status"] == "applied"
    assert migrate_envelope["payload"]["target"] == "sqlite"
    assert (root / ".fervis" / "fervis.sqlite3").is_file()
    assert checks["persistence.migrations"]["status"] == "passed"
    assert checks["persistence.tables"]["status"] == "passed"


def test_fervis_migrate_supports_explicit_database_url_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    _patch_fastapi_mount(root)
    _write_fastapi_database_url_config(root)
    database_path = root / "runtime.sqlite3"
    monkeypatch.setenv("FERVIS_DATABASE_URL", f"sqlite:///{database_path}")
    stdout = StringIO()

    exit_code = run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["target"] == "database_url"
    assert envelope["payload"]["location"] == "FERVIS_DATABASE_URL"
    assert envelope["payload"]["status"] == "applied"
    assert database_path.is_file()

    _configure_host_context(root)
    _configure_fastapi_auth(root)
    doctor_stdout = StringIO()
    doctor_exit_code = run_doctor_command(
        ("doctor", "--probe-read-context-key", "user_7"),
        project=discover_project(root),
        stdout=doctor_stdout,
    )
    doctor_envelope = json.loads(doctor_stdout.getvalue())
    checks = {check["id"]: check for check in doctor_envelope["payload"]["checks"]}
    assert doctor_exit_code == 0, json.dumps(doctor_envelope, indent=2)
    assert checks["persistence.target"]["status"] == "passed"
    assert checks["persistence.connection"]["status"] == "passed"
    assert checks["persistence.migrations"]["status"] == "passed"
    assert checks["persistence.tables"]["status"] == "passed"


def test_fervis_migrate_blocks_non_sqlite_database_url_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    _patch_fastapi_mount(root)
    _write_fastapi_database_url_config(root)
    monkeypatch.setenv("FERVIS_DATABASE_URL", "postgresql://example/fervis")
    stdout = StringIO()

    exit_code = run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["status"] == "blocked"
    assert envelope["payload"]["error"] == (
        "DatabaseUrlPersistence supports only sqlite URLs in this slice."
    )

    doctor_stdout = StringIO()
    doctor_exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=doctor_stdout,
    )
    doctor_envelope = json.loads(doctor_stdout.getvalue())
    checks = {check["id"]: check for check in doctor_envelope["payload"]["checks"]}
    assert doctor_exit_code == 2
    assert checks["persistence.target"]["status"] == "failed"
    assert checks["persistence.target"]["message"] == (
        "DatabaseUrlPersistence supports only sqlite URLs in this slice."
    )


def test_fervis_doctor_does_not_create_explicit_sqlite_database_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fastapi_project(tmp_path)
    _patch_fastapi_mount(root)
    _write_fastapi_database_url_config(root)
    database_path = root / "runtime.sqlite3"
    monkeypatch.setenv("FERVIS_DATABASE_URL", f"sqlite:///{database_path}")
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert not database_path.exists()
    assert checks["persistence.target"]["status"] == "passed"
    assert checks["persistence.connection"]["status"] == "passed"
    assert checks["persistence.migrations"]["status"] == "failed"


def test_fervis_sqlite_engine_enforces_foreign_keys(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    database_path = root / ".fervis" / "fervis.sqlite3"
    engine = create_sqlite_engine(f"sqlite:///{database_path}")

    with engine.begin() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO fervis_question "
                    "(question_id, conversation_id, conversation_sequence, "
                    "origin_message_ref, original_question, created_at) "
                    "VALUES ('q1', 'missing-conversation', 1, '', 'Question?', "
                    "'2026-06-23T00:00:00')"
                )
            )


def test_fervis_sqlite_schema_enforces_run_scoped_relationships(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    engine = create_sqlite_engine(f"sqlite:///{root / '.fervis' / 'fervis.sqlite3'}")

    with engine.begin() as connection:
        _insert_minimal_run(connection, run_id="r1", run_number=1)
        _insert_minimal_run(connection, run_id="r2", run_number=2)
        _insert_minimal_step(connection, run_id="r1", step_id="s1")
        _insert_minimal_model_call(
            connection,
            run_id="r1",
            step_id="s1",
            model_call_id="mc1",
        )

        connection.execute(
            text(
                "INSERT INTO fervis_model_call_usage "
                "(usage_id, run_id, model_call_id, usage_kind, quantity, unit, "
                "provider_usage_key, cost_micros, currency, price_basis_json, "
                "created_at) "
                "VALUES ('u1', 'r1', 'mc1', 'input_tokens', 1, 'token', "
                "'prompt_tokens', 0, 'USD', '{}', '2026-06-23T00:00:00')"
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO fervis_model_call_usage "
                    "(usage_id, run_id, model_call_id, usage_kind, quantity, unit, "
                    "provider_usage_key, cost_micros, currency, price_basis_json, "
                    "created_at) "
                    "VALUES ('u2', 'r2', 'mc1', 'input_tokens', 1, 'token', "
                    "'prompt_tokens', 0, 'USD', '{}', '2026-06-23T00:00:00')"
                )
            )


def test_fervis_artifact_body_constraint_allows_exactly_one_body(
    tmp_path: Path,
) -> None:
    root = _migrated_fastapi_project(tmp_path)
    engine = create_sqlite_engine(f"sqlite:///{root / '.fervis' / 'fervis.sqlite3'}")

    with engine.begin() as connection:
        _insert_minimal_run(connection, run_id="r1", run_number=1)
        _insert_minimal_step(connection, run_id="r1", step_id="s1")

        _insert_artifact(
            connection,
            artifact_id="a1",
            content="payload",
            storage_ref=None,
        )
        _insert_artifact(
            connection,
            artifact_id="a2",
            content=None,
            storage_ref="s3://bucket/key",
        )
        with pytest.raises(IntegrityError):
            _insert_artifact(
                connection,
                artifact_id="a3",
                content=None,
                storage_ref="",
            )
        with pytest.raises(IntegrityError):
            _insert_artifact(
                connection,
                artifact_id="a4",
                content="payload",
                storage_ref="s3://bucket/key",
            )
        with pytest.raises(IntegrityError):
            _insert_artifact(
                connection,
                artifact_id="a5",
                content=None,
                storage_ref=None,
            )


def test_fervis_doctor_detects_schema_column_drift(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    database_path = root / ".fervis" / "fervis.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("ALTER TABLE fervis_conversation ADD COLUMN stale TEXT")

    stdout = StringIO()
    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = {check["id"]: check for check in envelope["payload"]["checks"]}
    assert exit_code == 2
    assert checks["persistence.migrations"]["status"] == "passed"
    assert checks["persistence.tables"]["status"] == "failed"
    assert "extra columns stale" in checks["persistence.tables"]["message"]


def test_fervis_doctor_and_migrate_detect_schema_index_drift(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    database_path = root / ".fervis" / "fervis.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX fervis_work_idempotency_unique")

    doctor_stdout = StringIO()
    doctor_exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=doctor_stdout,
    )
    migrate_stdout = StringIO()
    migrate_exit_code = run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=migrate_stdout,
    )

    doctor_envelope = json.loads(doctor_stdout.getvalue())
    migrate_envelope = json.loads(migrate_stdout.getvalue())
    checks = {check["id"]: check for check in doctor_envelope["payload"]["checks"]}
    assert doctor_exit_code == 2
    assert migrate_exit_code == 2
    assert checks["persistence.tables"]["status"] == "failed"
    assert (
        "missing index fervis_work_idempotency_unique"
        in checks["persistence.tables"]["message"]
    )
    assert migrate_envelope["payload"]["status"] == "failed"
    assert (
        "missing index fervis_work_idempotency_unique"
        in migrate_envelope["payload"]["error"]
    )


def test_fervis_doctor_and_migrate_detect_partial_index_predicate_drift(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    database_path = root / ".fervis" / "fervis.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX fervis_work_active_conversation_unique")
        connection.execute(
            "CREATE UNIQUE INDEX fervis_work_active_conversation_unique "
            "ON fervis_run_work_item (tenant_id, conversation_id) "
            "WHERE status IN ('QUEUED')"
        )

    doctor_stdout = StringIO()
    doctor_exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=doctor_stdout,
    )
    migrate_stdout = StringIO()
    migrate_exit_code = run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=migrate_stdout,
    )

    doctor_envelope = json.loads(doctor_stdout.getvalue())
    migrate_envelope = json.loads(migrate_stdout.getvalue())
    checks = {check["id"]: check for check in doctor_envelope["payload"]["checks"]}
    assert doctor_exit_code == 2
    assert migrate_exit_code == 2
    assert checks["persistence.tables"]["status"] == "failed"
    assert (
        "fervis_work_active_conversation_unique"
        in checks["persistence.tables"]["message"]
    )
    assert migrate_envelope["payload"]["status"] == "failed"
    assert (
        "fervis_work_active_conversation_unique" in migrate_envelope["payload"]["error"]
    )


def test_fervis_init_blocks_invalid_source_encoding(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_bytes(b"# -*- coding: ascii -*-\n# \xff\n")
    stdout = StringIO()

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert envelope["payload"]["blocked_edits"]
    assert not (root / "config" / "fervis.json").exists()
    assert not (root / "config" / "__init__.py").exists()


def test_fervis_init_preserves_second_line_encoding_comment(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    main_path = root / "app" / "main.py"
    main_path.write_text(
        "# application entrypoint\n"
        "# -*- coding: utf-8 -*-\n"
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/')\n"
        "def get_health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )

    lines = main_path.read_text(encoding="utf-8").splitlines()
    assert exit_code == 0
    assert lines[:2] == ["# application entrypoint", "# -*- coding: utf-8 -*-"]
    assert lines.index("from fervis import configured_fervis") == 2


def _django_project(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    root.mkdir()
    (root / "manage.py").write_text(
        "#!/usr/bin/env python\n"
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "__init__.py").write_text("", encoding="utf-8")
    orders_dir = root / "orders"
    orders_dir.mkdir()
    (orders_dir / "__init__.py").write_text("", encoding="utf-8")
    (orders_dir / "apps.py").write_text("", encoding="utf-8")
    (config_dir / "settings.py").write_text(
        "INSTALLED_APPS = []\nROOT_URLCONF = 'config.urls'\n",
        encoding="utf-8",
    )
    (config_dir / "urls.py").write_text(
        "from django.urls import path\n\nurlpatterns = []\n",
        encoding="utf-8",
    )
    return root


def _fastapi_project(tmp_path: Path) -> Path:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    app_dir = root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n\n"
        "class HealthResponse(BaseModel):\n"
        "    status: str\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health/', response_model=HealthResponse)\n"
        "def get_health() -> HealthResponse:\n"
        "    return HealthResponse(status='ok')\n",
        encoding="utf-8",
    )
    return root


def _flask_project(tmp_path: Path) -> Path:
    root = tmp_path / "flask_api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'flask-api'\ndependencies = ['flask>=2']\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text(
        "from flask import Flask\n\napp = Flask(__name__)\n",
        encoding="utf-8",
    )
    return root


def _config_schema(root: Path) -> dict[str, object]:
    return json.loads((root / "config" / "fervis.json").read_text(encoding="utf-8"))


def _write_schema_config(root: Path, schema: dict[str, object]) -> None:
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "fervis.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _configure_host_context(root: Path) -> None:
    schema = _config_schema(root)
    host = schema["host"]
    assert isinstance(host, dict)
    host["organization_name"] = "Example"
    host["about_api"] = (
        "The Example API helps operators work with business and operational records."
    )
    _write_schema_config(root, schema)


def _valid_schema(framework: str) -> dict[str, object]:
    prefix = "/fervis/" if framework == "django" else "/fervis"
    source = (
        {
            "kind": "django_app",
            "name": "default",
            "app_modules": ["apps"],
            "path_prefixes": ["/"],
        }
        if framework == "django"
        else {
            "kind": "fastapi_app",
            "name": "default",
            "import_paths": ["app.main:app"],
            "path_prefixes": ["/health/"],
        }
    )
    return {
        "schema_version": "v0.1",
        "framework": framework,
        "default_environment": "local",
        "host": {
            "organization_name": "",
            "about_api": "",
            "timezone": "UTC",
        },
        "routes": {"prefix": prefix},
        "models": {
            "providers": [{"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}],
        },
        "sources": [source],
        "environments": {
            "local": {
                "models": {
                    "default": {"provider": "openai", "model_key": "gpt-5.4-mini"}
                },
                "persistence": {"kind": "sqlite", "path": ".fervis/fervis.sqlite3"},
            }
        },
    }


def _write_valid_django_config(root: Path) -> None:
    _write_schema_config(root, _valid_schema("django"))


def _write_valid_fastapi_config(root: Path) -> None:
    (root / "config").mkdir(exist_ok=True)
    _write_schema_config(root, _valid_schema("fastapi"))


def _write_fastapi_database_url_config(root: Path) -> None:
    (root / "config").mkdir(exist_ok=True)
    schema = _valid_schema("fastapi")
    environments = schema["environments"]
    assert isinstance(environments, dict)
    local = environments["local"]
    assert isinstance(local, dict)
    local["persistence"] = {
        "kind": "database_url",
        "url_env": "FERVIS_DATABASE_URL",
    }
    _write_schema_config(root, schema)


def _configure_fastapi_auth(root: Path) -> None:
    _write_fastapi_auth_helpers(root)
    run_auth_command(
        (
            "auth",
            "configure",
            "--principal-dependency",
            "app.api.deps:get_current_user",
            "--principal-id-attr",
            "id",
            "--principal-resolver",
            "app.users:get_user_by_id",
            "--transport-mode",
            "in_process",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )


def _write_fastapi_auth_helpers(root: Path) -> None:
    app_dir = root / "app"
    api_dir = app_dir / "api"
    api_dir.mkdir(exist_ok=True)
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "deps.py").write_text(
        "from app.users import User\n\n"
        "def get_current_user():\n"
        "    return User('anonymous')\n",
        encoding="utf-8",
    )
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    del tenant_id\n"
        "    return User(user_id)\n",
        encoding="utf-8",
    )


def _write_fastapi_auth_config(
    root: Path,
    *,
    credentials: dict[str, object] | None = None,
) -> None:
    environment: dict[str, object] = {"transport": {"mode": "in_process"}}
    if credentials is not None:
        environment["credentials"] = credentials
    (root / "config" / "fervis_auth.json").write_text(
        json.dumps(
            {
                "schema_version": "v0.1",
                "framework": "fastapi",
                "security": {"mode": "principal_reauthorization"},
                "principal": {
                    "source": "fastapi_dependency",
                    "dependency": "app.api.deps:get_current_user",
                    "id_attr": "id",
                    "resolver": "app.users:get_user_by_id",
                },
                "environments": {"local": environment},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_fastapi_mount(root: Path) -> None:
    main_path = root / "app" / "main.py"
    content = main_path.read_text(encoding="utf-8")
    if "from fervis import configured_fervis" not in content:
        content = content.replace(
            "from fastapi import FastAPI\n",
            "from fastapi import FastAPI\nfrom fervis import configured_fervis\n",
            1,
        )
    if "configured_fervis().mount(app)" not in content:
        content = content.rstrip() + "\n\nconfigured_fervis().mount(app)\n"
    main_path.write_text(content, encoding="utf-8")


def _migrated_fastapi_project(tmp_path: Path) -> Path:
    root = _fastapi_project(tmp_path)
    run_init_command(
        ("init", "--framework", "fastapi", "--yes"),
        project=discover_project(root),
        stdout=StringIO(),
    )
    run_migrate_command(
        ("migrate",),
        project=discover_project(root),
        stdout=StringIO(),
    )
    return root


def _insert_minimal_run(connection, *, run_id: str, run_number: int) -> None:
    connection.execute(
        text(
            "INSERT OR IGNORE INTO fervis_conversation "
            "(conversation_id, tenant_id, origin_kind, parent_conversation_id, "
            "forked_after_question_id, forked_after_run_id, origin_ref, "
            "read_context_ref, created_at) "
            "VALUES ('c1', 't1', 'initial', NULL, NULL, NULL, '', "
            ":read_context_ref, "
            "'2026-06-23T00:00:00')"
        ),
        {
            "read_context_ref": json.dumps(
                {"scheme": "test", "key": "owner", "tenant_key": None}
            )
        },
    )
    question_id = f"q{run_number}"
    connection.execute(
        text(
            "INSERT INTO fervis_question "
            "(question_id, conversation_id, conversation_sequence, "
            "origin_message_ref, original_question, created_at) "
            "VALUES (:question_id, 'c1', :sequence, '', 'Question?', "
            "'2026-06-23T00:00:00')"
        ),
        {"question_id": question_id, "sequence": run_number},
    )
    connection.execute(
        text(
            "INSERT INTO fervis_question_run "
            "(run_id, question_id, run_number, kind, trigger_kind, base_run_id, "
            "trigger_clarification_response_id, adapter_ref, runtime_version, "
            "created_at) "
            "VALUES (:run_id, :question_id, 1, 'model_assisted', 'initial', NULL, "
            "'', 'test', 'test', '2026-06-23T00:00:00')"
        ),
        {"run_id": run_id, "question_id": question_id},
    )


def _insert_minimal_step(connection, *, run_id: str, step_id: str) -> None:
    connection.execute(
        text(
            "INSERT INTO fervis_run_step "
            "(step_id, run_id, sequence, step_key, attempt, scope_type, scope_id, "
            "kind, started_at, finished_at, input_summary_json, "
            "output_summary_json, error_json, created_at) "
            "VALUES (:step_id, :run_id, 1, 'step', NULL, 'run', '', 'model', "
            "NULL, NULL, '{}', '{}', '{}', '2026-06-23T00:00:00')"
        ),
        {"step_id": step_id, "run_id": run_id},
    )


def _insert_minimal_model_call(
    connection,
    *,
    run_id: str,
    step_id: str,
    model_call_id: str,
) -> None:
    connection.execute(
        text(
            "INSERT INTO fervis_model_call "
            "(model_call_id, run_id, step_id, call_index, provider, model_key, "
            "provider_request_id, status, finish_reason, duration_ms, "
            "reasoning_effort, reasoning_budget_tokens, max_output_tokens, "
            "prompt_chars, schema_chars, tool_spec_chars, submitted_payload_chars, "
            "raw_output_chars, model_subcalls_json, created_at) "
            "VALUES (:model_call_id, :run_id, :step_id, 1, 'openai', 'gpt', '', "
            "'succeeded', 'stop', 1, '', NULL, NULL, 1, 1, 1, NULL, NULL, "
            "'[]', '2026-06-23T00:00:00')"
        ),
        {
            "model_call_id": model_call_id,
            "run_id": run_id,
            "step_id": step_id,
        },
    )


def _insert_artifact(
    connection,
    *,
    artifact_id: str,
    content: str | None,
    storage_ref: str | None,
) -> None:
    connection.execute(
        text(
            "INSERT INTO fervis_run_artifact "
            "(artifact_id, run_id, step_id, model_call_id, artifact_kind, "
            "content_hash, content, storage_ref, content_type, size_bytes, "
            "created_at) "
            "VALUES (:artifact_id, 'r1', 's1', NULL, 'payload', 'hash', "
            ":content, :storage_ref, 'text/plain', 1, '2026-06-23T00:00:00')"
        ),
        {
            "artifact_id": artifact_id,
            "content": content,
            "storage_ref": storage_ref,
        },
    )


def _run_fervis(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fervis.interfaces.cli.main", *args],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(API_DIR),
            "FERVIS_INVOCATION_CWD": str(root),
        },
        check=False,
        capture_output=True,
        text=True,
    )
