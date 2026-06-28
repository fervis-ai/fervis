from __future__ import annotations

from ._support import *  # noqa: F401,F403

def test_cli_import_does_not_configure_django_settings() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from django.conf import settings; "
                "import fervis.interfaces.cli.main; "
                "print(settings.configured)"
            ),
        ],
        cwd=API_DIR,
        env={**os.environ, "PYTHONPATH": str(API_DIR)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "False"


def test_sql_runtime_django_command_bootstraps_django_before_composition(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "manage.py").write_text(
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
    (root / "config").mkdir()
    (root / "config" / "__init__.py").write_text("", encoding="utf-8")
    (root / "config" / "settings.py").write_text(
        "SECRET_KEY = 'test'\n"
        "INSTALLED_APPS = []\n",
        encoding="utf-8",
    )
    (root / "config" / "fervis.py").write_text(
        "fervis = {'schema_version': 'v0.1'}\n",
        encoding="utf-8",
    )
    cli_main = importlib.import_module("fervis.interfaces.cli.main")
    setup_calls = []
    blocked = []

    def fake_django_setup():
        setup_calls.append("called")

    def fake_load_config(project):
        assert project.framework == "django"
        from fervis.project.integration import ModelConfig, ProviderConfig

        return SimpleNamespace(
            config=SimpleNamespace(
                model=ModelConfig(
                    default_provider="openai",
                    default_model_key="gpt-5.4-mini",
                    providers=[
                        ProviderConfig(
                            name="openai",
                            allowed_model_keys=["gpt-5.4-mini"],
                        )
                    ],
                )
            )
        )

    def fake_sql_storage_bundle(*, project, loaded_config):
        del loaded_config
        assert setup_calls == ["called"]
        assert project.framework == "django"
        raise RuntimeError("sql runtime composition stopped")

    def record_blocked_command(args, *, project, reason):
        blocked.append((args, project, reason))
        return 2

    import fervis.storage.sql.bundle as sql_bundle

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(root))
    monkeypatch.setattr("django.setup", fake_django_setup)
    monkeypatch.setattr(cli_main, "load_fervis_project_config", fake_load_config)
    monkeypatch.setattr(sql_bundle, "sql_storage_bundle", fake_sql_storage_bundle)
    monkeypatch.setattr(cli_main, "run_blocked_command", record_blocked_command)

    assert cli_main.main(("runtime", "ask", "How many sales?")) == 2

    assert setup_calls == ["called"]
    assert blocked[0][2] == "sql runtime composition stopped"


def test_fervis_project_command_flags_stay_on_project_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cli_main = importlib.import_module("fervis.interfaces.cli.main")
    routed = []

    def record_project_command(args, *, project):
        routed.append((args, project))
        raise SystemExit(0)

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(tmp_path))
    monkeypatch.setattr(cli_main, "run_project_command", record_project_command)

    with pytest.raises(SystemExit) as error:
        cli_main.main(("project", "inspect", "--help"))

    assert error.value.code == 0
    assert routed[0][0] == ("project", "inspect", "--help")
    assert routed[0][1].root_path == tmp_path


def test_fervis_global_help_stays_on_parser_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cli_main = importlib.import_module("fervis.interfaces.cli.main")

    def fail_if_command_is_blocked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("help should not require framework routing")

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(tmp_path))
    monkeypatch.setattr(cli_main, "run_blocked_command", fail_if_command_is_blocked)

    with pytest.raises(SystemExit) as error:
        cli_main.main(("--help",))

    assert error.value.code == 0


def test_fervis_fastapi_runtime_command_requires_valid_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    cli_main = importlib.import_module("fervis.interfaces.cli.main")
    blocked = []

    def record_blocked_command(args, *, project, reason):
        blocked.append((args, project, reason))
        return 2

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(root))
    monkeypatch.setattr(cli_main, "run_blocked_command", record_blocked_command)

    assert cli_main.main(("runtime", "ask", "How many orders?")) == 2
    assert blocked[0][0] == ("runtime", "ask", "How many orders?")
    assert blocked[0][1].framework == "fastapi"
    assert "Fervis config was not found at config/fervis.json." in blocked[0][2]


def test_fervis_flask_inspect_uses_sql_runtime_storage(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['flask>=3']\n",
        encoding="utf-8",
    )
    cli_main = importlib.import_module("fervis.interfaces.cli.main")
    routed = []

    def fake_load_config(project):
        assert project.framework == "flask"
        from fervis.project.integration import ModelConfig, ProviderConfig

        return SimpleNamespace(
            config=SimpleNamespace(
                model=ModelConfig(
                    default_provider="openai",
                    default_model_key="gpt-5.4-mini",
                    providers=[
                        ProviderConfig(
                            name="openai",
                            allowed_model_keys=["gpt-5.4-mini"],
                        )
                    ],
                )
            )
        )

    def fake_sql_storage_bundle(*, project, loaded_config):
        del loaded_config
        assert project.framework == "flask"
        return SimpleNamespace(
            engine=object(),
            lineage_query=object(),
            observability_query=object(),
            prompt_capture_query=object(),
            questions=object(),
            run_work=object(),
        )

    def record_run_fervis(args, *, ports):
        routed.append((args, ports.project.framework))
        return 0

    import fervis.storage.sql.bundle as sql_bundle

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(root))
    monkeypatch.setattr(cli_main, "load_fervis_project_config", fake_load_config)
    monkeypatch.setattr(sql_bundle, "sql_storage_bundle", fake_sql_storage_bundle)
    monkeypatch.setattr(cli_main, "run_fervis", record_run_fervis)

    assert cli_main.main(("inspect", "artifact", "artifact_1")) == 0
    assert routed == [(("inspect", "artifact", "artifact_1"), "flask")]


def test_fervis_executable_exposes_command_surface() -> None:
    result = subprocess.run(
        ["./bin/fervis", "--help"],
        cwd=API_DIR,
        env={**os.environ, "FERVIS_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (
        "{init,catalog,config,doctor,migrate,auth,models,model,explain,goldset,inspect,project,runtime,sources,usage,worker}"
        in result.stdout
    )


def test_goldset_uses_sql_runtime_composition(monkeypatch, tmp_path: Path) -> None:
    from fervis.interfaces.cli import main as cli_main

    root = tmp_path / "api"
    root.mkdir()
    (root / "manage.py").write_text(
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
    (root / "config").mkdir()
    (root / "config" / "__init__.py").write_text("", encoding="utf-8")
    (root / "config" / "settings.py").write_text(
        "SECRET_KEY = 'test'\n"
        "INSTALLED_APPS = []\n",
        encoding="utf-8",
    )
    (root / "config" / "fervis.py").write_text(
        "fervis = {'schema_version': 'v0.1'}\n",
        encoding="utf-8",
    )
    routed = []

    def fake_load_config(project):
        from fervis.project.integration import ModelConfig, ProviderConfig

        return SimpleNamespace(
            config=SimpleNamespace(
                model=ModelConfig(
                    default_provider="openai",
                    default_model_key="gpt-5.4-mini",
                    providers=[
                        ProviderConfig(
                            name="openai",
                            allowed_model_keys=["gpt-5.4-mini"],
                        )
                    ],
                )
            )
        )

    def fake_sql_storage_bundle(*, project, loaded_config):
        del project, loaded_config
        return SimpleNamespace(
            lineage_query=object(),
            observability_query=object(),
            prompt_capture_query=object(),
            questions=object(),
            run_work=object(),
            engine=object(),
        )

    def record_run_fervis(args, *, ports, stdout=None, stderr=None):
        del stdout, stderr
        routed.append((args, ports.question_run_follower is not None))
        return 0

    import fervis.storage.sql.bundle as sql_bundle

    monkeypatch.setenv("FERVIS_INVOCATION_CWD", str(root))
    monkeypatch.setattr(cli_main, "load_fervis_project_config", fake_load_config)
    monkeypatch.setattr(sql_bundle, "sql_storage_bundle", fake_sql_storage_bundle)
    monkeypatch.setattr(cli_main, "run_fervis", record_run_fervis)

    args = (
        "goldset",
        "run",
        "--suite-path",
        str(tmp_path / "suite"),
        "--tenant-id",
        "tenant_1",
        "--principal-id",
        "principal_1",
    )

    assert cli_main.main(args) == 0
    assert routed == [(args, True)]


def test_fervis_executable_validates_explicit_python_runtime() -> None:
    result = subprocess.run(
        ["./bin/fervis", "project", "inspect"],
        cwd=API_DIR,
        env={**os.environ, "FERVIS_PYTHON": "/usr/bin/false"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert "FERVIS_PYTHON does not provide a Python 3.11+ runtime" in result.stderr
    assert "Traceback" not in result.stderr


def test_fervis_executable_fastapi_runtime_command_reaches_python_cli(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(API_DIR / "bin" / "fervis"),
            "runtime",
            "ask",
            "How many orders?",
        ],
        cwd=tmp_path,
        env={**os.environ, "FERVIS_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["schema"] == "fervis-command-result.v0.1"
    assert payload["command"] == "runtime.ask"
    assert payload["status"] == "blocked"
    assert (
        "Fervis config was not found at config/fervis.json."
        in payload["payload"]["error"]["message"]
    )
    assert result.stderr == ""

def test_fervis_executable_project_inspect_preserves_invocation_cwd(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [str(API_DIR / "bin" / "fervis"), "project", "inspect"],
        cwd=tmp_path,
        env={**os.environ, "FERVIS_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["command"] == "project.inspect"
    assert payload["status"] == "blocked"
    assert payload["payload"]["root_path"] == str(tmp_path)

def test_fervis_python_module_project_inspect_preserves_invocation_cwd(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fervis.interfaces.cli.main",
            "project",
            "inspect",
        ],
        cwd=API_DIR,
        env={
            **os.environ,
            "PYTHONPATH": str(API_DIR),
            "FERVIS_INVOCATION_CWD": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["command"] == "project.inspect"
    assert payload["status"] == "blocked"
    assert payload["payload"]["root_path"] == str(tmp_path)
