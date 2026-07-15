from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from fervis.interfaces.cli.dispatch import run_init_command
from fervis.project import discover_project


def test_fervis_init_patches_flask_app_object_and_writes_json_config(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    (root / "manage.py").write_text("# Flask management script\n", encoding="utf-8")
    app_path = root / "app.py"
    app_path.write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return []\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
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
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["project"] == {
        "framework": "flask",
        "config_path": "config/fervis.json",
    }
    assert envelope["payload"]["changed_files"] == [
        "config/fervis.json",
        "app.py",
    ]
    assert envelope["payload"]["blocked_edits"] == []
    assert _read_json(root / "config" / "fervis.json")["sources"] == [
        {
            "kind": "flask_app",
            "name": "default",
            "app": "app:app",
            "app_args": [],
            "app_kwargs": {},
            "path_prefixes": ["/api/"],
            "blueprints": [],
        }
    ]
    text = app_path.read_text(encoding="utf-8")
    assert "from fervis import configured_fervis" in text
    assert "configured_fervis().init_app(app)" in text


def test_fervis_init_patches_explicit_flask_app_object_from_factory_call(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    app_path = root / "autoapp.py"
    app_path.write_text(
        "from service import create_app\n\n"
        "CONFIG = object()\n"
        "app = create_app(CONFIG)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--app",
            "autoapp:app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["blocked_edits"] == []
    assert app_path.read_text(encoding="utf-8") == (
        "from fervis import configured_fervis\n"
        "from service import create_app\n\n"
        "CONFIG = object()\n"
        "app = create_app(CONFIG)\n"
        "configured_fervis().init_app(app)\n"
    )


def test_fervis_init_flask_blocks_without_app_target(tmp_path: Path) -> None:
    root = _flask_project(tmp_path)
    stdout = StringIO()

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["project"]["framework"] == "flask"
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "config/fervis.json",
            "reason": "Flask app target must be provided with --app.",
        }
    ]
    assert not (root / "config" / "fervis.json").exists()


def test_fervis_init_patches_simple_flask_app_factory(tmp_path: Path) -> None:
    root = _flask_project(tmp_path)
    app_path = root / "app.py"
    app_path.write_text(
        "from flask import Flask\n\n"
        "def create_app():\n"
        "    app = Flask(__name__)\n"
        "    return app\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--app",
            "app:create_app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["payload"]["blocked_edits"] == []
    text = app_path.read_text(encoding="utf-8")
    assert "from fervis import configured_fervis" in text
    assert "    configured_fervis().init_app(app)\n    return app" in text


def test_fervis_init_ignores_returns_owned_by_nested_route_handlers(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    app_path = root / "app.py"
    app_path.write_text(
        "from flask import Flask\n\n"
        "def create_app():\n"
        "    app = Flask(__name__)\n\n"
        "    @app.get('/health')\n"
        "    def health():\n"
        "        return {'status': 'ok'}\n\n"
        "    return app\n",
        encoding="utf-8",
    )

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--app",
            "app:create_app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=StringIO(),
    )

    assert exit_code == 0
    text = app_path.read_text(encoding="utf-8")
    assert "    configured_fervis().init_app(app)\n    return app" in text


def test_fervis_init_honors_explicit_flask_app_when_manage_py_exists(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    (root / "manage.py").write_text("# Flask management script\n", encoding="utf-8")
    app_path = root / "app.py"
    app_path.write_text(
        "from flask import Flask\n\napp = Flask(__name__)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
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
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["project"]["framework"] == "flask"
    assert _read_json(root / "config" / "fervis.json")["framework"] == "flask"


def test_fervis_init_blocks_ambiguous_flask_factory_without_modifying_file(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    app_path = root / "app.py"
    original = (
        "from flask import Flask\n\n"
        "def create_app(testing=False):\n"
        "    if testing:\n"
        "        return Flask('test')\n"
        "    app = Flask(__name__)\n"
        "    return app\n"
    )
    app_path.write_text(original, encoding="utf-8")
    stdout = StringIO()

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--app",
            "app:create_app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "app.py",
            "reason": (
                "Could not prove `create_app` has one `app = Flask(...)` and "
                "one `return app`; mount manually."
            ),
        }
    ]
    assert app_path.read_text(encoding="utf-8") == original


def test_fervis_init_updates_json_config_after_blocked_flask_retry(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    app_path = root / "server.py"
    app_path.write_text(
        "import connexion\n\nconnex_app = connexion.App(__name__)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_init_command(
        (
            "init",
            "--framework",
            "flask",
            "--app",
            "server:connex_app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert envelope["payload"]["blocked_edits"] == [
        {
            "file": "server.py",
            "reason": (
                "Could not prove `connex_app` has one `app = Flask(...)` and one "
                "`return app`; mount manually."
            ),
        }
    ]
    assert _read_json(root / "config" / "fervis.json")["sources"] == [
        {
            "kind": "flask_app",
            "name": "default",
            "app": "server:connex_app",
            "app_args": [],
            "app_kwargs": {},
            "path_prefixes": ["/api/"],
            "blueprints": [],
        }
    ]
    assert app_path.read_text(encoding="utf-8") == (
        "import connexion\n\nconnex_app = connexion.App(__name__)\n"
    )
    app_package = root / "api"
    app_package.mkdir()
    (app_package / "__init__.py").write_text("", encoding="utf-8")
    (app_package / "app.py").write_text(
        "from flask import Flask\n\n"
        "def create_app(testing=False):\n"
        "    if testing:\n"
        "        return Flask('test')\n"
        "    app = Flask(__name__)\n"
        "    return app\n",
        encoding="utf-8",
    )
    second_exit, second_envelope = _run_init(
        root,
        (
            "--framework",
            "flask",
            "--app",
            "api.app:create_app",
            "--source-prefix",
            "/api/v1/dockd",
            "--blueprint",
            "dockd",
            "--yes",
        ),
    )

    assert second_exit == 2
    assert second_envelope["payload"]["changed_files"] == ["config/fervis.json"]
    assert second_envelope["payload"]["skipped_existing"] == []
    assert _read_json(root / "config" / "fervis.json")["sources"] == [
        {
            "kind": "flask_app",
            "name": "default",
            "app": "api.app:create_app",
            "app_args": [],
            "app_kwargs": {},
            "path_prefixes": ["/api/v1/dockd/"],
            "blueprints": ["dockd"],
        }
    ]


def test_fervis_init_does_not_overwrite_custom_flask_source_on_retry(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    (root / "app.py").write_text(
        "from flask import Flask\n\n"
        "def create_app(config=None):\n"
        "    app = Flask(__name__)\n"
        "    return app\n",
        encoding="utf-8",
    )
    _run_init(
        root,
        (
            "--framework",
            "flask",
            "--app",
            "app:create_app",
            "--source-prefix",
            "/api/",
            "--yes",
        ),
    )
    schema = _read_json(root / "config" / "fervis.json")
    schema["sources"][0]["app_kwargs"] = {"config": "testing"}
    (root / "config" / "fervis.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exit_code, envelope = _run_init(
        root,
        (
            "--framework",
            "flask",
            "--app",
            "api.app:create_app",
            "--source-prefix",
            "/api/v1/",
            "--yes",
        ),
    )

    assert exit_code == 2
    assert envelope["payload"]["changed_files"] == []
    assert "config/fervis.json" in envelope["payload"]["skipped_existing"]
    assert _read_json(root / "config" / "fervis.json")["sources"] == schema["sources"]


def _flask_project(tmp_path: Path) -> Path:
    root = tmp_path / "flask_api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'flask-api'\ndependencies = ['flask>=3.0']\n",
        encoding="utf-8",
    )
    return root


def _run_init(root: Path, args: tuple[str, ...]) -> tuple[int, dict[str, object]]:
    stdout = StringIO()
    exit_code = run_init_command(
        ("init", *args),
        project=discover_project(root),
        stdout=stdout,
    )
    return exit_code, json.loads(stdout.getvalue())


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
