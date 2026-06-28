from pathlib import Path

from fervis.project import discover_project


def test_discover_project_detects_django_root(tmp_path: Path) -> None:
    root = tmp_path / "shop"
    root.mkdir()
    _write_django_manage_py(root)
    (root / "config").mkdir()
    (root / "config" / "fervis.json").write_text(
        '{"schema_version": "v0.1", "framework": "django"}\n',
        encoding="utf-8",
    )
    nested = root / "apps" / "sales"
    nested.mkdir(parents=True)

    project = discover_project(nested)

    assert {
        "framework": project.framework,
        "root_path": project.root_path,
        "config_path": project.config_path,
        "expected_config_path": project.expected_config_path,
        "confidence": project.confidence,
        "blocked_reason": project.blocked_reason,
    } == {
        "framework": "django",
        "root_path": root,
        "config_path": Path("config") / "fervis.json",
        "expected_config_path": Path("config") / "fervis.json",
        "confidence": "high",
        "blocked_reason": None,
    }


def test_discover_project_reports_missing_fervis_config(tmp_path: Path) -> None:
    root = tmp_path / "shop"
    root.mkdir()
    _write_django_manage_py(root)

    project = discover_project(root)

    assert {
        "framework": project.framework,
        "config_path": project.config_path,
        "expected_config_path": project.expected_config_path,
        "blocked_reason": project.blocked_reason,
        "is_blocked": project.is_blocked,
    } == {
        "framework": "django",
        "config_path": None,
        "expected_config_path": Path("config") / "fervis.json",
        "blocked_reason": "Fervis config was not found at config/fervis.json.",
        "is_blocked": True,
    }


def test_discover_project_detects_fastapi_marker(tmp_path: Path) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi>=0.1']\n",
        encoding="utf-8",
    )
    (root / "config").mkdir()
    (root / "config" / "fervis.json").write_text(
        '{"schema_version": "v0.1", "framework": "fastapi"}\n',
        encoding="utf-8",
    )
    (root / "app").mkdir()

    project = discover_project(root / "app")

    assert {
        "framework": project.framework,
        "root_path": project.root_path,
        "config_path": project.config_path,
        "expected_config_path": project.expected_config_path,
        "confidence": project.confidence,
        "blocked_reason": project.blocked_reason,
    } == {
        "framework": "fastapi",
        "root_path": root,
        "config_path": Path("config") / "fervis.json",
        "expected_config_path": Path("config") / "fervis.json",
        "confidence": "high",
        "blocked_reason": None,
    }


def test_discover_project_detects_flask_marker(tmp_path: Path) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['flask>=3.0']\n",
        encoding="utf-8",
    )
    (root / "config").mkdir()
    (root / "config" / "fervis.json").write_text(
        '{"schema_version": "v0.1", "framework": "flask"}\n',
        encoding="utf-8",
    )
    (root / "app").mkdir()

    project = discover_project(root / "app")

    assert {
        "framework": project.framework,
        "root_path": project.root_path,
        "config_path": project.config_path,
        "expected_config_path": project.expected_config_path,
        "confidence": project.confidence,
        "blocked_reason": project.blocked_reason,
    } == {
        "framework": "flask",
        "root_path": root,
        "config_path": Path("config") / "fervis.json",
        "expected_config_path": Path("config") / "fervis.json",
        "confidence": "high",
        "blocked_reason": None,
    }


def test_discover_project_detects_flask_requirements_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "requirements.txt").write_text("Flask>=3.0\n", encoding="utf-8")
    (root / "app").mkdir()

    project = discover_project(root / "app")

    assert project.framework == "flask"
    assert project.root_path == root
    assert project.expected_config_path == Path("config") / "fervis.json"


def test_discover_project_detects_flask_from_included_requirements(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "requirements").mkdir()
    (root / "requirements.txt").write_text(
        "-r requirements/prod.txt\n",
        encoding="utf-8",
    )
    (root / "requirements" / "prod.txt").write_text(
        "Flask>=3.0\n",
        encoding="utf-8",
    )

    project = discover_project(root)

    assert project.framework == "flask"
    assert project.root_path == root


def test_discover_project_uses_existing_flask_config_as_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "config").mkdir()
    (root / "config" / "fervis.json").write_text(
        '{"schema_version": "v0.1", "framework": "flask"}\n',
        encoding="utf-8",
    )

    project = discover_project(root)

    assert project.framework == "flask"
    assert project.confidence == "high"
    assert project.blocked_reason is None


def test_discover_project_uses_config_to_disambiguate_multiple_framework_markers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi', 'flask']\n",
        encoding="utf-8",
    )
    (root / "config").mkdir()
    (root / "config" / "fervis.json").write_text(
        '{"schema_version": "v0.1", "framework": "fastapi"}\n',
        encoding="utf-8",
    )

    project = discover_project(root)

    assert project.framework == "fastapi"
    assert project.confidence == "high"
    assert project.blocked_reason is None


def test_discover_project_blocks_ambiguous_multiple_framework_markers_without_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi', 'flask']\n",
        encoding="utf-8",
    )

    project = discover_project(root)

    assert project.framework == "unknown"
    assert project.confidence == "low"
    assert project.blocked_reason == (
        "Multiple framework markers were found: fastapi, flask. "
        "Run Fervis init with --framework to choose one."
    )


def test_discover_project_does_not_let_config_override_framework_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shop"
    root.mkdir()
    _write_django_manage_py(root)
    (root / "config").mkdir()
    (root / "config" / "fervis.json").write_text(
        '{"schema_version": "v0.1", "framework": "fastapi"}\n',
        encoding="utf-8",
    )

    project = discover_project(root)

    assert project.framework == "django"
    assert project.config_path == Path("config") / "fervis.json"


def test_discover_project_does_not_treat_django_comment_as_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "manage.py").write_text(
        "# Flask script; not DJANGO_SETTINGS_MODULE\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['flask>=3.0']\n",
        encoding="utf-8",
    )

    project = discover_project(root)

    assert project.framework == "flask"


def test_discover_project_does_not_treat_generic_python_layout_as_fastapi(
    tmp_path: Path,
) -> None:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'api'\n", encoding="utf-8")
    (root / "app").mkdir()

    project = discover_project(root / "app")

    assert project.framework == "unknown"
    assert (
        project.blocked_reason
        == "No Django, FastAPI, or Flask project marker was found."
    )


def test_discover_project_reports_unknown_without_importing_frameworks(
    tmp_path: Path,
) -> None:
    project = discover_project(tmp_path)

    assert {
        "framework": project.framework,
        "root_path": project.root_path,
        "config_path": project.config_path,
        "expected_config_path": project.expected_config_path,
        "confidence": project.confidence,
        "blocked_reason": project.blocked_reason,
        "is_blocked": project.is_blocked,
    } == {
        "framework": "unknown",
        "root_path": tmp_path,
        "config_path": None,
        "expected_config_path": None,
        "confidence": "low",
        "blocked_reason": "No Django, FastAPI, or Flask project marker was found.",
        "is_blocked": True,
    }


def _write_django_manage_py(root: Path) -> None:
    (root / "manage.py").write_text(
        "#!/usr/bin/env python\n"
        "import os\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')\n",
        encoding="utf-8",
    )
