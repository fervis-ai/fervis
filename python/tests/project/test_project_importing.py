from __future__ import annotations

from pathlib import Path
import sys

from fervis.project.importing import (
    import_object,
    project_import_context,
    project_python_import_paths,
)


def test_repeated_project_import_context_reuses_host_module(tmp_path: Path) -> None:
    module_path = tmp_path / "host_app.py"
    module_path.write_text(
        "class App:\n"
        "    pass\n"
        "app = App()\n",
        encoding="utf-8",
    )

    with project_import_context(tmp_path):
        first = import_object("host_app:app")
    with project_import_context(tmp_path):
        second = import_object("host_app:app")

    assert second is first
    sys.modules.pop("host_app", None)


def test_nested_project_import_preserves_importing_namespace_module(
    tmp_path: Path,
) -> None:
    package = tmp_path / "host_app"
    package.mkdir()
    (package / "dependency.py").write_text("VALUE = 'ready'\n", encoding="utf-8")
    (package / "main.py").write_text(
        "from pathlib import Path\n"
        "from fervis.project.importing import import_object, project_import_context\n\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "with project_import_context(ROOT):\n"
        "    VALUE = import_object('host_app.dependency:VALUE')\n",
        encoding="utf-8",
    )

    with project_import_context(tmp_path):
        value = import_object("host_app.main:VALUE")

    assert value == "ready"
    for name in tuple(sys.modules):
        if name == "host_app" or name.startswith("host_app."):
            sys.modules.pop(name, None)


def test_uv_workspace_glob_resolves_member_src_layout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = [".", "packages/*"]\n',
        encoding="utf-8",
    )
    member = tmp_path / "packages" / "api"
    (member / "src" / "workspace_api").mkdir(parents=True)
    (member / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.wheel]\npackages = ["src/workspace_api"]\n',
        encoding="utf-8",
    )
    (member / "src" / "workspace_api" / "__init__.py").write_text(
        'VALUE = "ready"\n',
        encoding="utf-8",
    )

    import_paths = project_python_import_paths(tmp_path)

    assert member.resolve() in import_paths
    assert (member / "src").resolve() in import_paths
    assert (tmp_path / "packages" / "*").resolve() not in import_paths
    with project_import_context(tmp_path):
        assert import_object("workspace_api:VALUE") == "ready"
