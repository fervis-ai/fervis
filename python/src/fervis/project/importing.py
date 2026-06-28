"""Isolated host-project imports for config and catalog loading."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tomllib
from types import ModuleType
from collections.abc import Iterable, Iterator
import importlib
from typing import Any


@dataclass(frozen=True)
class ProjectModuleState:
    root_names: set[str]
    saved_modules: dict[str, ModuleType]


@contextmanager
def project_import_context(project_root: Path) -> Iterator[None]:
    root = project_root.resolve()
    import_paths = project_python_import_paths(project_root)
    runtime_path = _runtime_package_path()
    module_state = _remove_project_modules(project_root)
    original_cwd = Path.cwd()
    original_invocation_cwd = os.environ.get("FERVIS_INVOCATION_CWD")
    inserted_paths: list[str] = []
    for path in reversed(import_paths):
        text = str(path)
        sys.path.insert(0, text)
        inserted_paths.append(text)
    if str(runtime_path) not in sys.path:
        sys.path.insert(len(import_paths), str(runtime_path))
        inserted_paths.append(str(runtime_path))
    try:
        os.chdir(root)
        os.environ["FERVIS_INVOCATION_CWD"] = str(root)
        yield
    finally:
        os.chdir(original_cwd)
        if original_invocation_cwd is None:
            os.environ.pop("FERVIS_INVOCATION_CWD", None)
        else:
            os.environ["FERVIS_INVOCATION_CWD"] = original_invocation_cwd
        for path in inserted_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        _restore_project_modules(module_state)


def project_python_import_paths(project_root: Path) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (project_root.resolve(),)
            + tuple(
                (project_root / source_root).resolve()
                for source_root in project_python_source_roots(project_root)
            )
        )
    )


def project_python_source_roots(project_root: Path) -> tuple[str, ...]:
    pyproject = project_root / "pyproject.toml"
    source_roots = ["src"] if (project_root / "src").is_dir() else []
    if not pyproject.is_file():
        return tuple(source_roots)
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return tuple(source_roots)
    source_roots.extend(_pyproject_source_roots(project_root, data))
    return tuple(dict.fromkeys(source_roots))


def _pyproject_source_roots(
    project_root: Path,
    data: dict[str, object],
) -> list[str]:
    source_roots: list[str] = []
    hatch_sources = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("sources")
    )
    if isinstance(hatch_sources, list):
        source_roots.extend(_safe_source_roots(project_root, hatch_sources))
    hatch_packages = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages")
    )
    if isinstance(hatch_packages, list):
        source_roots.extend(_package_source_roots(project_root, hatch_packages))
    setuptools_package_dir = (
        data.get("tool", {}).get("setuptools", {}).get("package-dir") or {}
    )
    if isinstance(setuptools_package_dir, dict):
        source_roots.extend(
            _safe_source_roots(project_root, setuptools_package_dir.values())
        )
    setuptools_find_where = (
        data.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("where")
    )
    if isinstance(setuptools_find_where, list):
        source_roots.extend(_safe_source_roots(project_root, setuptools_find_where))
    poetry_packages = data.get("tool", {}).get("poetry", {}).get("packages")
    if isinstance(poetry_packages, list):
        source_roots.extend(
            _safe_source_roots(
                project_root,
                tuple(
                    item.get("from")
                    for item in poetry_packages
                    if isinstance(item, dict) and "from" in item
                ),
            )
        )
    uv_workspace_members = (
        data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members")
    )
    if isinstance(uv_workspace_members, list):
        source_roots.extend(_safe_source_roots(project_root, uv_workspace_members))
    return source_roots


def _package_source_roots(project_root: Path, values: Iterable[object]) -> list[str]:
    roots: list[str] = []
    for value in values:
        package_path = _safe_source_root(project_root, value)
        if not package_path:
            continue
        roots.append(Path(package_path).parent.as_posix())
    return roots


def _safe_source_roots(project_root: Path, values: Iterable[object]) -> list[str]:
    return [
        source_root
        for source_root in (_safe_source_root(project_root, value) for value in values)
        if source_root
    ]


def _safe_source_root(project_root: Path, value: object) -> str:
    text = str(value).strip().strip("/")
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "Python source roots must be relative paths inside the project."
        )
    absolute = (project_root / path).resolve()
    absolute.relative_to(project_root.resolve())
    return path.as_posix()


def _remove_project_modules(project_root: Path) -> ProjectModuleState:
    root_names = _project_root_module_names(project_root)
    affected_names = [
        name for name in sys.modules if _module_belongs_to_roots(name, root_names)
    ]
    saved = {name: sys.modules[name] for name in affected_names}
    for name in affected_names:
        sys.modules.pop(name, None)
    return ProjectModuleState(root_names=root_names, saved_modules=saved)


def _project_root_module_names(project_root: Path) -> set[str]:
    names: set[str] = set()
    for import_path in project_python_import_paths(project_root):
        if not import_path.is_dir():
            continue
        for path in import_path.iterdir():
            name = path.stem if path.is_file() and path.suffix == ".py" else path.name
            if name.isidentifier() and name != "fervis":
                names.add(name)
    return names


def _restore_project_modules(module_state: ProjectModuleState) -> None:
    for name in [
        name
        for name in sys.modules
        if _module_belongs_to_roots(name, module_state.root_names)
    ]:
        sys.modules.pop(name, None)
    sys.modules.update(module_state.saved_modules)


def _module_belongs_to_roots(name: str, roots: set[str]) -> bool:
    root = name.split(".", 1)[0]
    return root in roots


def import_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Import path must use module:object: {import_path}")
    module_name, object_name = import_path.split(":", 1)
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    value: Any = module
    for part in object_name.split("."):
        value = getattr(value, part)
    return value


def _runtime_package_path() -> Path:
    return Path(__file__).resolve().parents[3]
