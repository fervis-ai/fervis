"""Initial source declarations inferred from host project structure."""

from __future__ import annotations

from pathlib import Path

from .importing import project_python_import_paths
from .mounting.common import BlockedPatch


NEVER_SOURCE_MODULES = frozenset({"manage", "fervis"})
NON_APPLICATION_MODULES = frozenset({"config", "settings", "tests", "test"})


def detect_django_source_schema(root: Path) -> dict[str, object] | BlockedPatch:
    modules = _django_application_modules(root)
    if not modules:
        return BlockedPatch(
            "config/fervis.json",
            (
                "Could not identify local Django API modules. Add a source "
                "explicitly with `fervis sources add django-app`."
            ),
        )
    return {
        "kind": "django_app",
        "name": "default",
        "app_modules": list(modules),
        "path_prefixes": ["/"],
    }


def _django_application_modules(root: Path) -> tuple[str, ...]:
    module_names: set[str] = set()
    for import_path in project_python_import_paths(root):
        if not import_path.is_dir():
            continue
        for path in import_path.iterdir():
            name = path.stem if path.is_file() and path.suffix == ".py" else path.name
            if not _is_candidate_module_name(name):
                continue
            if path.is_dir() and not _is_django_app_package(path):
                continue
            module_names.add(name)
    return tuple(
        name for name in sorted(module_names) if name not in NON_APPLICATION_MODULES
    )


def _is_candidate_module_name(name: str) -> bool:
    return (
        name.isidentifier()
        and not name.startswith("_")
        and name not in NEVER_SOURCE_MODULES
    )


def _is_django_app_package(path: Path) -> bool:
    return (path / "__init__.py").is_file() and (path / "apps.py").is_file()
