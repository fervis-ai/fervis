"""Host Django setup for project-level CLI checks."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from collections.abc import Iterator

from .discovery import ProjectInspection
from .importing import project_import_context
from .mounting.common import BlockedPatch
from .mounting.django import django_settings_path


@contextmanager
def django_project_runtime(project: ProjectInspection) -> Iterator[None]:
    if project.framework != "django":
        yield
        return
    settings_module = _settings_module(project.root_path)
    original = os.environ.get("DJANGO_SETTINGS_MODULE")
    with project_import_context(project.root_path):
        os.environ["DJANGO_SETTINGS_MODULE"] = settings_module
        import django

        django.setup()
        try:
            yield
        finally:
            if original is None:
                os.environ.pop("DJANGO_SETTINGS_MODULE", None)
            else:
                os.environ["DJANGO_SETTINGS_MODULE"] = original


def _settings_module(root: Path) -> str:
    settings_path = django_settings_path(root)
    if isinstance(settings_path, BlockedPatch):
        raise ValueError(settings_path.reason)
    return Path(settings_path).with_suffix("").as_posix().replace("/", ".")
