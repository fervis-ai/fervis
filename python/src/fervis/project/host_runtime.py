"""Framework runtime setup for project-level commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .discovery import ProjectInspection
from .django_runtime import django_project_runtime


@contextmanager
def host_project_runtime(project: ProjectInspection) -> Iterator[None]:
    if project.framework == "django":
        with django_project_runtime(project):
            yield
        return
    yield
