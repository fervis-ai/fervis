"""Framework mount patching and validation."""

from __future__ import annotations

from ..discovery import ProjectInspection
from ..integration import FervisConfig
from ..source_detection import detect_django_source_schema
from .common import BlockedPatch, FrameworkCheck, FrameworkPatchResult
from .django import django_checks, patch_django
from .fastapi import (
    fastapi_checks,
    fastapi_factory_source_schema,
    fastapi_source_schema,
    patch_fastapi,
)
from .flask import flask_checks, flask_source_schema, patch_flask


def patch_framework_mount(
    project: ProjectInspection,
    framework: str,
    *,
    app_factory: str | None = None,
    app_target: str | None = None,
) -> FrameworkPatchResult:
    if framework == "django":
        return patch_django(project)
    if framework == "fastapi":
        return patch_fastapi(project, app_factory=app_factory)
    if framework == "flask":
        return patch_flask(project, app_target=app_target)
    return FrameworkPatchResult(
        blocked=[("config/fervis.json", f"Unsupported framework {framework}.")]
    )


def framework_mount_checks(
    project: ProjectInspection,
    config: FervisConfig,
) -> list[FrameworkCheck]:
    if project.framework == "django":
        return django_checks(project, config)
    if project.framework == "fastapi":
        return fastapi_checks(project, config)
    if project.framework == "flask":
        return flask_checks(project, config)
    return [
        FrameworkCheck(
            id="framework.hook_installed",
            passed=False,
            message="No supported framework hook can be validated.",
        )
    ]


def framework_source_schema(
    project: ProjectInspection,
    framework: str,
    *,
    app_factory: str | None = None,
    app_target: str | None = None,
    path_prefixes: tuple[str, ...] | None = None,
    blueprints: tuple[str, ...] = (),
) -> dict[str, object] | BlockedPatch:
    if framework == "django":
        return detect_django_source_schema(project.root_path)
    if framework == "fastapi":
        if app_factory:
            return fastapi_factory_source_schema(
                project.root_path,
                app_factory,
                path_prefixes=path_prefixes,
            )
        return fastapi_source_schema(project.root_path, path_prefixes=path_prefixes)
    if framework == "flask":
        return flask_source_schema(
            app_target,
            path_prefixes=path_prefixes,
            blueprints=blueprints,
        )
    return BlockedPatch("config/fervis.json", f"Unsupported framework {framework}.")
