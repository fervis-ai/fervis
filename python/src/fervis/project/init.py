"""Generate the host project's initial Fervis config."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .discovery import ProjectInspection, discover_project
from .config_io import (
    ConfigIOError,
    PROJECT_CONFIG_PATH,
    load_project_json_config,
    write_json_schema,
)
from .config_versions.main import PROJECT_CONFIG_SCHEMA_VERSION
from .edit_result import BlockedEdit
from .mounting import framework_source_schema, patch_framework_mount
from .mounting.common import BlockedPatch


@dataclass(frozen=True)
class InitResult:
    project: ProjectInspection
    changed_files: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    blocked_edits: list[BlockedEdit] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_edits)

    def to_payload(self) -> dict[str, object]:
        return {
            "changed_files": self.changed_files,
            "skipped_existing": self.skipped_existing,
            "blocked_edits": [
                {"file": item.file, "reason": item.reason}
                for item in self.blocked_edits
            ],
        }


def initialize_fervis_project(
    project: ProjectInspection,
    *,
    requested_framework: str | None,
    yes: bool,
    app_factory: str | None = None,
    app_target: str | None = None,
    path_prefixes: tuple[str, ...] | None = None,
    blueprints: tuple[str, ...] = (),
) -> InitResult:
    framework = requested_framework or project.framework
    if project.framework == "unknown" and not (
        (requested_framework == "fastapi" and app_factory)
        or (requested_framework == "flask" and app_target)
    ):
        return InitResult(
            project=project,
            blocked_edits=[
                BlockedEdit(
                    file=str(PROJECT_CONFIG_PATH),
                    reason=(
                        "No Django, FastAPI, or Flask project root marker was found."
                    ),
                )
            ],
        )
    if framework not in {"django", "fastapi", "flask"}:
        return InitResult(
            project=project,
            blocked_edits=[
                BlockedEdit(
                    file=str(PROJECT_CONFIG_PATH),
                    reason="Could not determine framework; rerun with --framework.",
                )
            ],
        )
    explicit_flask_target = requested_framework == "flask" and bool(app_target)
    if project.framework not in {"unknown", framework} and not (
        explicit_flask_target and project.config_path is None
    ):
        return InitResult(
            project=project,
            blocked_edits=[
                BlockedEdit(
                    file=str(PROJECT_CONFIG_PATH),
                    reason=(
                        f"Requested framework {framework} does not match detected "
                        f"framework {project.framework}."
                    ),
                )
            ],
        )
    if not yes:
        return InitResult(
            project=project,
            blocked_edits=[
                BlockedEdit(
                    file=str(PROJECT_CONFIG_PATH),
                    reason="Pass --yes to write generated Fervis config files.",
                )
            ],
        )

    config_path = project.expected_config_path or PROJECT_CONFIG_PATH
    absolute_path = project.root_path / config_path
    changed_files: list[str] = []
    skipped_existing: list[str] = []
    source: dict[str, object] | None = None
    if not absolute_path.exists() or (framework == "flask" and app_target):
        source = framework_source_schema(
            project,
            framework,
            app_factory=app_factory,
            app_target=app_target,
            path_prefixes=path_prefixes,
            blueprints=blueprints,
        )
        if isinstance(source, BlockedPatch):
            return InitResult(
                project=project,
                blocked_edits=[
                    BlockedEdit(file=source.path, reason=source.reason),
                ],
            )

    if absolute_path.exists():
        if source is None:
            skipped_existing.append(str(config_path))
        else:
            update = _config_source_update(
                project,
                config_path=config_path,
                source=source,
            )
            if isinstance(update, BlockedEdit):
                return InitResult(project=project, blocked_edits=[update])
            if update is None:
                skipped_existing.append(str(config_path))
            else:
                write_json_schema(absolute_path, update)
                changed_files.append(str(config_path))
    else:
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        assert source is not None
        write_json_schema(absolute_path, _config_template(framework, source))
        changed_files.append(str(config_path))

    framework_patch = patch_framework_mount(
        project,
        framework,
        app_factory=app_factory,
        app_target=app_target,
    )
    changed_files.extend(framework_patch.changed_files)
    skipped_existing.extend(framework_patch.skipped_existing)
    blocked_edits = [
        BlockedEdit(file=file, reason=reason)
        for file, reason in framework_patch.blocked
    ]
    refreshed = _refreshed_project(project, requested_framework=requested_framework)
    return InitResult(
        project=refreshed,
        changed_files=changed_files,
        skipped_existing=skipped_existing,
        blocked_edits=blocked_edits,
    )


def _config_source_update(
    project: ProjectInspection,
    *,
    config_path,
    source: dict[str, object],
) -> dict[str, object] | BlockedEdit | None:
    try:
        schema = load_project_json_config(
            project.root_path,
            config_path=config_path,
        ).raw_schema
    except ConfigIOError as exc:
        return BlockedEdit(file=str(config_path), reason=str(exc))
    if schema.get("sources") == [source]:
        return None
    if _can_replace_generated_default_source(schema.get("sources"), source):
        updated = dict(schema)
        updated["sources"] = [source]
        return updated
    return None


def _can_replace_generated_default_source(
    value: object,
    desired: dict[str, object],
) -> bool:
    if not isinstance(value, list) or len(value) != 1:
        return False
    existing = value[0]
    if not isinstance(existing, dict):
        return False
    if set(existing) != set(desired):
        return False
    if existing.get("kind") != desired.get("kind"):
        return False
    if existing.get("name") != "default" or desired.get("name") != "default":
        return False
    if existing.get("kind") == "flask_app":
        return existing.get("app_args") == [] and existing.get("app_kwargs") == {}
    return True


def _refreshed_project(
    project: ProjectInspection,
    *,
    requested_framework: str | None,
) -> ProjectInspection:
    refreshed = discover_project(project.root_path)
    if (
        requested_framework in {"django", "fastapi", "flask"}
        and refreshed.config_path is None
    ):
        return replace(refreshed, framework=requested_framework)
    return refreshed


def _config_template(framework: str, source: dict[str, object]) -> dict[str, object]:
    routes_prefix = "/fervis/"
    if framework == "fastapi":
        routes_prefix = "/fervis"
    return {
        "schema_version": PROJECT_CONFIG_SCHEMA_VERSION,
        "framework": framework,
        "default_environment": "local",
        "host": {
            "organization_name": "",
            "about_api": "",
        },
        "routes": {"prefix": routes_prefix},
        "sources": [source],
        "models": {
            "providers": [
                {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]},
            ],
        },
        "environments": {
            "local": {
                "models": {
                    "default": {"provider": "openai", "model_key": "gpt-5.4-mini"},
                },
                "persistence": {"kind": "sqlite", "path": ".fervis/fervis.sqlite3"},
            },
        },
    }
