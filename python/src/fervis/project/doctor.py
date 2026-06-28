"""Structured Fervis project diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from fervis.interfaces.agent.actions import (
    choose_framework_init_action,
    edit_config_action,
    run_init_action,
    set_env_action,
)
from fervis.model_io.models import ModelRef
from fervis.model_io.providers.specs import (
    supported_provider_spec,
)
from .configuration import (
    ConfigProblem,
    LoadedFervisConfig,
    load_fervis_project_config,
)
from .discovery import ProjectInspection
from .mounting import framework_mount_checks


@dataclass(frozen=True)
class DoctorCheck:
    id: str
    status: str
    message: str
    fix: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
        }
        if self.fix is not None:
            payload["fix"] = self.fix
        return payload


@dataclass(frozen=True)
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def is_failed(self) -> bool:
        return any(check.status == "failed" for check in self.checks)

    def to_payload(self) -> dict[str, object]:
        return {
            "checks": [check.to_payload() for check in self.checks],
        }


@dataclass(frozen=True)
class DoctorOptions:
    probe_read_context_key: str | None = None


def inspect_fervis_project(
    project: ProjectInspection,
    *,
    options: DoctorOptions | None = None,
) -> DoctorReport:
    options = options or DoctorOptions()
    checks: list[DoctorCheck] = [
        _project_detected_check(project),
        _config_exists_check(project),
    ]
    if project.framework == "unknown" or project.config_path is None:
        return DoctorReport(checks)

    loaded = load_fervis_project_config(project)
    checks.append(_config_load_check(loaded))
    if isinstance(loaded, ConfigProblem):
        return DoctorReport(checks)

    checks.extend(_config_success_checks(project))
    checks.extend(_model_checks(loaded))
    checks.extend(_framework_checks(project, loaded))
    checks.extend(_catalog_checks(project, loaded))
    checks.extend(_auth_checks(project, loaded, options=options))
    checks.extend(_persistence_checks(project, loaded))
    return DoctorReport(checks)


def _config_success_checks(project: ProjectInspection) -> list[DoctorCheck]:
    return [
        DoctorCheck(
            id="config.schema",
            status="passed",
            message="config/fervis.json contains a valid Fervis schema.",
        ),
        DoctorCheck(
            id="config.framework_matches",
            status="passed",
            message=f"Config matches {project.framework}.",
        ),
        DoctorCheck(
            id="model.default_ref_syntax",
            status="passed",
            message="Default model ref uses provider:model syntax.",
        ),
        DoctorCheck(
            id="model.provider_declared",
            status="passed",
            message="Default model provider is declared.",
        ),
        DoctorCheck(
            id="source.explicit",
            status="passed",
            message="At least one explicit source is declared.",
        ),
        DoctorCheck(
            id="routes.prefix_valid",
            status="passed",
            message="Runtime route prefix is structurally valid.",
        ),
    ]


def _model_checks(loaded: LoadedFervisConfig) -> list[DoctorCheck]:
    ref = ModelRef(
        provider=loaded.config.model.default_provider,
        model_id=loaded.config.model.default_model_key,
    )
    spec = supported_provider_spec(ref.provider)
    checks = [
        DoctorCheck(
            id="model.ref_resolves",
            status="passed",
            message=f"Active model resolves to provider {ref.provider}.",
        ),
        DoctorCheck(
            id="model.strict_tools",
            status="passed" if spec.strict_tool_certified else "failed",
            message=(
                f"Provider {ref.provider} supports required strict tool calls."
                if spec.strict_tool_certified
                else f"Provider {ref.provider} does not support strict tool calls."
            ),
        ),
    ]
    checks.append(_model_api_key_check(spec.api_key_env))
    return checks


def _model_api_key_check(api_key_env: str) -> DoctorCheck:
    if os.getenv(api_key_env):
        return DoctorCheck(
            id="model.active_api_key",
            status="passed",
            message=f"Active model API key is available in {api_key_env}.",
        )
    return DoctorCheck(
        id="model.active_api_key",
        status="failed",
        message=f"Active model API key is missing from {api_key_env}.",
        fix=set_env_action(api_key_env),
    )


def _framework_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for check in framework_mount_checks(project, loaded.config):
        checks.append(
            DoctorCheck(
                id=check.id,
                status="passed" if check.passed else "failed",
                message=check.message,
                fix=None if check.passed else check.fix,
            )
        )
    return checks


def _persistence_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    from .persistence import inspect_persistence
    from fervis.storage.sql.checks import sql_storage_writability_checks

    checks = inspect_persistence(project, loaded)
    if all(check.passed for check in checks):
        checks.extend(
            sql_storage_writability_checks(project=project, loaded_config=loaded)
        )
    return [
        DoctorCheck(
            id=check.id,
            status="passed" if check.passed else "failed",
            message=check.message,
            fix=None if check.passed else check.fix,
        )
        for check in checks
    ]


def _catalog_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> list[DoctorCheck]:
    from .catalog_checks import catalog_checks

    return catalog_checks(project, loaded)


def _auth_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
    *,
    options: DoctorOptions,
) -> list[DoctorCheck]:
    from .auth_config.checks import auth_checks

    return [
        DoctorCheck(
            id=check.id,
            status=check.status,
            message=check.message,
            fix=check.fix,
        )
        for check in auth_checks(
            project,
            loaded,
            probe_read_context_key=options.probe_read_context_key,
        )
    ]


def _project_detected_check(project: ProjectInspection) -> DoctorCheck:
    if project.framework == "unknown":
        return DoctorCheck(
            id="project.detected",
            status="failed",
            message=project.blocked_reason or "No supported project framework found.",
            fix=choose_framework_init_action(),
        )
    return DoctorCheck(
        id="project.detected",
        status="passed",
        message=f"Detected {project.framework} project.",
    )


def _config_exists_check(project: ProjectInspection) -> DoctorCheck:
    if project.config_path is None:
        fix = (
            choose_framework_init_action()
            if project.framework == "unknown"
            else run_init_action(project.framework)
        )
        return DoctorCheck(
            id="config.exists",
            status="failed",
            message="Fervis config was not found at config/fervis.json.",
            fix=fix,
        )
    return DoctorCheck(
        id="config.exists",
        status="passed",
        message="Found config/fervis.json.",
    )


def _config_load_check(
    loaded: LoadedFervisConfig | ConfigProblem,
) -> DoctorCheck:
    if isinstance(loaded, ConfigProblem):
        return DoctorCheck(
            id=_problem_check_id(loaded),
            status="failed",
            message=loaded.message,
            fix=edit_config_action(),
        )
    return DoctorCheck(
        id="config.imports",
        status="passed",
        message="config/fervis.json validates successfully.",
    )


def _problem_check_id(problem: ConfigProblem) -> str:
    if problem.code == "config_export_missing":
        return "config.exported_object"
    if problem.code == "config_framework_mismatch":
        return "config.framework_matches"
    if problem.code == "model_ref_invalid":
        return "model.default_ref_syntax"
    if problem.code in {"model_provider_missing", "model_provider_unsupported"}:
        return "model.provider_declared"
    if problem.code.startswith("source_") or problem.code == "sources_missing":
        return "source.explicit"
    if problem.code == "routes_prefix_invalid":
        return "routes.prefix_valid"
    return "config.imports"
