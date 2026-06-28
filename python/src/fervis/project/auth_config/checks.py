"""Doctor checks for schema-backed host auth configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os

from fervis.interfaces.agent.actions import (
    configure_auth_action,
    set_env_action,
)
from fervis.host_api.credentials import credential_key_env_from_auth_schema
from fervis.project.configuration import ConfigProblem
from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.importing import import_object, project_import_context

from .loading import load_auth_project_schema
from .validation import read_context_probe_checks


@dataclass(frozen=True)
class AuthCheck:
    id: str
    status: str
    message: str
    fix: dict[str, object] | None = None

    @property
    def passed(self) -> bool:
        return self.status != "failed"


def auth_checks(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
    *,
    probe_read_context_key: str | None = None,
) -> list[AuthCheck]:
    auth_schema = load_auth_project_schema(
        project,
        active_environment=loaded.active_environment,
    )
    if isinstance(auth_schema, ConfigProblem):
        return [
            AuthCheck(
                id="auth.config",
                status="failed",
                message=auth_schema.message,
                fix=_configure_fix(project, loaded),
            )
        ]
    checks = [
        AuthCheck(
            id="auth.config",
            status="passed",
            message=("config/fervis_auth.json contains a valid versioned auth schema."),
        ),
        AuthCheck(
            id="auth.security_mode_configured",
            status="passed",
            message="Auth schema declares the host security mode.",
        ),
        AuthCheck(
            id="auth.transport_mode_configured",
            status="passed",
            message="Auth schema declares the host read transport mode.",
        ),
        AuthCheck(
            id="auth.principal_source_configured",
            status="passed",
            message="Auth schema declares how Fervis captures the request principal.",
        ),
    ]
    checks.extend(_transport_checks(project, auth_schema.schema))
    checks.extend(_credential_checks(auth_schema.schema))
    if all(check.passed for check in checks):
        checks.extend(
            AuthCheck(
                id=check.id,
                status=check.status,
                message=check.message,
                fix=check.fix,
            )
            for check in read_context_probe_checks(
                project=project,
                loaded_config=loaded,
                auth_schema=auth_schema.schema,
                probe_key=probe_read_context_key,
            )
        )
    return checks


def _credential_checks(schema: dict[str, object]) -> list[AuthCheck]:
    key_env = credential_key_env_from_auth_schema(schema)
    if key_env is None:
        return []
    if os.getenv(key_env):
        return [
            AuthCheck(
                id="auth.delegated_credential_key",
                status="passed",
                message=(
                    "Delegated read credential encryption key is configured "
                    f"in {key_env}."
                ),
            )
        ]
    return [
        AuthCheck(
            id="auth.delegated_credential_key",
            status="failed",
            message=(
                "Delegated read credential capture is configured, but the "
                f"encryption key is missing from {key_env}."
            ),
            fix=set_env_action(key_env),
        )
    ]


def _transport_checks(
    project: ProjectInspection,
    schema: dict[str, object],
) -> list[AuthCheck]:
    transport = _dict(schema["transport"])
    if transport.get("mode") != "http":
        return []
    base_url_env = str(transport.get("base_url_env") or "")
    checks: list[AuthCheck] = []
    if os.getenv(base_url_env):
        checks.append(
            AuthCheck(
                id="auth.http_base_url",
                status="passed",
                message=f"HTTP execution base URL is available in {base_url_env}.",
            )
        )
    else:
        checks.append(
            AuthCheck(
                id="auth.http_base_url",
                status="failed",
                message=f"HTTP execution base URL is missing from {base_url_env}.",
                fix=set_env_action(base_url_env),
            )
        )
    checks.append(_http_overlay_check(project, schema))
    return checks


def _http_overlay_check(
    project: ProjectInspection,
    schema: dict[str, object],
) -> AuthCheck:
    transport = _dict(schema["transport"])
    import_path = str(transport.get("request_overlay_source") or "")
    try:
        with project_import_context(project.root_path):
            overlay = import_object(import_path)
    except Exception as exc:
        return AuthCheck(
            id="auth.http_request_overlay",
            status="failed",
            message=(
                f"HTTP request overlay source {import_path} could not be "
                f"imported: {exc}"
            ),
            fix=configure_auth_action(framework=project.framework),
        )
    if not callable(overlay):
        return AuthCheck(
            id="auth.http_request_overlay",
            status="failed",
            message=f"HTTP request overlay source is not callable: {import_path}",
            fix=configure_auth_action(framework=project.framework),
        )
    return AuthCheck(
        id="auth.http_request_overlay",
        status="passed",
        message="HTTP request overlay source is importable.",
    )


def _configure_fix(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> dict[str, object]:
    del loaded
    return configure_auth_action(framework=project.framework)


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
