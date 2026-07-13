"""Auth configuration command services."""

from __future__ import annotations

from dataclasses import dataclass, field

from fervis.project.config_io import AUTH_CONFIG_PATH, write_json_schema
from fervis.project.config_versions.auth import (
    AUTH_CONFIG_SCHEMA_VERSION,
    active_auth_schema,
    normalize_auth_schema,
)
from fervis.project.configuration import (
    ConfigProblem,
    load_fervis_project_schema,
)
from fervis.project.discovery import ProjectInspection
from fervis.project.edit_result import ProjectEditResult, blocked_edit
from fervis.project.importing import import_object, project_import_context


@dataclass(frozen=True)
class AuthConfigureResult:
    edit: ProjectEditResult = field(default_factory=ProjectEditResult)
    checks: list[dict[str, object]] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.edit.is_blocked

    def to_payload(self) -> dict[str, object]:
        payload = self.edit.to_payload()
        payload["checks"] = self.checks
        return payload


def configure_auth(
    project: ProjectInspection,
    *,
    framework: str | None,
    security_mode: str,
    transport_mode: str,
    principal_source: str | None = None,
    principal_dependency: str | None = None,
    principal_id_attr: str | None = None,
    principal_resolver: str | None = None,
    base_url_env: str | None = None,
    request_overlay_source: str | None = None,
    auth_query_params: tuple[str, ...] = (),
    credential_headers: tuple[str, ...] = (),
    credential_key_env: str | None = None,
    credential_ttl_seconds: int | None = None,
    explicit_env: str | None = None,
) -> AuthConfigureResult:
    loaded_config = load_fervis_project_schema(project, explicit_env=explicit_env)
    if isinstance(loaded_config, ConfigProblem):
        return _blocked("config/fervis.json", loaded_config.message)
    target_framework = _target_framework(project, framework)
    if isinstance(target_framework, AuthConfigureResult):
        return target_framework
    raw_schema = _auth_schema(
        framework=target_framework,
        security_mode=security_mode,
        transport_mode=transport_mode,
        environment=loaded_config.active_environment.name,
        principal_source=principal_source,
        principal_dependency=principal_dependency,
        principal_id_attr=principal_id_attr,
        principal_resolver=principal_resolver,
        base_url_env=base_url_env,
        request_overlay_source=request_overlay_source,
        auth_query_params=auth_query_params,
        credential_headers=credential_headers,
        credential_key_env=credential_key_env,
        credential_ttl_seconds=credential_ttl_seconds,
    )
    try:
        schema, _ = normalize_auth_schema(raw_schema)
        active_schema = active_auth_schema(
            schema,
            environment_name=loaded_config.active_environment.name,
        )
    except ValueError as exc:
        return _blocked(str(AUTH_CONFIG_PATH), str(exc))
    import_problem = _validate_import_paths(project, active_schema)
    if import_problem is not None:
        return _blocked(str(AUTH_CONFIG_PATH), import_problem)
    return _write_auth_schema(project, schema, active_schema=active_schema)


def _target_framework(
    project: ProjectInspection,
    framework: str | None,
) -> str | AuthConfigureResult:
    if framework in {None, ""}:
        return project.framework
    if framework == "django-drf":
        framework = "django"
    if framework not in {"django", "fastapi", "flask"}:
        return _blocked(
            str(AUTH_CONFIG_PATH),
            "Auth framework must be django-drf, django, fastapi, or flask.",
        )
    if framework != project.framework:
        return _blocked(
            str(AUTH_CONFIG_PATH),
            f"Auth framework {framework} does not match detected {project.framework}.",
        )
    return framework


def _auth_schema(
    *,
    framework: str,
    security_mode: str,
    transport_mode: str,
    environment: str,
    principal_source: str | None,
    principal_dependency: str | None,
    principal_id_attr: str | None,
    principal_resolver: str | None,
    base_url_env: str | None,
    request_overlay_source: str | None,
    auth_query_params: tuple[str, ...],
    credential_headers: tuple[str, ...],
    credential_key_env: str | None,
    credential_ttl_seconds: int | None,
) -> dict[str, object]:
    principal: dict[str, object]
    transport: dict[str, object] = {"mode": transport_mode}
    if transport_mode == "http":
        transport.update(
            {
                "base_url_env": base_url_env or "",
                "request_overlay_source": request_overlay_source or "",
            }
        )
        auth_query_param_values = [
            item.strip() for item in auth_query_params if item.strip()
        ]
        if auth_query_param_values:
            transport["auth_query_params"] = auth_query_param_values
    if framework == "django":
        principal = {
            "source": "django_request_user",
            "id_attr": principal_id_attr or "pk",
        }
    elif framework == "fastapi":
        principal = {
            "source": "fastapi_dependency",
            "dependency": principal_dependency or "",
            "id_attr": principal_id_attr or "",
        }
        if transport_mode == "in_process":
            principal["resolver"] = principal_resolver or ""
    else:
        source = principal_source or "flask_login_current_user"
        if source in {"flask_login_current_user", "flask_g"}:
            principal = {
                "source": source,
                "id_attr": principal_id_attr or "get_id",
            }
            if principal_resolver:
                principal["resolver"] = principal_resolver
        else:
            principal = {
                "source": source,
                "capture": principal_dependency or "",
                "resolver": principal_resolver or "",
            }
    environment_schema: dict[str, object] = {
        "transport": transport,
    }
    credential_header_values = [
        item.strip() for item in credential_headers if item.strip()
    ]
    if credential_header_values:
        environment_schema["credentials"] = {
            "source": "captured_request_headers",
            "headers": credential_header_values,
            "ttl_seconds": credential_ttl_seconds or 900,
            "encryption_key_env": credential_key_env or "FERVIS_READ_CREDENTIAL_KEY",
        }
    return {
        "schema_version": AUTH_CONFIG_SCHEMA_VERSION,
        "framework": framework,
        "security": {"mode": security_mode},
        "principal": principal,
        "environments": {
            environment: environment_schema,
        },
    }


def _write_auth_schema(
    project: ProjectInspection,
    schema: dict[str, object],
    *,
    active_schema: dict[str, object],
) -> AuthConfigureResult:
    absolute = project.root_path / AUTH_CONFIG_PATH
    absolute.parent.mkdir(parents=True, exist_ok=True)
    original = absolute.read_text(encoding="utf-8") if absolute.exists() else None
    write_json_schema(absolute, schema)
    updated = absolute.read_text(encoding="utf-8")
    if original == updated:
        return AuthConfigureResult(
            edit=ProjectEditResult(skipped_existing=[str(AUTH_CONFIG_PATH)]),
            checks=_configure_checks(active_schema),
        )
    return AuthConfigureResult(
        edit=ProjectEditResult(changed_files=[str(AUTH_CONFIG_PATH)]),
        checks=_configure_checks(active_schema),
    )


def _configure_checks(schema: dict[str, object]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = [
        {"id": "auth.schema_valid", "status": "passed"},
        {"id": "auth.security_mode_configured", "status": "passed"},
        {"id": "auth.transport_mode_configured", "status": "passed"},
        {"id": "auth.principal_source_configured", "status": "passed"},
    ]
    principal = _dict(schema["principal"])
    transport = _dict(schema["transport"])
    if principal.get("source") == "fastapi_dependency":
        checks.append(
            {"id": "auth.principal_dependency_importable", "status": "passed"}
        )
        if transport.get("mode") == "in_process":
            checks.append(
                {"id": "auth.principal_resolver_importable", "status": "passed"}
            )
    if transport.get("mode") == "http":
        checks.append(
            {"id": "auth.http_request_overlay_importable", "status": "passed"}
        )
    if schema.get("credentials"):
        checks.append(
            {"id": "auth.delegated_credentials_configured", "status": "passed"}
        )
    return checks


def _validate_import_paths(
    project: ProjectInspection,
    schema: dict[str, object],
) -> str | None:
    principal = _dict(schema["principal"])
    transport = _dict(schema["transport"])
    if transport.get("mode") == "http":
        paths = [
            (
                "Fervis auth",
                "HTTP request overlay source",
                str(transport.get("request_overlay_source") or ""),
            ),
        ]
    else:
        paths = []
    if principal.get("source") == "fastapi_dependency":
        paths.append(
            (
                "FastAPI auth",
                "principal dependency",
                str(principal.get("dependency") or ""),
            )
        )
        if transport.get("mode") == "in_process":
            paths.append(
                (
                    "FastAPI auth",
                    "principal resolver",
                    str(principal.get("resolver") or ""),
                )
            )
    if principal.get("source") in {"flask_login_current_user", "flask_g"}:
        resolver = str(principal.get("resolver") or "")
        if resolver:
            paths.append(("Flask auth", "principal resolver", resolver))
    if principal.get("source") == "callable":
        paths.append(
            (
                "Flask auth",
                "principal capture callable",
                str(principal.get("capture") or ""),
            )
        )
        paths.append(
            (
                "Flask auth",
                "principal resolver",
                str(principal.get("resolver") or ""),
            )
        )
    with project_import_context(project.root_path):
        for prefix, label, import_path in paths:
            try:
                value = import_object(import_path)
            except (ImportError, AttributeError, ValueError) as exc:
                return f"{prefix} {label} could not be imported: {exc}"
            if not callable(value):
                return f"{prefix} {label} is not callable: {import_path}"
    return None


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _blocked(file: str, reason: str) -> AuthConfigureResult:
    return AuthConfigureResult(edit=blocked_edit(file=file, reason=reason))
