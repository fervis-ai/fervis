"""Capture and resolve Flask principals for host API execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.host_api.contracts.ports import EndpointExecutionError
from fervis.project.importing import import_object


@dataclass(frozen=True)
class FlaskPrincipalOverride:
    source: str
    resolver: str
    key: str | None
    tenant_id: str | None = None
    principal_id_attr: str = "id"


def capture_flask_read_context(
    schema: dict[str, object] | None,
    *,
    request: Any,
) -> ReadContextRef:
    principal = flask_principal_schema(schema)
    source = str(principal.get("source") or "")
    if source == "flask_g":
        return _capture_flask_g(principal)
    if source == "flask_login_current_user":
        return _capture_flask_login(principal)
    if source == "callable":
        captured = import_object(str(principal["capture"]))(request)
        if isinstance(captured, ReadContextRef):
            return captured
        return ReadContextRef.from_storage_dict(captured)
    return ReadContextRef(scheme="anonymous")


def flask_principal_schema(
    schema: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(schema, dict):
        return {}
    principal = schema.get("principal")
    if not isinstance(principal, dict):
        return {}
    if principal.get("source") not in {
        "flask_g",
        "flask_login_current_user",
        "callable",
    }:
        return {}
    return principal


def flask_principal_override(
    schema: dict[str, object] | None,
    *,
    read_context_ref: ReadContextRef,
    tenant_id: str | None,
) -> FlaskPrincipalOverride:
    principal = flask_principal_schema(schema)
    if not principal:
        raise ValueError("Flask reads require configured principal reauthorization.")
    resolver = str(principal.get("resolver") or "")
    if not resolver:
        raise ValueError("Flask reads require a configured principal resolver.")
    return FlaskPrincipalOverride(
        source=str(principal["source"]),
        resolver=resolver,
        key=read_context_ref.key,
        tenant_id=tenant_id,
        principal_id_attr=str(principal.get("id_attr") or "id"),
    )


def resolve_flask_principal(
    override: FlaskPrincipalOverride,
) -> Any:
    resolver = import_object(override.resolver)
    principal = resolver(override.key, override.tenant_id)
    _validate_resolved_principal(principal, override=override)
    return principal


def _capture_flask_g(principal: dict[str, object]) -> ReadContextRef:
    try:
        from flask import g
    except ImportError as exc:
        raise RuntimeError("Flask read-context capture requires flask.") from exc
    subject = getattr(g, "current_user", None) or getattr(g, "user", None)
    if subject is None:
        return ReadContextRef(scheme="anonymous")
    return _subject_ref(subject, principal)


def _capture_flask_login(principal: dict[str, object]) -> ReadContextRef:
    try:
        current_user = import_object("flask_login:current_user")
    except ImportError as exc:
        raise RuntimeError(
            "flask_login_current_user capture requires flask-login."
        ) from exc
    if getattr(current_user, "is_anonymous", False):
        return ReadContextRef(scheme="anonymous")
    return _subject_ref(current_user, principal)


def _subject_ref(subject: Any, principal: dict[str, object]) -> ReadContextRef:
    id_attr = str(principal.get("id_attr") or "id")
    key = getattr(subject, id_attr, None)
    if callable(key):
        key = key()
    if key is None and isinstance(subject, dict):
        key = subject.get(id_attr)
    if key is None:
        key = subject
    return ReadContextRef(scheme="flask_principal", key=str(key))


def _validate_resolved_principal(
    principal: object,
    *,
    override: FlaskPrincipalOverride,
) -> None:
    if principal is None:
        raise EndpointExecutionError(
            f"Flask resolver could not resolve principal: {override.key}"
        )
    actual = getattr(principal, override.principal_id_attr, None)
    if callable(actual):
        actual = actual()
    if actual is None and isinstance(principal, dict):
        actual = principal.get(override.principal_id_attr)
    if str(actual or "") != str(override.key or ""):
        raise EndpointExecutionError(
            "Flask resolver returned a different principal than requested."
        )
