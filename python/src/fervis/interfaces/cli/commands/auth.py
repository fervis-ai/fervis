"""`fervis auth` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.agent.actions import run_doctor_action
from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)
from fervis.project import ProjectInspection
from fervis.project.auth_config import configure_auth


def auth_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    if args.auth_command != "configure":
        raise ValueError(f"unsupported auth command: {args.auth_command}")
    result = configure_auth(
        project,
        framework=args.framework,
        security_mode=args.security_mode,
        transport_mode=args.transport_mode,
        principal_source=_principal_source(args.principal_source),
        principal_dependency=args.principal_dependency,
        principal_id_attr=args.principal_id_attr,
        principal_resolver=args.principal_resolver,
        base_url_env=args.base_url_env,
        request_overlay_source=args.request_overlay_source,
        auth_query_params=tuple(args.auth_query_param or ()),
        credential_headers=tuple(args.capture_credential_header or ()),
        credential_key_env=args.credential_key_env,
        credential_ttl_seconds=args.credential_ttl_seconds,
        explicit_env=args.env,
    )
    exit_code = 2 if result.is_blocked else 0
    return command_envelope_result(
        kind=FervisCommandKind.AUTH,
        command="auth.configure",
        project=project,
        payload_schema="fervis-auth-configure-result.v0.1",
        payload=result.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=exit_code,
        next_actions=[run_doctor_action()] if exit_code == 0 else [],
    )


def _principal_source(value: str | None) -> str | None:
    if value == "flask-login":
        return "flask_login_current_user"
    if value == "flask-g":
        return "flask_g"
    return value
