"""Shared command-envelope helpers for CLI command adapters."""

from __future__ import annotations

from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisOutputFormat,
    FervisRenderOptions,
    FervisViewKind,
)
from fervis.interfaces.cli.envelope import (
    command_envelope,
    status_for_exit_code,
)
from fervis.project import ProjectInspection


def command_envelope_result(
    *,
    kind: FervisCommandKind,
    command: str,
    project: ProjectInspection,
    payload_schema: str,
    payload: object,
    view_kind: FervisViewKind,
    exit_code: int = 0,
    next_actions: list[dict[str, object]] | None = None,
) -> FervisCommandResult:
    return FervisCommandResult(
        kind=kind,
        payload=command_envelope(
            command=command,
            status=status_for_exit_code(exit_code),
            exit_code=exit_code,
            project=project,
            next_actions=next_actions or [],
            payload_schema=payload_schema,
            payload=payload,
        ),
        view_kind=view_kind,
        render_options=FervisRenderOptions(output_format=FervisOutputFormat.AGENT),
    )


def project_command_result(
    *,
    kind: FervisCommandKind,
    command: str,
    project: ProjectInspection,
    payload_schema: str,
    payload: object,
    exit_code: int = 0,
) -> FervisCommandResult:
    return command_envelope_result(
        kind=kind,
        command=command,
        project=project,
        payload_schema=payload_schema,
        payload=payload,
        view_kind=FervisViewKind.COMMAND,
        exit_code=exit_code,
    )
