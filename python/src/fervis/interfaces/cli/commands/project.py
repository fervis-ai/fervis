"""`fervis project` command adapter."""

from __future__ import annotations

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)
from fervis.project import ProjectInspection


def project_inspect_result(project: ProjectInspection) -> FervisCommandResult:
    exit_code = 2 if project.is_blocked else 0
    return command_envelope_result(
        kind=FervisCommandKind.PROJECT_INSPECT,
        command="project.inspect",
        project=project,
        payload_schema="fervis-project-inspection.v0.1",
        payload=project.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=exit_code,
    )


def blocked_command_result(
    command: str,
    *,
    project: ProjectInspection,
    reason: str,
) -> FervisCommandResult:
    return command_envelope_result(
        kind=FervisCommandKind.BLOCKED,
        command=command,
        project=project,
        payload_schema="fervis-command-error.v0.1",
        payload={
            "error": {
                "code": "command_blocked",
                "message": reason,
                "retryable": False,
            }
        },
        view_kind=FervisViewKind.COMMAND,
        exit_code=2,
        next_actions=[
            {
                "kind": "retry",
                "description": (
                    "Fix the command input described in the error payload, then "
                    "run the command again."
                ),
            }
        ],
    )
