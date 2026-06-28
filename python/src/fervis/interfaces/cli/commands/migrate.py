"""`fervis migrate` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.agent.actions import (
    edit_config_action,
    run_doctor_action,
)
from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)
from fervis.project import ProjectInspection
from fervis.project.configuration import (
    ConfigProblem,
    load_fervis_project_config,
)
from fervis.project.persistence import migrate_persistence


def migrate_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    del args
    loaded = load_fervis_project_config(project)
    if isinstance(loaded, ConfigProblem):
        return command_envelope_result(
            kind=FervisCommandKind.MIGRATE,
            command="migrate",
            project=project,
            payload_schema="fervis-command-error.v0.1",
            payload={
                "error": {
                    "code": loaded.code,
                    "message": loaded.message,
                    "retryable": False,
                }
            },
            view_kind=FervisViewKind.COMMAND,
            exit_code=2,
            next_actions=[edit_config_action()],
        )
    result = migrate_persistence(project, loaded)
    return command_envelope_result(
        kind=FervisCommandKind.MIGRATE,
        command="migrate",
        project=project,
        payload_schema="fervis-migration-result.v0.1",
        payload=result.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=result.exit_code,
        next_actions=[run_doctor_action()] if result.exit_code == 0 else [],
    )
