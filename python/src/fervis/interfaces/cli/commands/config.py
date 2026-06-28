"""`fervis config` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import project_command_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
)
from fervis.project import ProjectInspection
from fervis.project.config_commands import (
    config_get,
    config_set,
    config_show,
    config_upgrade,
)


def config_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    if args.config_command == "show":
        result = config_show(project, explicit_env=args.env)
        return project_command_result(
            kind=FervisCommandKind.CONFIG,
            command="config.show",
            project=project,
            payload_schema="fervis-config-view.v0.1",
            payload=result.payload,
            exit_code=2 if result.is_blocked else 0,
        )
    if args.config_command == "get":
        result = config_get(project, args.path, explicit_env=args.env)
        return project_command_result(
            kind=FervisCommandKind.CONFIG,
            command="config.get",
            project=project,
            payload_schema="fervis-config-value.v0.1",
            payload=result.payload,
            exit_code=2 if result.is_blocked else 0,
        )
    if args.config_command == "set":
        result = config_set(project, args.path, args.value, explicit_env=args.env)
        return project_command_result(
            kind=FervisCommandKind.CONFIG,
            command="config.set",
            project=project,
            payload_schema="fervis-config-edit-result.v0.1",
            payload=result.to_payload(),
            exit_code=2 if result.is_blocked else 0,
        )
    if args.config_command == "upgrade":
        result = config_upgrade(project)
        return project_command_result(
            kind=FervisCommandKind.CONFIG,
            command="config.upgrade",
            project=project,
            payload_schema="fervis-config-upgrade-result.v0.1",
            payload=result.to_payload(),
            exit_code=2 if result.is_blocked else 0,
        )
    raise ValueError(f"unsupported config command: {args.config_command}")
