"""`fervis catalog` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)
from fervis.project import ProjectInspection
from fervis.project.catalog_command import catalog_view


def catalog_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    del args
    result = catalog_view(project)
    return command_envelope_result(
        kind=FervisCommandKind.CATALOG,
        command="catalog",
        project=project,
        payload_schema=result.payload_schema,
        payload=result.payload,
        view_kind=FervisViewKind.COMMAND,
        exit_code=result.exit_code,
        next_actions=result.next_actions,
    )
