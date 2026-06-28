"""`fervis init` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.agent.actions import (
    resolve_blocked_edits_action,
    run_doctor_action,
)
from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import FervisCommandKind, FervisViewKind
from fervis.interfaces.cli.parsers import comma_list
from fervis.project import ProjectInspection, initialize_fervis_project


def init_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
):
    path_prefixes = (
        tuple(args.source_prefix)
        if getattr(args, "source_prefix", None)
        else comma_list(args.path_prefixes)
        if args.path_prefixes
        else None
    )
    result = initialize_fervis_project(
        project,
        requested_framework=args.framework,
        yes=bool(args.yes),
        app_factory=args.app_factory,
        app_target=args.app,
        path_prefixes=path_prefixes,
        blueprints=tuple(args.blueprint or ()),
    )
    exit_code = 2 if result.is_blocked else 0
    next_actions = (
        [resolve_blocked_edits_action()] if result.is_blocked else [run_doctor_action()]
    )
    return command_envelope_result(
        kind=FervisCommandKind.INIT,
        command="init",
        project=result.project,
        payload_schema="fervis-init-result.v0.1",
        payload=result.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=exit_code,
        next_actions=next_actions,
    )
