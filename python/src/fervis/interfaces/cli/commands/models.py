"""`fervis models` and `fervis model ...` command adapters."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import project_command_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
)
from fervis.project import ProjectInspection
from fervis.project.model_commands import allow_model, models_view, use_model


def models_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    if args.command == FervisCommandKind.MODELS:
        result = models_view(project)
        return project_command_result(
            kind=FervisCommandKind.MODELS,
            command="models",
            project=project,
            payload_schema="fervis-models-view.v0.1",
            payload=result.payload,
            exit_code=2 if result.is_blocked else 0,
        )
    if args.command == FervisCommandKind.MODEL and args.model_command in {
        "allow",
        "use",
    }:
        if args.model_command == "allow":
            edit_result = allow_model(project, args.model_ref)
        else:
            edit_result = use_model(project, args.model_ref, explicit_env=args.env)
        return project_command_result(
            kind=FervisCommandKind.MODEL,
            command=f"model.{args.model_command}",
            project=project,
            payload_schema="fervis-config-edit-result.v0.1",
            payload=edit_result.to_payload(),
            exit_code=2 if edit_result.is_blocked else 0,
        )
    raise ValueError(f"unsupported model command: {args.command}")
