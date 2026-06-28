"""`fervis sources` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import project_command_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
)
from fervis.interfaces.cli.parsers import comma_list
from fervis.project import ProjectInspection
from fervis.project.config_commands import (
    add_django_source,
    add_fastapi_source,
    add_flask_source,
)


def sources_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    if args.source_kind == "django-app":
        result = add_django_source(
            project,
            name=args.name,
            app_modules=comma_list(args.app_modules),
            path_prefixes=comma_list(args.path_prefixes),
        )
    elif args.source_kind == "fastapi-app":
        result = add_fastapi_source(
            project,
            name=args.name,
            import_paths=comma_list(args.import_paths),
            path_prefixes=comma_list(args.path_prefixes),
        )
    elif args.source_kind == "flask-app":
        result = add_flask_source(
            project,
            name=args.name,
            app=args.app,
            path_prefixes=tuple(args.source_prefix),
            blueprints=tuple(args.blueprint),
        )
    else:
        raise ValueError(f"unsupported source kind: {args.source_kind}")
    return project_command_result(
        kind=FervisCommandKind.SOURCES,
        command="sources.add",
        project=project,
        payload_schema="fervis-source-edit-result.v0.1",
        payload=result.to_payload(),
        exit_code=2 if result.is_blocked else 0,
    )
