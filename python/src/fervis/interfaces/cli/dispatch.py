"""Fervis command-line surface.

The CLI is an adapter. It parses user intent, calls framework-neutral services,
and renders service views. It does not read ORM models directly.
"""

from __future__ import annotations

import sys
import argparse
import json
import time
from typing import Callable, TextIO

from fervis.interfaces.cli.commands.auth import auth_result
from fervis.interfaces.cli.commands.catalog import catalog_result
from fervis.interfaces.cli.commands.config import config_result
from fervis.interfaces.cli.commands.doctor import doctor_result
from fervis.interfaces.cli.commands.debug import (
    debug_artifact_result,
    debug_prompts_result,
    debug_result,
)
from fervis.interfaces.cli.commands.explain import explain_result
from fervis.interfaces.cli.commands.goldset import goldset_result
from fervis.interfaces.cli.commands.init import init_result
from fervis.interfaces.cli.commands.migrate import migrate_result
from fervis.interfaces.cli.commands.models import models_result
from fervis.interfaces.cli.commands.project import (
    blocked_command_result,
    project_inspect_result,
)
from fervis.interfaces.cli.commands.runtime import runtime_ask_result
from fervis.interfaces.cli.commands.sources import sources_result
from fervis.interfaces.cli.commands.usage import usage_result
from fervis.interfaces.cli.commands.worker import worker_result
from fervis.interfaces.cli.envelope import (
    CommandEnvelope,
)
from fervis.lineage.views.service import (
    LineageRootNotFound,
)
from fervis.observability.usage import (
    ObservabilityRootNotFound,
)
from fervis.interfaces.cli.runtime_ask import (
    RuntimeAskEventStream,
    RuntimeAskJsonlSink,
)
from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
)
from fervis.interfaces.cli.rendering import (
    render_fervis_result,
)
from fervis.interfaces.cli.parsers import (
    auth_parser,
    catalog_parser,
    command_name,
    config_parser,
    doctor_parser,
    init_parser,
    is_runtime_ask_argv,
    is_worker_argv,
    migrate_parser,
    models_parser,
    parser,
    project_parser,
    sources_parser,
)
from fervis.project import (
    ProjectInspection,
)


def run_fervis(
    argv: tuple[str, ...],
    *,
    ports: FervisCliPorts,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    try:
        if is_worker_argv(argv):
            return _run_fervis_worker(
                argv,
                ports=ports,
                stdout=stdout,
            )
        if is_runtime_ask_argv(argv):
            return _run_fervis_runtime_ask(
                argv,
                ports=ports,
                stdout=stdout,
            )
        result = evaluate_fervis(argv, ports=ports, stderr=stderr)
    except (LineageRootNotFound, ObservabilityRootNotFound, ValueError) as error:
        del stderr
        result = blocked_command_result(
            command_name(argv),
            project=ports.project,
            reason=str(error),
        )
    rendered = render_fervis_result(result)
    if rendered:
        stdout.write(rendered)
        stdout.write("\n")
    if isinstance(result.payload, CommandEnvelope):
        return result.payload.exit_code
    if isinstance(result.payload, RuntimeAskEventStream):
        return result.payload.exit_code
    if isinstance(result.payload, int):
        return result.payload
    return 0


def _run_fervis_runtime_ask(
    argv: tuple[str, ...],
    *,
    ports: FervisCliPorts,
    stdout: TextIO,
) -> int:
    args = parser().parse_args(argv)
    writer = RuntimeAskJsonlSink(
        stdout,
        tenant_id=args.tenant_id,
        principal_id=args.principal_id,
    )
    result = runtime_ask_result(args, ports=ports, event_sink=writer)
    if isinstance(result.payload, RuntimeAskEventStream):
        return result.payload.exit_code
    return 0


def _run_fervis_worker(
    argv: tuple[str, ...],
    *,
    ports: FervisCliPorts,
    stdout: TextIO,
) -> int:
    args = parser().parse_args(argv)
    try:
        while True:
            result = worker_result(args, ports=ports)
            rendered = _render_worker_cycle(result, pretty=bool(args.once))
            if rendered:
                stdout.write(rendered)
                stdout.write("\n")
                stdout.flush()
            if args.once:
                if isinstance(result.payload, CommandEnvelope):
                    return result.payload.exit_code
                return 0
            time.sleep(max(0.1, float(args.sleep_seconds)))
    except KeyboardInterrupt:
        return 0


def _render_worker_cycle(result: FervisCommandResult, *, pretty: bool) -> str:
    if pretty:
        return render_fervis_result(result)
    if isinstance(result.payload, CommandEnvelope):
        return json.dumps(result.payload.to_payload(), sort_keys=True)
    return render_fervis_result(result)


def run_project_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    args = project_parser().parse_args(argv)
    if args.project_command == "inspect":
        return run_project_inspect(project=project, stdout=stdout)
    raise ValueError(f"unsupported project command: {args.project_command}")


def run_init_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=init_parser().parse_args,
        build_result=init_result,
    )


def run_catalog_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=catalog_parser().parse_args,
        build_result=catalog_result,
    )


def run_doctor_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=doctor_parser().parse_args,
        build_result=doctor_result,
    )


def run_migrate_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=migrate_parser().parse_args,
        build_result=migrate_result,
    )


def run_auth_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=auth_parser().parse_args,
        build_result=auth_result,
    )


def run_models_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=models_parser().parse_args,
        build_result=models_result,
    )


def run_config_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=config_parser().parse_args,
        build_result=config_result,
    )


def run_sources_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    return _run_project_result_command(
        argv,
        project=project,
        stdout=stdout,
        parse=sources_parser().parse_args,
        build_result=sources_result,
    )


def run_help(argv: tuple[str, ...]) -> int:
    parser().parse_args(argv)
    return 0


def run_project_inspect(
    *,
    project: ProjectInspection,
    stdout: TextIO = sys.stdout,
) -> int:
    result = project_inspect_result(project)
    return _write_command_result(result, stdout=stdout)


def run_blocked_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    reason: str,
    stdout: TextIO = sys.stdout,
) -> int:
    result = blocked_command_result(command_name(argv), project=project, reason=reason)
    return _write_command_result(result, stdout=stdout)


def _run_project_result_command(
    argv: tuple[str, ...],
    *,
    project: ProjectInspection,
    stdout: TextIO,
    parse: Callable[[tuple[str, ...]], argparse.Namespace],
    build_result: Callable[..., FervisCommandResult],
) -> int:
    args = parse(argv)
    result = build_result(args, project=project)
    return _write_command_result(result, stdout=stdout)


def _write_command_result(result: FervisCommandResult, *, stdout: TextIO) -> int:
    rendered = render_fervis_result(result)
    if rendered:
        stdout.write(rendered)
        stdout.write("\n")
    if not isinstance(result.payload, CommandEnvelope):
        raise ValueError("project command result requires a command envelope")
    return result.payload.exit_code


def evaluate_fervis(
    argv: tuple[str, ...],
    *,
    ports: FervisCliPorts,
    stderr: TextIO = sys.stderr,
) -> FervisCommandResult:
    args = parser().parse_args(argv)
    return execute_fervis(args, ports=ports, stderr=stderr)


def execute_fervis(
    args: argparse.Namespace,
    *,
    ports: FervisCliPorts,
    stderr: TextIO = sys.stderr,
) -> FervisCommandResult:
    if args.command == FervisCommandKind.INIT:
        return init_result(args, project=ports.project)
    if args.command == FervisCommandKind.CATALOG:
        return catalog_result(args, project=ports.project)
    if args.command == FervisCommandKind.DOCTOR:
        return doctor_result(args, project=ports.project)
    if args.command == FervisCommandKind.MIGRATE:
        return migrate_result(args, project=ports.project)
    if args.command == FervisCommandKind.AUTH:
        return auth_result(args, project=ports.project)
    if args.command in {FervisCommandKind.MODEL, FervisCommandKind.MODELS}:
        return models_result(args, project=ports.project)
    if args.command == FervisCommandKind.CONFIG:
        return config_result(args, project=ports.project)
    if args.command == FervisCommandKind.DEBUG:
        if args.debug_command == "prompts":
            return debug_prompts_result(args, ports=ports)
        if args.debug_command == "artifact":
            return debug_artifact_result(args, ports=ports)
        return debug_result(args, ports=ports)
    if args.command == FervisCommandKind.EXPLAIN:
        return explain_result(args, ports=ports)
    if args.command == FervisCommandKind.GOLDSET:
        return goldset_result(args, ports=ports, progress_stream=stderr)
    if args.command == "project" and args.project_command == "inspect":
        return project_inspect_result(ports.project)
    if args.command == FervisCommandKind.SOURCES:
        return sources_result(args, project=ports.project)
    if args.command == "runtime" and args.runtime_command == "ask":
        return runtime_ask_result(args, ports=ports)
    if args.command == FervisCommandKind.USAGE:
        return usage_result(args, ports=ports)
    if args.command == FervisCommandKind.WORKER:
        return worker_result(args, ports=ports)
    raise ValueError(f"unsupported command: {args.command}")
