"""`fervis doctor` command adapter."""

from __future__ import annotations

import argparse
import json

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)
from fervis.project import (
    DoctorOptions,
    ProjectInspection,
    inspect_fervis_project,
)


def doctor_result(
    args: argparse.Namespace,
    *,
    project: ProjectInspection,
) -> FervisCommandResult:
    report = inspect_fervis_project(
        project,
        options=DoctorOptions(probe_read_context_key=args.probe_read_context_key),
    )
    exit_code = 2 if report.is_failed else 0
    return command_envelope_result(
        kind=FervisCommandKind.DOCTOR,
        command="doctor",
        project=project,
        payload_schema="fervis-doctor-report.v0.1",
        payload=report.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=exit_code,
        next_actions=_doctor_next_actions(report.to_payload()),
    )


def _doctor_next_actions(payload: dict[str, object]) -> list[dict[str, object]]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return []
    actions: list[dict[str, object]] = []
    seen: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            continue
        fix = check.get("fix")
        if isinstance(fix, dict):
            key = json.dumps(fix, sort_keys=True)
            if key not in seen:
                seen.add(key)
                actions.append(fix)
    return actions
