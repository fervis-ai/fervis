"""`fervis goldset` command adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from fervis.evaluation.goldsets.loader import load_goldset_suite
from fervis.evaluation.goldsets.runner import run_goldset_suite
from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)
from fervis.interfaces.cli.principals import cli_question_principal


def goldset_result(
    args: argparse.Namespace,
    *,
    ports: FervisCliPorts,
) -> FervisCommandResult:
    suite = load_goldset_suite(args.suite_path)
    result = run_goldset_suite(
        suite,
        questions=ports.questions,
        question_run_follower=ports.question_run_follower,
        question_run_limits=ports.question_run_limits,
        model_policy=ports.model_policy,
        principal=cli_question_principal(
            tenant_id=args.tenant_id,
            principal_id=args.principal_id,
            project=ports.project,
        ),
        case_ids=_case_ids(args.case_ids),
        limit=args.limit,
        model_key=args.model_key,
        wait_seconds=args.wait_seconds,
        ledger_file=Path(args.ledger_file) if args.ledger_file else None,
    )
    return command_envelope_result(
        kind=FervisCommandKind.GOLDSET,
        command="goldset.run",
        project=ports.project,
        payload_schema="fervis-goldset-run.v0.1",
        payload=result.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=result.exit_code,
    )


def _case_ids(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(case_id.strip() for case_id in value.split(",") if case_id.strip())
