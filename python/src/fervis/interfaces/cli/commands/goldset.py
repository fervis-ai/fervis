"""`fervis goldset` command adapter."""

from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
import sys
from typing import TextIO

from fervis.evaluation.goldsets.contracts import GoldsetCaseResult
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
    progress_stream: TextIO = sys.stderr,
) -> FervisCommandResult:
    suite = load_goldset_suite(_suite_ref(args))
    principal_id = _principal_id(args)
    result = run_goldset_suite(
        suite,
        questions=ports.questions,
        question_run_follower=ports.question_run_follower,
        question_run_limits=ports.question_run_limits,
        model_policy=ports.model_policy,
        principal=cli_question_principal(
            tenant_id=_tenant_id(args),
            principal_id=principal_id,
            project=ports.project,
        ),
        case_ids=_case_ids(args),
        limit=args.limit,
        model_key=args.model_key,
        wait_seconds=args.wait_seconds,
        ledger_file=Path(args.ledger_file) if args.ledger_file else None,
        stable_runs=args.stable_runs,
        enforce_structured_determinism=args.enforce_structured_determinism,
        attempts=args.attempts,
        retry_provider_failures=args.retry_provider_failures,
        retry_sleep_seconds=args.retry_sleep_seconds,
        case_run_observer=(
            partial(_write_stability_run, stream=progress_stream)
            if args.stable_runs > 1
            else None
        ),
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


def _write_stability_run(
    run_number: int,
    run_count: int,
    result: GoldsetCaseResult,
    *,
    stream: TextIO,
) -> None:
    status = "PASS" if result.status == "passed" else "FAIL"
    answer = result.answer or result.message
    duration = ""
    if result.duration_ms is not None:
        duration = f" | Completed in {result.duration_ms / 1000:.2f} sec"
    stream.write(
        f"STABILITY RUN {run_number}/{run_count} {status} "
        f"(run_id: {result.run_id or 'unavailable'}): {answer}{duration}\n"
    )
    stream.flush()


def _suite_ref(args: argparse.Namespace) -> str:
    value = str(args.suite_path or os.environ.get("FERVIS_GOLDSET_SUITE") or "").strip()
    if not value:
        raise ValueError(
            "goldset run requires --suite-path, --suite, or FERVIS_GOLDSET_SUITE"
        )
    return value


def _case_ids(args: argparse.Namespace) -> tuple[str, ...]:
    value = args.case_ids or os.environ.get("FERVIS_GOLDSET_CASE_IDS")
    if not value:
        return ()
    return tuple(case_id.strip() for case_id in value.split(",") if case_id.strip())


def _tenant_id(args: argparse.Namespace) -> str:
    value = str(
        args.tenant_id or os.environ.get("FERVIS_GOLDSET_TENANT_ID") or ""
    ).strip()
    if not value:
        raise ValueError("goldset run requires --tenant-id or FERVIS_GOLDSET_TENANT_ID")
    return value


def _principal_id(args: argparse.Namespace) -> str:
    value = str(
        args.principal_id or os.environ.get("FERVIS_GOLDSET_PRINCIPAL_ID") or ""
    ).strip()
    if not value:
        raise ValueError(
            "goldset run requires --principal-id or FERVIS_GOLDSET_PRINCIPAL_ID"
        )
    return value
