"""`fervis worker` command adapter."""

from __future__ import annotations

import argparse

from fervis.interfaces.cli.commands.common import command_envelope_result
from fervis.interfaces.cli.contracts import (
    FervisCliPorts,
    FervisCommandKind,
    FervisCommandResult,
    FervisViewKind,
)


def worker_result(
    args: argparse.Namespace,
    *,
    ports: FervisCliPorts,
) -> FervisCommandResult:
    if ports.run_worker is None:
        raise ValueError("Fervis worker is not configured for this project.")
    cycle = ports.run_worker.process_once(
        worker_id=args.worker_id,
        batch_size=max(1, int(args.batch_size)),
        lease_seconds=max(1, int(args.lease_seconds)),
    )
    return command_envelope_result(
        kind=FervisCommandKind.WORKER,
        command="worker",
        project=ports.project,
        payload_schema="fervis-worker-cycle.v0.1",
        payload=cycle.to_payload(),
        view_kind=FervisViewKind.COMMAND,
        exit_code=0,
    )
