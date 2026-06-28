from __future__ import annotations

import json
from io import StringIO

from fervis.interfaces.cli.dispatch import run_fervis

from ._support import _ports


class _Worker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def process_once(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ):
        self.calls.append(
            {
                "worker_id": worker_id,
                "batch_size": batch_size,
                "lease_seconds": lease_seconds,
            }
        )
        return _Cycle()


class _Cycle:
    def to_payload(self) -> dict[str, object]:
        return {
            "claimed_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "queue_counts": {"QUEUED": 0, "RUNNING": 0, "COMPLETED": 1, "FAILED": 0},
            "runs": [
                {
                    "run_id": "run_1",
                    "active_attempt": 1,
                    "status": "COMPLETED",
                    "error": None,
                }
            ],
        }


def test_fervis_worker_once_returns_agent_cycle_envelope() -> None:
    stdout = StringIO()
    worker = _Worker()

    exit_code = run_fervis(
        (
            "worker",
            "--once",
            "--worker-id",
            "worker-1",
            "--batch-size",
            "5",
            "--lease-seconds",
            "30",
        ),
        ports=_ports(run_worker=worker),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["schema"] == "fervis-command-result.v0.1"
    assert envelope["command"] == "worker"
    assert envelope["status"] == "succeeded"
    assert envelope["payload_schema"] == "fervis-worker-cycle.v0.1"
    assert envelope["payload"] == _Cycle().to_payload()
    assert worker.calls == [
        {
            "worker_id": "worker-1",
            "batch_size": 5,
            "lease_seconds": 30,
        }
    ]


def test_fervis_worker_blocks_when_worker_port_is_not_configured() -> None:
    stdout = StringIO()

    exit_code = run_fervis(
        ("worker", "--once"),
        ports=_ports(),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["command"] == "worker"
    assert envelope["status"] == "blocked"
    assert envelope["payload"]["error"]["message"] == (
        "Fervis worker is not configured for this project."
    )
