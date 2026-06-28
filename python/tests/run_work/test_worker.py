from __future__ import annotations

from dataclasses import dataclass

from fervis.run_work.worker import WorkerRunResult
from fervis.run_work.worker import process_run_work_batch


@dataclass(frozen=True)
class _WorkItem:
    run_id: str
    active_attempt: int


class _Queue:
    def __init__(self, items: list[_WorkItem]) -> None:
        self.items = items

    def claim_run_work_items(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ) -> list[_WorkItem]:
        assert worker_id == "worker-1"
        assert batch_size == 5
        assert lease_seconds == 30
        return self.items

    def queue_counts(self) -> dict[str, int]:
        return {"QUEUED": 0, "RUNNING": 0, "COMPLETED": 1, "FAILED": 1}


class _Processor:
    def __init__(self) -> None:
        self.processed: list[dict[str, object]] = []
        self.failed: list[dict[str, object]] = []

    def process_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> WorkerRunResult:
        self.processed.append(
            {
                "run_id": run_id,
                "worker_id": worker_id,
                "active_attempt": active_attempt,
            }
        )
        if run_id == "run_failed":
            raise RuntimeError("provider refused request")
        return WorkerRunResult(
            run_id=run_id,
            active_attempt=active_attempt,
            status="COMPLETED",
        )

    def fail_run_work(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
        error: str,
    ) -> WorkerRunResult:
        self.failed.append(
            {
                "run_id": run_id,
                "worker_id": worker_id,
                "active_attempt": active_attempt,
                "error": error,
            }
        )
        return WorkerRunResult(
            status="FAILED",
            run_id=run_id,
            active_attempt=active_attempt,
            error=error,
        )


def test_process_run_work_batch_processes_claimed_runs_and_terminalizes_failures():
    processor = _Processor()

    cycle = process_run_work_batch(
        worker_id="worker-1",
        batch_size=5,
        lease_seconds=30,
        work_queue=_Queue(
            [
                _WorkItem(run_id="run_completed", active_attempt=1),
                _WorkItem(run_id="run_failed", active_attempt=2),
            ]
        ),
        worker=processor,
    )

    assert cycle.to_payload() == {
        "claimed_count": 2,
        "completed_count": 1,
        "failed_count": 1,
        "queue_counts": {"QUEUED": 0, "RUNNING": 0, "COMPLETED": 1, "FAILED": 1},
        "runs": [
            {
                "run_id": "run_completed",
                "active_attempt": 1,
                "status": "COMPLETED",
                "error": None,
            },
            {
                "run_id": "run_failed",
                "active_attempt": 2,
                "status": "FAILED",
                "error": "provider refused request",
            },
        ],
    }
    assert processor.processed == [
        {"run_id": "run_completed", "worker_id": "worker-1", "active_attempt": 1},
        {"run_id": "run_failed", "worker_id": "worker-1", "active_attempt": 2},
    ]
    assert processor.failed == [
        {
            "run_id": "run_failed",
            "worker_id": "worker-1",
            "active_attempt": 2,
            "error": "provider refused request",
        }
    ]
