"""Queued run work request/result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class QueuedRunRequest:
    run_id: str
    worker_id: str = ""
    active_attempt: int | None = None


@dataclass(frozen=True)
class FailQueuedRunRequest:
    run_id: str
    error: str
    worker_id: str = ""
    active_attempt: int | None = None


@dataclass(frozen=True)
class QueuedRunResult:
    status: str
    run_id: str
    answer: str | None = None
    result_data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None


def run_wall_clock_duration_ms(
    *,
    created_at: datetime,
    completed_at: datetime | None,
) -> int | None:
    if completed_at is None:
        return None
    elapsed = completed_at - created_at
    return max(0, int(elapsed.total_seconds() * 1000))
