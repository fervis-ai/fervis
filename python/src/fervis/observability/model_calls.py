"""Model-call inspection over canonical lineage rows."""

from __future__ import annotations

from fervis.lineage.enums import RunStepKey
from fervis.observability.query import (
    ObservabilityModelCall,
    ObservabilityQueryPort,
)


class ModelCallInspectionService:
    def __init__(self, query: ObservabilityQueryPort) -> None:
        self._query = query

    def for_run(
        self, run_id: str, *, step_key: RunStepKey | None = None
    ) -> tuple[ObservabilityModelCall, ...]:
        calls = self._query.model_calls_for_run(run_id, step_key=step_key)
        return tuple(
            sorted(calls, key=lambda call: (call.step_sequence, call.call_index))
        )
