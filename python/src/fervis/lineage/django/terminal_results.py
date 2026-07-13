"""Terminal run-result reads for Django-backed lineage."""

from __future__ import annotations

from fervis.lineage import models
from fervis.lineage.enums import RunResultKind


def run_has_terminal_result(run_id: str, *, tenant_id: str | None = None) -> bool:
    return _run_result(run_id, tenant_id=tenant_id) is not None


def terminal_status_for_run(run_id: str, *, tenant_id: str | None = None) -> str | None:
    result = _run_result(run_id, tenant_id=tenant_id)
    if result is None:
        return None
    return terminal_status_from_result_kind(result.result_kind)


def terminal_status_from_result_kind(result_kind: str) -> str:
    if result_kind == RunResultKind.RUNTIME_ERROR.value:
        return "FAILED"
    if result_kind == RunResultKind.FACTUAL_TERMINAL.value:
        return "COMPLETED"
    if result_kind == RunResultKind.ANSWERED.value:
        return "COMPLETED"
    return "RUNNING"


def _run_result(run_id: str, *, tenant_id: str | None):
    rows = models.RunResult.objects.filter(run_id=run_id)
    if tenant_id is not None:
        rows = rows.filter(run__question__conversation__tenant_id=tenant_id)
    return rows.first()
