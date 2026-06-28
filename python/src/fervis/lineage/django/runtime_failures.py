"""Lineage terminal writers for interface and queue infrastructure failures."""

from __future__ import annotations

from fervis.lineage import models
from fervis.lineage.django.recorder import DjangoLineageRecorder
from fervis.lineage.enums import RunResultKind, RuntimeErrorKind
from fervis.lineage.ids import lineage_id
from fervis.lineage.recorder import (
    RunResultWrite,
    RuntimeErrorResultWrite,
    RuntimeErrorWrite,
)


def record_worker_runtime_error(
    *,
    run_id: str,
    error_code: str,
    message: str,
) -> None:
    if models.RunResult.objects.filter(run_id=run_id).exists():
        return
    result = RunResultWrite(
        run_result_id=lineage_id("run_result", run_id, "runtime_error"),
        run_id=run_id,
        result_kind=RunResultKind.RUNTIME_ERROR,
    )
    DjangoLineageRecorder().record_runtime_error_result(
        RuntimeErrorResultWrite(
            result=result,
            error=RuntimeErrorWrite(
                runtime_error_detail_id=lineage_id(
                    "runtime_error",
                    run_id,
                    error_code,
                ),
                run_id=run_id,
                run_result_id=result.run_result_id,
                failed_step_id=None,
                error_kind=_runtime_error_kind(error_code),
                message=message,
            ),
        )
    )


def _runtime_error_kind(error_code: str) -> RuntimeErrorKind:
    try:
        return RuntimeErrorKind(error_code)
    except ValueError:
        return RuntimeErrorKind.INFRASTRUCTURE_FAILED
