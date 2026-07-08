"""Terminal lineage helpers for SQL-backed runtime storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.lineage.enums import FactResultKind, RunResultKind, RuntimeErrorKind
from fervis.lineage.ids import lineage_id
from fervis.lineage.recorder import (
    RunResultWrite,
    RuntimeErrorResultWrite,
    RuntimeErrorWrite,
)
from fervis.lineage.recorder_core import LineageRecorder
from fervis.project.persistence.schema import metadata

from .lineage_store import SQLLineageRecorderStore
from .rows import row_mapping
from .transaction import sql_connection


@dataclass(frozen=True)
class TerminalResult:
    status: str
    answer: str | None
    result_data: dict[str, Any]
    error: str | None


def run_has_terminal_result(engine: Engine, run_id: str) -> bool:
    run_result = metadata.tables["fervis_run_result"]
    with sql_connection(engine) as connection:
        return (
            connection.execute(
                sa.select(run_result.c.run_result_id).where(
                    run_result.c.run_id == run_id
                )
            ).first()
            is not None
        )


def terminal_result_for_run(engine: Engine, run_id: str) -> TerminalResult | None:
    run_result = metadata.tables["fervis_run_result"]
    runtime_error = metadata.tables["fervis_runtime_error_detail"]
    answer = metadata.tables["fervis_answer"]
    presentation = metadata.tables["fervis_answer_presentation"]
    fact_result = metadata.tables["fervis_fact_result"]
    with sql_connection(engine) as connection:
        result = connection.execute(
            sa.select(run_result).where(run_result.c.run_id == run_id)
        ).first()
        if result is None:
            return None
        result_values = row_mapping(result)
        error_row = connection.execute(
            sa.select(runtime_error.c.message).where(runtime_error.c.run_id == run_id)
        ).first()
        answer_text = connection.execute(
            sa.select(presentation.c.rendered_value)
            .select_from(
                presentation.join(
                    answer,
                    presentation.c.answer_id == answer.c.answer_id,
                )
            )
            .where(
                answer.c.run_id == run_id, presentation.c.rendered_value.is_not(None)
            )
            .order_by(presentation.c.created_at)
        ).scalar()
        result_data = _terminal_result_data(
            connection,
            fact_result=fact_result,
            run_id=run_id,
            run_result_id=str(result_values["run_result_id"]),
        )
    error_values = row_mapping(error_row) if error_row is not None else {}
    return TerminalResult(
        status=_terminal_status(str(result_values["result_kind"])),
        answer=answer_text,
        result_data=result_data,
        error=str(error_values["message"]) if error_values else None,
    )


def _terminal_result_data(
    connection,
    *,
    fact_result,
    run_id: str,
    run_result_id: str,
) -> dict[str, Any]:
    clarification = connection.execute(
        sa.select(fact_result.c.payload_json)
        .where(
            fact_result.c.run_id == run_id,
            fact_result.c.result_kind == FactResultKind.NEEDS_CLARIFICATION.value,
            fact_result.c.payload_json.is_not(None),
        )
        .order_by(fact_result.c.created_at)
    ).scalar()
    clarification_payload = _dict(clarification)
    if clarification_payload:
        return {
            "kind": "needs_clarification",
            "details": clarification_payload,
        }
    return {"run_result_id": run_result_id}


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = json.loads(value)
        return raw if isinstance(raw, dict) else {}
    return {}


def record_runtime_error_result(
    *,
    engine: Engine,
    run_id: str,
    error_code: str,
) -> None:
    if run_has_terminal_result(engine, run_id):
        return
    result = RunResultWrite(
        run_result_id=lineage_id("run_result", run_id, "runtime_error"),
        run_id=run_id,
        result_kind=RunResultKind.RUNTIME_ERROR,
    )
    LineageRecorder(SQLLineageRecorderStore(engine)).record_runtime_error_result(
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
                message=error_code,
            ),
        )
    )


def _runtime_error_kind(error_code: str) -> RuntimeErrorKind:
    try:
        return RuntimeErrorKind(error_code)
    except ValueError:
        return RuntimeErrorKind.INFRASTRUCTURE_FAILED


def _terminal_status(result_kind: str) -> str:
    if result_kind == RunResultKind.RUNTIME_ERROR.value:
        return "FAILED"
    if result_kind == RunResultKind.FACTUAL_TERMINAL.value:
        return "NEEDS_CLARIFICATION"
    if result_kind == RunResultKind.ANSWERED.value:
        return "COMPLETED"
    return "RUNNING"
