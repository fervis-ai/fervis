"""SQL-backed model-turn prompt capture query adapter."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    RunStepKind,
    RunStepKey,
)
from fervis.observability.prompt_captures import (
    ModelTurnPromptCapture,
    PromptCaptureArtifact,
    PromptCaptureQueryPort,
    PromptCaptureUsage,
)
from fervis.project.persistence.schema import metadata

from .rows import row_mappings
from .transaction import sql_connection


class SQLPromptCaptureQuery(PromptCaptureQueryPort):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def model_turn_prompt_captures_for_run(
        self, run_id: str
    ) -> tuple[ModelTurnPromptCapture, ...]:
        step = metadata.tables["fervis_run_step"]
        call = metadata.tables["fervis_model_call"]
        with sql_connection(self.engine) as connection:
            rows = row_mappings(
                connection.execute(
                    sa.select(
                        call,
                        step.c.sequence,
                        step.c.attempt,
                        step.c.step_key,
                        step.c.input_summary_json,
                        step.c.output_summary_json,
                        step.c.error_json,
                    )
                    .select_from(call.join(step, call.c.step_id == step.c.step_id))
                    .where(
                        call.c.run_id == run_id,
                        step.c.kind == RunStepKind.MODEL_TURN.value,
                    )
                    .order_by(step.c.sequence, call.c.call_index)
                ).all()
            )
            call_ids = tuple(str(row["model_call_id"]) for row in rows)
            artifacts_by_call = _artifacts_by_call(connection, call_ids)
            usage_by_call = _usage_by_call(connection, call_ids)
        return tuple(
            _capture_row(
                row,
                artifacts=artifacts_by_call.get(str(row["model_call_id"]), ()),
                usage_rows=usage_by_call.get(str(row["model_call_id"]), ()),
            )
            for row in rows
        )


def _artifacts_by_call(connection, call_ids: tuple[str, ...]):
    if not call_ids:
        return {}
    artifact = metadata.tables["fervis_run_artifact"]
    rows = row_mappings(
        connection.execute(
            sa.select(artifact)
            .where(artifact.c.model_call_id.in_(call_ids))
            .order_by(artifact.c.model_call_id, artifact.c.artifact_id)
        ).all()
    )
    grouped: dict[str, list[PromptCaptureArtifact]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_call_id"]), []).append(_artifact_row(row))
    return {key: tuple(value) for key, value in grouped.items()}


def _usage_by_call(connection, call_ids: tuple[str, ...]):
    if not call_ids:
        return {}
    usage = metadata.tables["fervis_model_call_usage"]
    rows = row_mappings(
        connection.execute(
            sa.select(usage)
            .where(usage.c.model_call_id.in_(call_ids))
            .order_by(usage.c.model_call_id, usage.c.usage_kind, usage.c.usage_id)
        ).all()
    )
    grouped: dict[str, list[PromptCaptureUsage]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_call_id"]), []).append(_usage_row(row))
    return {key: tuple(value) for key, value in grouped.items()}


def _capture_row(
    row: dict[str, object],
    *,
    artifacts: tuple[PromptCaptureArtifact, ...],
    usage_rows: tuple[PromptCaptureUsage, ...],
) -> ModelTurnPromptCapture:
    return ModelTurnPromptCapture(
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        attempt=row["attempt"],
        step_key=RunStepKey(row["step_key"]),
        call_index=int(row["call_index"]),
        provider=str(row["provider"]),
        model_key=str(row["model_key"]),
        status=ModelCallStatus(row["status"]),
        provider_request_id=str(row["provider_request_id"]),
        finish_reason=str(row["finish_reason"]),
        duration_ms=row["duration_ms"],
        prompt_chars=int(row["prompt_chars"]),
        schema_chars=int(row["schema_chars"]),
        tool_spec_chars=int(row["tool_spec_chars"]),
        submitted_payload_chars=row["submitted_payload_chars"],
        raw_output_chars=row["raw_output_chars"],
        step_input_summary=dict(row["input_summary_json"] or {}),
        step_output_summary=dict(row["output_summary_json"] or {}),
        error_json=dict(row["error_json"] or {}),
        artifacts=artifacts,
        usage_rows=usage_rows,
    )


def _artifact_row(row: dict[str, object]) -> PromptCaptureArtifact:
    return PromptCaptureArtifact(
        artifact_kind=ArtifactKind(row["artifact_kind"]),
        content=str(row["content"] or ""),
        content_type=str(row["content_type"]),
    )


def _usage_row(row: dict[str, object]) -> PromptCaptureUsage:
    return PromptCaptureUsage(
        usage_kind=ModelUsageKind(row["usage_kind"]),
        quantity=int(row["quantity"]),
        provider_usage_key=str(row["provider_usage_key"]),
    )
