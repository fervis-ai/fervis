"""SQL-backed observability query adapter."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
)
from fervis.observability.query import (
    ModelCallDetailLevel,
    ObservabilityArtifact,
    ObservabilityArtifactContent,
    ObservabilityModelCall,
    ObservabilityQueryPort,
    ObservabilityRun,
    ObservabilityUsage,
)
from fervis.project.persistence.schema import metadata

from .rows import (
    json_object,
    json_objects,
    optional_int,
    optional_text,
    required_int,
    row_mapping,
    row_mappings,
)
from .transaction import sql_connection


class SQLObservabilityQuery(ObservabilityQueryPort):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def run_id_for_answer(self, answer_id: str) -> str | None:
        answer = metadata.tables["fervis_answer"]
        with sql_connection(self.engine) as connection:
            value = connection.execute(
                sa.select(answer.c.run_id).where(answer.c.answer_id == answer_id)
            ).scalar()
        return str(value) if value is not None else None

    def run_by_id(self, run_id: str) -> ObservabilityRun | None:
        run = metadata.tables["fervis_question_run"]
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                sa.select(
                    run.c.run_id,
                    run.c.base_run_id,
                ).where(run.c.run_id == run_id)
            ).first()
        if row is None:
            return None
        values = row_mapping(row)
        return ObservabilityRun(
            run_id=str(values["run_id"]),
            base_run_id=values["base_run_id"],
        )

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        run = metadata.tables["fervis_question_run"]
        return _string_column(
            self.engine,
            sa.select(run.c.run_id).where(run.c.run_id == run_id),
        )

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        run = metadata.tables["fervis_question_run"]
        return _string_column(
            self.engine,
            sa.select(run.c.run_id)
            .where(run.c.question_id == question_id)
            .order_by(run.c.run_number),
        )

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        question = metadata.tables["fervis_question"]
        run = metadata.tables["fervis_question_run"]
        return _string_column(
            self.engine,
            sa.select(run.c.run_id)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id)
            )
            .where(question.c.conversation_id == conversation_id)
            .order_by(question.c.conversation_sequence, run.c.run_number),
        )

    def model_calls_for_run_ids(
        self,
        run_ids: tuple[str, ...],
        *,
        detail: ModelCallDetailLevel = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]:
        if not run_ids:
            return ()
        model_call = metadata.tables["fervis_model_call"]
        run_step = metadata.tables["fervis_run_step"]
        return self._model_calls(
            sa.select(model_call, run_step.c.step_key, run_step.c.sequence)
            .select_from(
                model_call.join(run_step, model_call.c.step_id == run_step.c.step_id)
            )
            .where(model_call.c.run_id.in_(run_ids))
            .order_by(
                model_call.c.run_id, run_step.c.sequence, model_call.c.call_index
            ),
            detail=detail,
        )

    def model_calls_for_run(
        self,
        run_id: str,
        step_key: RunStepKey | None = None,
        *,
        detail: ModelCallDetailLevel = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]:
        model_call = metadata.tables["fervis_model_call"]
        run_step = metadata.tables["fervis_run_step"]
        statement = (
            sa.select(model_call, run_step.c.step_key, run_step.c.sequence)
            .select_from(
                model_call.join(run_step, model_call.c.step_id == run_step.c.step_id)
            )
            .where(model_call.c.run_id == run_id)
        )
        if step_key is not None:
            statement = statement.where(run_step.c.step_key == step_key.value)
        return self._model_calls(
            statement.order_by(run_step.c.sequence, model_call.c.call_index),
            detail=detail,
        )

    def artifact_content(self, artifact_id: str) -> ObservabilityArtifactContent | None:
        artifact = metadata.tables["fervis_run_artifact"]
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                sa.select(artifact).where(artifact.c.artifact_id == artifact_id)
            ).first()
        if row is None:
            return None
        values = row_mapping(row)
        return ObservabilityArtifactContent(
            artifact_id=str(values["artifact_id"]),
            artifact_kind=ArtifactKind(values["artifact_kind"]),
            content_hash=str(values["content_hash"]),
            content_type=str(values["content_type"]),
            size_bytes=int(values["size_bytes"]),
            content=values["content"],
            storage_ref=values["storage_ref"],
        )

    def _model_calls(
        self,
        statement,
        *,
        detail: ModelCallDetailLevel,
    ) -> tuple[ObservabilityModelCall, ...]:
        with sql_connection(self.engine) as connection:
            rows = row_mappings(connection.execute(statement).all())
            call_ids = tuple(str(row["model_call_id"]) for row in rows)
            usage_by_call = _usage_by_call(connection, call_ids)
            artifacts_by_call = (
                _artifacts_by_call(connection, call_ids)
                if detail == "inspection"
                else {}
            )
        return tuple(
            _model_call_row(
                row,
                usage_rows=usage_by_call.get(str(row["model_call_id"]), ()),
                artifacts=artifacts_by_call.get(str(row["model_call_id"]), ()),
            )
            for row in rows
        )


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
    grouped: dict[str, list[ObservabilityUsage]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_call_id"]), []).append(_usage_row(row))
    return {key: tuple(value) for key, value in grouped.items()}


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
    grouped: dict[str, list[ObservabilityArtifact]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_call_id"]), []).append(_artifact_row(row))
    return {key: tuple(value) for key, value in grouped.items()}


def _model_call_row(
    row: dict[str, object],
    *,
    usage_rows: tuple[ObservabilityUsage, ...],
    artifacts: tuple[ObservabilityArtifact, ...],
) -> ObservabilityModelCall:
    return ObservabilityModelCall(
        model_call_id=str(row["model_call_id"]),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        step_key=RunStepKey(row["step_key"]),
        step_sequence=required_int(row["sequence"], field="sequence"),
        call_index=required_int(row["call_index"], field="call_index"),
        provider=str(row["provider"]),
        model_key=str(row["model_key"]),
        status=ModelCallStatus(row["status"]),
        provider_request_id=str(row["provider_request_id"]),
        finish_reason=str(row["finish_reason"]),
        duration_ms=optional_int(row["duration_ms"], field="duration_ms"),
        reasoning_effort=str(row["reasoning_effort"]),
        reasoning_budget_tokens=optional_int(
            row["reasoning_budget_tokens"], field="reasoning_budget_tokens"
        ),
        max_output_tokens=optional_int(
            row["max_output_tokens"], field="max_output_tokens"
        ),
        prompt_chars=required_int(row["prompt_chars"], field="prompt_chars"),
        schema_chars=required_int(row["schema_chars"], field="schema_chars"),
        tool_spec_chars=required_int(
            row["tool_spec_chars"], field="tool_spec_chars"
        ),
        submitted_payload_chars=optional_int(
            row["submitted_payload_chars"], field="submitted_payload_chars"
        ),
        raw_output_chars=optional_int(
            row["raw_output_chars"], field="raw_output_chars"
        ),
        model_subcalls=json_objects(
            row["model_subcalls_json"], field="model_subcalls_json"
        ),
        usage_rows=usage_rows,
        artifacts=artifacts,
    )


def _usage_row(row: dict[str, object]) -> ObservabilityUsage:
    return ObservabilityUsage(
        usage_kind=ModelUsageKind(row["usage_kind"]),
        quantity=required_int(row["quantity"], field="quantity"),
        unit=ModelUsageUnit(row["unit"]),
        provider_usage_key=str(row["provider_usage_key"]),
        cost_micros=optional_int(row["cost_micros"], field="cost_micros"),
        currency=str(row["currency"]),
        price_basis_json=json_object(
            row["price_basis_json"] or {}, field="price_basis_json"
        ),
    )


def _artifact_row(row: dict[str, object]) -> ObservabilityArtifact:
    return ObservabilityArtifact(
        artifact_id=str(row["artifact_id"]),
        artifact_kind=ArtifactKind(row["artifact_kind"]),
        content_hash=str(row["content_hash"]),
        content_type=str(row["content_type"]),
        size_bytes=required_int(row["size_bytes"], field="size_bytes"),
        has_content=row["content"] is not None,
        storage_ref=optional_text(row["storage_ref"], field="storage_ref"),
    )


def _string_column(engine: Engine, statement) -> tuple[str, ...]:
    with sql_connection(engine) as connection:
        return tuple(str(value) for value in connection.execute(statement).scalars())
