"""Lookup runtime bridge to canonical lineage step recording."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from fervis.model_io.turns import ModelTurnPurpose
from fervis.observability.event_contracts import EventPayloadKey
from fervis.lineage.enums import RunStepKey, RunStepKind
from fervis.lineage.ports import LineageRecorderPort
from fervis.lineage.recorder import (
    CatalogEndpointWrite,
    ModelCallAuditWrite,
    RunStepWrite,
    SourceReadWrite,
)
from fervis.model_io.turn_artifacts import ModelTurnArtifact
from fervis.lookup.orchestration.request import LookupRuntimePorts
from fervis.lookup.lineage.errors import LineagePersistenceUnavailable
from fervis.lookup.lineage.step_summaries import model_turn_output_summary
from fervis.lookup.lineage.model_turns import model_turn_audit_write

_MODEL_TURN_STEP_KEYS = {
    ModelTurnPurpose.CONVERSATION_RESOLUTION: RunStepKey.CONVERSATION_RESOLUTION,
    ModelTurnPurpose.QUESTION_CONTRACT: RunStepKey.QUESTION_CONTRACT,
    ModelTurnPurpose.QUERY_ENRICHMENT: RunStepKey.QUERY_ENRICHMENT,
    ModelTurnPurpose.GROUNDING: RunStepKey.GROUNDING,
    ModelTurnPurpose.READ_ELIGIBILITY: RunStepKey.READ_ELIGIBILITY,
    ModelTurnPurpose.PLAN_SELECTION: RunStepKey.PLAN_SELECTION,
    ModelTurnPurpose.SOURCE_BINDING: RunStepKey.SOURCE_BINDING,
    ModelTurnPurpose.PATTERN_FACT_PLANNING: RunStepKey.FACT_PLANNING,
    ModelTurnPurpose.FACT_PLAN: RunStepKey.FACT_PLANNING,
    ModelTurnPurpose.ANSWER_SYNTHESIS: RunStepKey.ANSWER_SYNTHESIS,
}

_DETERMINISTIC_STEP_SEQUENCE = {
    RunStepKey.COMPILE: 8900,
    RunStepKey.EXECUTE: 9000,
    RunStepKey.RENDER: 9100,
}


@dataclass(frozen=True)
class LineageRuntimeStepSink:
    run_id: str
    recorder: LineageRecorderPort
    attempt: int | None = None

    def record_model_turn(
        self,
        *,
        purpose: str,
        turn: int,
        prompt_chars: int | None = None,
        schema_chars: int | None = None,
        output_summary_json: dict[str, Any] | None = None,
        error_json: dict[str, Any] | None = None,
    ) -> RunStepWrite:
        step_key = _model_turn_step_key(purpose)
        sequence = _attempt_sequence(
            base_sequence=_required_positive_int(turn, "turn"),
            attempt=self.attempt,
        )
        return self.recorder.record_step(
            _step(
                run_id=self.run_id,
                sequence=sequence,
                step_key=step_key,
                kind=RunStepKind.MODEL_TURN,
                attempt=self.attempt,
                input_summary_json={
                    EventPayloadKey.PURPOSE: purpose,
                    EventPayloadKey.PROMPT_CHARS: prompt_chars,
                    EventPayloadKey.SCHEMA_CHARS: schema_chars,
                },
                output_summary_json=output_summary_json or {},
                error_json=error_json or {},
            )
        )

    def record_model_turn_audit(
        self,
        *,
        step: RunStepWrite,
        provider: str,
        model_key: str,
        artifact: ModelTurnArtifact,
        usage: dict[str, object],
        duration_ms: int,
        succeeded: bool,
    ) -> ModelCallAuditWrite:
        try:
            audit = model_turn_audit_write(
                run_id=self.run_id,
                step=step,
                provider=provider,
                model_key=model_key,
                artifact=artifact,
                usage=usage,
                duration_ms=duration_ms,
                succeeded=succeeded,
            )
            return self.recorder.record_model_call_audit(audit)
        except LineagePersistenceUnavailable:
            raise
        except Exception as exc:
            raise LineagePersistenceUnavailable(
                "model-turn audit lineage persistence failed"
            ) from exc

    def record_execution(
        self,
        *,
        relation_count: int | None = None,
        proof_refs: tuple[str, ...] = (),
        error_json: dict[str, Any] | None = None,
        catalog_endpoints: tuple[CatalogEndpointWrite, ...] = (),
        source_reads: tuple[SourceReadWrite, ...] = (),
    ) -> RunStepWrite:
        return self._record_deterministic(
            step_key=RunStepKey.EXECUTE,
            output_summary_json={
                key: value
                for key, value in {
                    EventPayloadKey.RELATION_COUNT: relation_count,
                    EventPayloadKey.PROOF_REFS: list(proof_refs),
                }.items()
                if value not in (None, [], ())
            },
            error_json=error_json or {},
            catalog_endpoints=catalog_endpoints,
            source_reads=source_reads,
        )

    def record_step_source_context(
        self,
        step: RunStepWrite,
        *,
        catalog_endpoints: tuple[CatalogEndpointWrite, ...],
        source_reads: tuple[SourceReadWrite, ...],
    ) -> RunStepWrite:
        if not catalog_endpoints and not source_reads:
            return step
        return self.recorder.record_step_with_source_context(
            step,
            catalog_endpoints,
            source_reads,
        )

    def execution_step_id(self) -> str:
        return self._deterministic_step_id(RunStepKey.EXECUTE)

    def compile_step_id(self) -> str:
        return self._deterministic_step_id(RunStepKey.COMPILE)

    def model_turn_step_id(self, *, purpose: str, turn: int) -> str:
        step_key = _model_turn_step_key(purpose)
        sequence = _attempt_sequence(
            base_sequence=_required_positive_int(turn, "turn"),
            attempt=self.attempt,
        )
        return _step_id(run_id=self.run_id, sequence=sequence, step_key=step_key)

    def record_compile(
        self,
        *,
        proof_node_count: int,
        proof_edge_count: int,
    ) -> RunStepWrite:
        return self._record_deterministic(
            step_key=RunStepKey.COMPILE,
            output_summary_json={
                "proofNodeCount": proof_node_count,
                "proofEdgeCount": proof_edge_count,
            },
            error_json={},
        )

    def record_render(
        self,
        *,
        kind: str,
        row_count: int,
    ) -> RunStepWrite:
        return self._record_deterministic(
            step_key=RunStepKey.RENDER,
            output_summary_json={
                EventPayloadKey.KIND: kind,
                EventPayloadKey.ROW_COUNT: row_count,
            },
            error_json={},
        )

    def _deterministic_step_id(self, step_key: RunStepKey) -> str:
        sequence = _attempt_sequence(
            base_sequence=_DETERMINISTIC_STEP_SEQUENCE[step_key],
            attempt=self.attempt,
        )
        return _step_id(run_id=self.run_id, sequence=sequence, step_key=step_key)

    def _record_deterministic(
        self,
        *,
        step_key: RunStepKey,
        output_summary_json: dict[str, Any],
        error_json: dict[str, Any],
        catalog_endpoints: tuple[CatalogEndpointWrite, ...] = (),
        source_reads: tuple[SourceReadWrite, ...] = (),
    ) -> RunStepWrite:
        sequence = _attempt_sequence(
            base_sequence=_DETERMINISTIC_STEP_SEQUENCE[step_key],
            attempt=self.attempt,
        )
        step = _step(
            run_id=self.run_id,
            sequence=sequence,
            step_key=step_key,
            kind=RunStepKind.DETERMINISTIC,
            attempt=self.attempt,
            output_summary_json=output_summary_json,
            error_json=error_json,
        )
        if catalog_endpoints or source_reads:
            return self.recorder.record_step_with_source_context(
                step,
                catalog_endpoints,
                source_reads,
            )
        return self.recorder.record_step(step)


def record_model_turn_step(
    ports: LookupRuntimePorts,
    *,
    purpose: str,
    turn: int,
    prompt_chars: int | None,
    schema_chars: int | None,
    output_summary_json: dict[str, Any] | None = None,
    error_json: dict[str, Any] | None = None,
) -> RunStepWrite | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.record_model_turn(
        purpose=purpose,
        turn=turn,
        prompt_chars=prompt_chars,
        schema_chars=schema_chars,
        output_summary_json=output_summary_json,
        error_json=error_json,
    )


def record_model_turn_audit(
    ports: LookupRuntimePorts,
    *,
    step: RunStepWrite | None,
    provider: str,
    model_key: str,
    artifact: ModelTurnArtifact,
    usage: dict[str, object],
    duration_ms: int,
    succeeded: bool,
) -> ModelCallAuditWrite | None:
    if step is None:
        return None
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.record_model_turn_audit(
        step=step,
        provider=provider,
        model_key=model_key,
        artifact=artifact,
        usage=usage,
        duration_ms=duration_ms,
        succeeded=succeeded,
    )


def record_execution_step(
    ports: LookupRuntimePorts,
    *,
    relation_count: int | None = None,
    proof_refs: tuple[str, ...] = (),
    error_json: dict[str, Any] | None = None,
    catalog_endpoints: tuple[CatalogEndpointWrite, ...] = (),
    source_reads: tuple[SourceReadWrite, ...] = (),
) -> RunStepWrite | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.record_execution(
        relation_count=relation_count,
        proof_refs=proof_refs,
        error_json=error_json,
        catalog_endpoints=catalog_endpoints,
        source_reads=source_reads,
    )


def record_step_source_context(
    ports: LookupRuntimePorts,
    *,
    step: RunStepWrite | None,
    catalog_endpoints: tuple[CatalogEndpointWrite, ...],
    source_reads: tuple[SourceReadWrite, ...],
) -> RunStepWrite | None:
    if step is None:
        return None
    sink = _lineage_step_sink(ports)
    if sink is None:
        return step
    return sink.record_step_source_context(
        step,
        catalog_endpoints=catalog_endpoints,
        source_reads=source_reads,
    )


def record_compile_step(
    ports: LookupRuntimePorts,
    *,
    proof_node_count: int,
    proof_edge_count: int,
) -> RunStepWrite | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.record_compile(
        proof_node_count=proof_node_count,
        proof_edge_count=proof_edge_count,
    )


def compile_step_id(ports: LookupRuntimePorts) -> str | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.compile_step_id()


def execution_step_id(ports: LookupRuntimePorts) -> str | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.execution_step_id()


def model_turn_step_id(
    ports: LookupRuntimePorts,
    *,
    purpose: str,
    turn: int,
) -> str | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.model_turn_step_id(purpose=purpose, turn=turn)


def record_render_step(
    ports: LookupRuntimePorts,
    *,
    kind: str,
    row_count: int,
) -> RunStepWrite | None:
    sink = _lineage_step_sink(ports)
    if sink is None:
        return None
    return sink.record_render(kind=kind, row_count=row_count)


def lineage_error_json(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        EventPayloadKey.ERROR_CODE,
        EventPayloadKey.ERROR_CLASS,
        EventPayloadKey.ERROR_CONTEXT,
    )
    return {key: payload[key] for key in keys if payload.get(key)}


def lineage_model_turn_output_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return model_turn_output_summary(payload)


def _lineage_step_sink(ports: LookupRuntimePorts) -> Any | None:
    sink = ports.lineage_step_sink
    if sink is None and ports.lineage_required:
        raise LineagePersistenceUnavailable("lineage step sink is required")
    return sink


def _model_turn_step_key(purpose: str) -> RunStepKey:
    try:
        return _MODEL_TURN_STEP_KEYS[purpose]
    except KeyError as exc:
        raise ValueError(
            f"unsupported model turn purpose for lineage: {purpose}"
        ) from exc


def _attempt_sequence(*, base_sequence: int, attempt: int | None) -> int:
    if attempt is None:
        return base_sequence
    return ((_required_positive_int(attempt, "attempt") - 1) * 10000) + base_sequence


def _required_positive_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def _step(
    *,
    run_id: str,
    sequence: int,
    step_key: RunStepKey,
    kind: RunStepKind,
    attempt: int | None,
    input_summary_json: dict[str, Any] | None = None,
    output_summary_json: dict[str, Any] | None = None,
    error_json: dict[str, Any] | None = None,
) -> RunStepWrite:
    return RunStepWrite(
        step_id=_step_id(run_id=run_id, sequence=sequence, step_key=step_key),
        run_id=run_id,
        sequence=sequence,
        step_key=step_key,
        kind=kind,
        attempt=attempt,
        input_summary_json=input_summary_json or {},
        output_summary_json=output_summary_json or {},
        error_json=error_json or {},
    )


def _step_id(*, run_id: str, sequence: int, step_key: RunStepKey) -> str:
    digest = sha256(f"{run_id}:{sequence}:{step_key.value}".encode()).hexdigest()[:24]
    return f"step_{digest}"
