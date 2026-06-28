from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fervis.model_io.backbone.dto import ToolSpec
from fervis.model_io.turns import ModelTurnPurpose
from fervis.observability.usage_types import UsageKey
from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    RunStepKey,
)
from fervis.lineage.recorder import (
    CatalogEndpointWrite,
    RunStepWrite,
    SourceReadWrite,
)
from fervis.model_io.turn_artifacts import model_turn_artifact
from fervis.lookup.lineage.errors import LineagePersistenceUnavailable
from fervis.lookup.lineage.steps import LineageRuntimeStepSink


@dataclass
class _StepRecorder:
    steps: list[RunStepWrite] = field(default_factory=list)
    model_calls: list[object] = field(default_factory=list)
    usage_rows: list[object] = field(default_factory=list)
    artifacts: list[object] = field(default_factory=list)

    def record_step(self, step: RunStepWrite) -> RunStepWrite:
        self.steps.append(step)
        return step

    def record_step_with_source_context(
        self,
        step: RunStepWrite,
        catalog_endpoints: tuple[CatalogEndpointWrite, ...],
        source_reads: tuple[SourceReadWrite, ...],
    ) -> RunStepWrite:
        del catalog_endpoints
        del source_reads
        self.steps.append(step)
        return step

    def record_model_call(self, model_call: object) -> object:
        self.model_calls.append(model_call)
        return model_call

    def record_model_call_usage(self, usage: object) -> object:
        self.usage_rows.append(usage)
        return usage

    def record_artifact(self, artifact: object) -> object:
        self.artifacts.append(artifact)
        return artifact

    def record_model_call_audit(self, audit: object) -> object:
        self.model_calls.append(audit.model_call)
        self.usage_rows.extend(audit.usage_rows)
        self.artifacts.extend(audit.artifacts)
        return audit


class _ThrowingAuditRecorder(_StepRecorder):
    def record_model_call_audit(self, audit: object) -> object:
        raise RuntimeError("db unavailable")


def test_lookup_lineage_step_sink_offsets_sequences_by_worker_attempt() -> None:
    recorder = _StepRecorder()
    sink = LineageRuntimeStepSink(
        run_id="run_1",
        recorder=recorder,
        attempt=2,
    )

    sink.record_model_turn(
        purpose=ModelTurnPurpose.SOURCE_BINDING,
        turn=6,
    )
    sink.record_execution(relation_count=1)

    assert [
        (step.sequence, step.attempt, step.step_key) for step in recorder.steps
    ] == [
        (10006, 2, RunStepKey.SOURCE_BINDING),
        (19000, 2, RunStepKey.EXECUTE),
    ]


def test_lookup_lineage_step_sink_maps_every_model_turn_purpose() -> None:
    recorder = _StepRecorder()
    sink = LineageRuntimeStepSink(run_id="run_1", recorder=recorder)
    purposes = tuple(
        value
        for name, value in vars(ModelTurnPurpose).items()
        if name.isupper() and isinstance(value, str)
    )

    for index, purpose in enumerate(purposes, start=1):
        sink.record_model_turn(purpose=purpose, turn=index)

    assert len(recorder.steps) == len(purposes)


def test_lookup_lineage_step_sink_rejects_unknown_model_turn_purpose() -> None:
    sink = LineageRuntimeStepSink(run_id="run_1", recorder=_StepRecorder())

    with pytest.raises(ValueError, match="unsupported model turn purpose"):
        sink.record_model_turn(purpose="unknown_turn", turn=1)


def test_lookup_lineage_step_sink_records_model_turn_audit_rows() -> None:
    recorder = _StepRecorder()
    sink = LineageRuntimeStepSink(run_id="run_1", recorder=recorder)
    raw_output = '{"tool":"answer","arguments":{"choice":"yes"}}'
    step = sink.record_model_turn(
        purpose=ModelTurnPurpose.SOURCE_BINDING,
        turn=6,
        prompt_chars=12,
        schema_chars=34,
    )

    sink.record_model_turn_audit(
        step=step,
        provider="openai",
        model_key="GPT_TEST",
        artifact=model_turn_artifact(
            system_prompt="system",
            prompt_text="prompt chars",
            provider_schema={"type": "object"},
            tool_specs=(_tool_spec(),),
            submitted_payload={"choice": "yes"},
            raw_output=raw_output,
            parsed_payload={"choice": "yes"},
        ),
        usage={
            UsageKey.INPUT_TOKENS: 10,
            UsageKey.OUTPUT_TOKENS: 3,
            UsageKey.THINKING_TOKENS: 2,
            UsageKey.MODEL_SUBCALLS: [
                {"provider": "openai", "model": "GPT_TEST", "attempt": 1}
            ],
        },
        duration_ms=10,
        succeeded=True,
    )

    assert len(recorder.model_calls) == 1
    model_call = recorder.model_calls[0]
    assert model_call.run_id == "run_1"
    assert model_call.step_id == step.step_id
    assert model_call.status is ModelCallStatus.SUCCEEDED
    assert model_call.prompt_chars == len("prompt chars")
    assert model_call.schema_chars > 0
    assert model_call.tool_spec_chars > 0
    assert model_call.raw_output_chars == len(raw_output)
    assert model_call.model_subcalls_json == (
        {"provider": "openai", "model": "GPT_TEST", "attempt": 1},
    )
    assert {(usage.usage_kind, usage.quantity) for usage in recorder.usage_rows} == {
        (ModelUsageKind.INPUT_TOKENS, 10),
        (ModelUsageKind.OUTPUT_TOKENS, 3),
        (ModelUsageKind.THINKING_TOKENS, 2),
    }
    assert {artifact.artifact_kind for artifact in recorder.artifacts} == {
        ArtifactKind.SYSTEM_PROMPT,
        ArtifactKind.PROMPT,
        ArtifactKind.SCHEMA,
        ArtifactKind.TOOL_SPEC,
        ArtifactKind.SUBMITTED_PAYLOAD,
        ArtifactKind.RAW_OUTPUT,
        ArtifactKind.PARSED_PAYLOAD,
    }
    assert {artifact.model_call_id for artifact in recorder.artifacts} == {
        model_call.model_call_id
    }


def test_lookup_lineage_step_sink_preserves_zero_cost_usage_rows_as_priced() -> None:
    recorder = _StepRecorder()
    sink = LineageRuntimeStepSink(run_id="run_1", recorder=recorder)
    step = sink.record_model_turn(
        purpose=ModelTurnPurpose.SOURCE_BINDING,
        turn=6,
        prompt_chars=12,
        schema_chars=34,
    )

    sink.record_model_turn_audit(
        step=step,
        provider="openai",
        model_key="GPT_TEST",
        artifact=model_turn_artifact(
            system_prompt="system",
            prompt_text="prompt",
            provider_schema={"type": "object"},
            tool_specs=(_tool_spec(),),
            submitted_payload={"choice": "yes"},
        ),
        usage={
            UsageKey.INPUT_TOKENS: 20,
            UsageKey.OUTPUT_TOKENS: 5,
            UsageKey.THINKING_TOKENS: 3,
            UsageKey.COST_USD: 0.0015,
            UsageKey.INPUT_COST_USD: 0.001,
            UsageKey.OUTPUT_COST_USD: 0.0005,
            UsageKey.THINKING_COST_USD: 0,
            UsageKey.COST_SOURCE: "configured_provider_pricing",
            UsageKey.PRICING_VERSION: "test-pricing",
        },
        duration_ms=10,
        succeeded=True,
    )

    assert [
        (row.provider_usage_key, row.cost_micros, row.currency)
        for row in recorder.usage_rows
    ] == [
        (UsageKey.INPUT_TOKENS, 1000, "USD"),
        (UsageKey.OUTPUT_TOKENS, 500, "USD"),
        (UsageKey.THINKING_TOKENS, 0, "USD"),
    ]


def test_lookup_lineage_step_sink_fails_closed_when_model_turn_audit_fails() -> None:
    sink = LineageRuntimeStepSink(run_id="run_1", recorder=_ThrowingAuditRecorder())
    step = sink.record_model_turn(
        purpose=ModelTurnPurpose.SOURCE_BINDING,
        turn=6,
        prompt_chars=12,
        schema_chars=34,
    )

    with pytest.raises(
        LineagePersistenceUnavailable,
        match="model-turn audit lineage persistence failed",
    ):
        sink.record_model_turn_audit(
            step=step,
            provider="openai",
            model_key="GPT_TEST",
            artifact=model_turn_artifact(
                system_prompt="system",
                prompt_text="prompt",
                provider_schema={"type": "object"},
                tool_specs=(_tool_spec(),),
                submitted_payload={},
            ),
            usage={},
            duration_ms=10,
            succeeded=True,
        )


def _tool_spec() -> ToolSpec:
    return ToolSpec(
        name="answer",
        description="Answer",
        input_schema={"type": "object"},
    )
