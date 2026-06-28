from __future__ import annotations

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
)
from fervis.observability.model_calls import ModelCallInspectionService
from fervis.observability.query import (
    ModelCallDetailLevel,
    ObservabilityArtifact,
    ObservabilityModelCall,
    ObservabilityQueryPort,
    ObservabilityRun,
    ObservabilityUsage,
)


def test_model_call_inspection_service_returns_call_usage_and_artifacts() -> None:
    service = ModelCallInspectionService(
        _ModelCallReadPort(
            calls=(
                ObservabilityModelCall(
                    model_call_id="call_source",
                    run_id="run_1",
                    step_id="step_source_binding",
                    step_key=RunStepKey.SOURCE_BINDING,
                    step_sequence=2,
                    call_index=1,
                    provider="openai",
                    model_key="gpt-test",
                    status=ModelCallStatus.SUCCEEDED,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=20,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="input_tokens",
                            cost_micros=1000,
                            currency="USD",
                        ),
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                            quantity=5,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="output_tokens",
                            cost_micros=500,
                            currency="USD",
                        ),
                    ),
                    artifacts=(
                        ObservabilityArtifact(
                            artifact_id="artifact_prompt",
                            artifact_kind=ArtifactKind.PROMPT,
                            content_hash="sha256:prompt",
                            content_type="text/plain",
                            size_bytes=11,
                            has_content=True,
                            storage_ref=None,
                        ),
                        ObservabilityArtifact(
                            artifact_id="artifact_schema",
                            artifact_kind=ArtifactKind.SCHEMA,
                            content_hash="sha256:schema",
                            content_type="application/json",
                            size_bytes=13,
                            has_content=True,
                            storage_ref=None,
                        ),
                    ),
                ),
            )
        )
    )

    calls = service.for_run("run_1")

    assert [call.model_call_id for call in calls] == ["call_source"]
    source_call = calls[0]
    assert source_call.step_key == RunStepKey.SOURCE_BINDING
    assert source_call.status == ModelCallStatus.SUCCEEDED
    assert [
        (usage.usage_kind, usage.quantity, usage.cost_micros)
        for usage in source_call.usage_rows
    ] == [
        (ModelUsageKind.INPUT_TOKENS, 20, 1000),
        (ModelUsageKind.OUTPUT_TOKENS, 5, 500),
    ]
    assert [
        (artifact.artifact_kind, artifact.size_bytes, artifact.has_content)
        for artifact in source_call.artifacts
    ] == [
        (ArtifactKind.PROMPT, 11, True),
        (ArtifactKind.SCHEMA, 13, True),
    ]


class _ModelCallReadPort(ObservabilityQueryPort):
    def __init__(self, *, calls: tuple[ObservabilityModelCall, ...]) -> None:
        self._calls = calls

    def run_id_for_answer(self, answer_id: str) -> str | None:
        return None

    def run_by_id(self, run_id: str) -> ObservabilityRun | None:
        return ObservabilityRun(run_id=run_id)

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        return (run_id,)

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        return ()

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        return ()

    def model_calls_for_run_ids(
        self, run_ids: tuple[str, ...], *, detail: ModelCallDetailLevel = "inspection"
    ) -> tuple[ObservabilityModelCall, ...]:
        run_id_set = set(run_ids)
        return tuple(call for call in self._calls if call.run_id in run_id_set)

    def model_calls_for_run(
        self,
        run_id: str,
        step_key: RunStepKey | None = None,
        *,
        detail: ModelCallDetailLevel = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]:
        return tuple(
            call
            for call in self._calls
            if call.run_id == run_id and (step_key is None or call.step_key == step_key)
        )
