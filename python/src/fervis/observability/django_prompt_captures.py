"""Django-backed model-turn prompt capture query adapter."""

from __future__ import annotations

from fervis.lineage import models
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
    PromptCaptureUsage,
)


class DjangoPromptCaptureQuery:
    def model_turn_prompt_captures_for_run(
        self, run_id: str
    ) -> tuple[ModelTurnPromptCapture, ...]:
        calls = (
            models.ModelCall.objects.filter(
                run_id=run_id,
                step__kind=RunStepKind.MODEL_TURN.value,
            )
            .select_related("step")
            .prefetch_related("artifacts", "usage_rows")
            .order_by("step__sequence", "call_index")
        )
        return tuple(_model_turn_capture(call) for call in calls)


def _model_turn_capture(call: models.ModelCall) -> ModelTurnPromptCapture:
    return ModelTurnPromptCapture(
        run_id=call.run_id,
        sequence=call.step.sequence,
        attempt=call.step.attempt,
        step_key=RunStepKey(call.step.step_key),
        call_index=call.call_index,
        provider=call.provider,
        model_key=call.model_key,
        status=ModelCallStatus(call.status),
        provider_request_id=call.provider_request_id,
        finish_reason=call.finish_reason,
        duration_ms=call.duration_ms,
        prompt_chars=call.prompt_chars,
        schema_chars=call.schema_chars,
        tool_spec_chars=call.tool_spec_chars,
        submitted_payload_chars=call.submitted_payload_chars,
        raw_output_chars=call.raw_output_chars,
        step_input_summary=dict(call.step.input_summary_json or {}),
        step_output_summary=dict(call.step.output_summary_json or {}),
        error_json=dict(call.step.error_json or {}),
        artifacts=tuple(
            _artifact_content(artifact) for artifact in call.artifacts.all()
        ),
        usage_rows=tuple(_usage_row(usage) for usage in call.usage_rows.all()),
    )


def _artifact_content(artifact: models.RunArtifact) -> PromptCaptureArtifact:
    return PromptCaptureArtifact(
        artifact_kind=ArtifactKind(artifact.artifact_kind),
        content=artifact.content or "",
        content_type=artifact.content_type,
    )


def _usage_row(usage: models.ModelCallUsage) -> PromptCaptureUsage:
    return PromptCaptureUsage(
        usage_kind=ModelUsageKind(usage.usage_kind),
        quantity=usage.quantity,
        provider_usage_key=usage.provider_usage_key,
    )
