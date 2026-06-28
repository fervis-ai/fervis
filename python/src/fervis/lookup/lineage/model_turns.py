"""Adapt lookup model-turn artifacts into lineage write contracts."""

from __future__ import annotations

from fervis.lineage.enums import ModelCallStatus
from fervis.lineage.model_calls import (
    ModelCallCapture,
    model_call_audit_write,
)
from fervis.lineage.recorder import (
    ModelCallAuditWrite,
    RunStepWrite,
)
from fervis.model_io.turn_artifacts import ModelTurnArtifact


def model_turn_audit_write(
    *,
    run_id: str,
    step: RunStepWrite,
    provider: str,
    model_key: str,
    artifact: ModelTurnArtifact,
    usage: dict[str, object],
    duration_ms: int,
    succeeded: bool,
) -> ModelCallAuditWrite:
    return model_call_audit_write(
        run_id=run_id,
        step=step,
        capture=ModelCallCapture(
            provider=provider,
            model_key=model_key,
            status=(ModelCallStatus.SUCCEEDED if succeeded else ModelCallStatus.FAILED),
            duration_ms=duration_ms,
            system_prompt=artifact.system_prompt,
            prompt_text=artifact.prompt_text,
            provider_schema=artifact.provider_schema,
            tool_specs=artifact.tool_specs,
            usage=dict(usage),
            submitted_payload=artifact.submitted_payload,
            raw_output=artifact.raw_output,
            parsed_payload=artifact.parsed_payload,
        ),
    )
