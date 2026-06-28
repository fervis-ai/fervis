"""Framework-neutral model-turn prompt capture contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    RunStepKey,
)


@dataclass(frozen=True)
class PromptCaptureArtifact:
    artifact_kind: ArtifactKind
    content: str
    content_type: str


@dataclass(frozen=True)
class PromptCaptureUsage:
    usage_kind: ModelUsageKind
    quantity: int
    provider_usage_key: str


@dataclass(frozen=True)
class ModelTurnPromptCapture:
    run_id: str
    sequence: int
    attempt: int | None
    step_key: RunStepKey
    call_index: int
    provider: str
    model_key: str
    status: ModelCallStatus
    provider_request_id: str = ""
    finish_reason: str = ""
    duration_ms: int | None = None
    prompt_chars: int = 0
    schema_chars: int = 0
    tool_spec_chars: int = 0
    submitted_payload_chars: int | None = None
    raw_output_chars: int | None = None
    step_input_summary: dict[str, object] | None = None
    step_output_summary: dict[str, object] | None = None
    error_json: dict[str, object] | None = None
    artifacts: tuple[PromptCaptureArtifact, ...] = ()
    usage_rows: tuple[PromptCaptureUsage, ...] = ()


class PromptCaptureQueryPort(Protocol):
    def model_turn_prompt_captures_for_run(
        self, run_id: str
    ) -> tuple[ModelTurnPromptCapture, ...]: ...


def prompt_capture_artifacts_by_kind(
    capture: ModelTurnPromptCapture,
) -> dict[ArtifactKind, PromptCaptureArtifact]:
    return {artifact.artifact_kind: artifact for artifact in capture.artifacts}
