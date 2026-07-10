"""Framework-neutral observability query contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
)


@dataclass(frozen=True)
class ObservabilityUsage:
    usage_kind: ModelUsageKind
    quantity: int
    unit: ModelUsageUnit
    provider_usage_key: str
    cost_micros: int | None = None
    currency: str = ""
    price_basis_json: dict[str, object] | None = None


@dataclass(frozen=True)
class ObservabilityRun:
    run_id: str
    base_run_id: str | None = None


@dataclass(frozen=True)
class ObservabilityArtifact:
    artifact_id: str
    artifact_kind: ArtifactKind
    content_hash: str
    content_type: str
    size_bytes: int
    has_content: bool
    storage_ref: str | None


@dataclass(frozen=True)
class ObservabilityArtifactContent:
    artifact_id: str
    artifact_kind: ArtifactKind
    content_hash: str
    content_type: str
    size_bytes: int
    content: str | None
    storage_ref: str | None


@dataclass(frozen=True)
class ObservabilityModelCall:
    model_call_id: str
    run_id: str
    step_id: str
    step_key: RunStepKey
    step_sequence: int
    call_index: int
    provider: str
    model_key: str
    status: ModelCallStatus
    provider_request_id: str = ""
    finish_reason: str = ""
    duration_ms: int | None = None
    reasoning_effort: str = ""
    reasoning_budget_tokens: int | None = None
    max_output_tokens: int | None = None
    prompt_chars: int = 0
    schema_chars: int = 0
    tool_spec_chars: int = 0
    submitted_payload_chars: int | None = None
    raw_output_chars: int | None = None
    model_subcalls: tuple[dict[str, object], ...] = ()
    usage_rows: tuple[ObservabilityUsage, ...] = ()
    artifacts: tuple[ObservabilityArtifact, ...] = ()


ModelCallDetailLevel = Literal["cost", "inspection"]


class ObservabilityQueryPort(Protocol):
    def run_id_for_answer(self, answer_id: str) -> str | None: ...

    def run_by_id(self, run_id: str) -> ObservabilityRun | None: ...

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]: ...

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]: ...

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]: ...

    def model_calls_for_run_ids(
        self, run_ids: tuple[str, ...], *, detail: ModelCallDetailLevel = "inspection"
    ) -> tuple[ObservabilityModelCall, ...]: ...

    def model_calls_for_run(
        self,
        run_id: str,
        step_key: RunStepKey | None = None,
        *,
        detail: ModelCallDetailLevel = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]: ...

    def artifact_content(
        self, artifact_id: str
    ) -> ObservabilityArtifactContent | None: ...
