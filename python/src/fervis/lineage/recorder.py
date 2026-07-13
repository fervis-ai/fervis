"""Framework-neutral lineage recorder write contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from fervis.types.enums import StrEnum
from typing import Any
from uuid import UUID

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.lineage.enums import (
    ARTIFACT_KINDS_REQUIRING_MODEL_CALL,
    AnswerValueKind,
    ArtifactKind,
    ConversationOriginKind,
    FactResultKind,
    MemoryArtifactSourceKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    PresentationClientKey,
    PresentationKind,
    ProgramInvocationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunStepScopeType,
    RunTriggerKind,
    QuestionRunKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lookup.clarification import (
    ClarificationNeed,
    ClarificationReason,
    clarification_from_payload,
)


JsonObject = dict[str, Any]
JsonArray = tuple[JsonObject, ...]


@dataclass(frozen=True)
class ConversationWrite:
    conversation_id: str
    tenant_id: str
    read_context_ref: ReadContextRef = field(
        default_factory=lambda: ReadContextRef(scheme="anonymous")
    )
    origin_kind: ConversationOriginKind = ConversationOriginKind.INITIAL
    parent_conversation_id: str | None = None
    forked_after_question_id: str | None = None
    forked_after_run_id: str | None = None
    origin_ref: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.read_context_ref, ReadContextRef):
            object.__setattr__(
                self,
                "read_context_ref",
                ReadContextRef.from_storage_dict(self.read_context_ref),
            )
        _canonicalize_optional_str(self, "parent_conversation_id")
        _canonicalize_optional_str(self, "forked_after_question_id")
        _canonicalize_optional_str(self, "forked_after_run_id")
        if self.origin_kind is ConversationOriginKind.INITIAL:
            _require_absent(self.parent_conversation_id, "parent_conversation_id")
            _require_absent(self.forked_after_question_id, "forked_after_question_id")
            _require_absent(self.forked_after_run_id, "forked_after_run_id")
            return
        _require_present(self.parent_conversation_id, "parent_conversation_id")
        _require_present(self.forked_after_question_id, "forked_after_question_id")
        _require_present(self.forked_after_run_id, "forked_after_run_id")


@dataclass(frozen=True)
class QuestionWrite:
    question_id: str
    conversation_id: str
    conversation_sequence: int
    original_question: str
    origin_message_ref: str = ""


@dataclass(frozen=True)
class QuestionRunWrite:
    run_id: str
    question_id: str
    run_number: int
    kind: QuestionRunKind
    trigger_kind: RunTriggerKind
    adapter_ref: str
    runtime_version: str
    base_run_id: str | None = None

    def __post_init__(self) -> None:
        _require_optional_nonempty_str(self.base_run_id, "base_run_id")
        valid_pairings = {
            (QuestionRunKind.MODEL_ASSISTED, RunTriggerKind.INITIAL),
            (QuestionRunKind.MODEL_ASSISTED, RunTriggerKind.RETRY),
            (QuestionRunKind.DETERMINISTIC, RunTriggerKind.RERUN),
        }
        if (self.kind, self.trigger_kind) not in valid_pairings:
            raise ValueError("invalid question run kind and trigger pairing")
        if self.trigger_kind is RunTriggerKind.INITIAL:
            _require_absent(self.base_run_id, "base_run_id")
            return
        _require_present(self.base_run_id, "base_run_id")


@dataclass(frozen=True)
class AnswerProgramWrite:
    program_id: str
    schema_revision: int
    canonical_json: str


@dataclass(frozen=True)
class ProgramInvocationWrite:
    invocation_id: str
    run_id: str
    program_id: str
    bindings_json: str
    kind: ProgramInvocationKind
    base_invocation_id: str | None = None
    patch_id: str | None = None
    binding_patch_json: str | None = None
    revision_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ProgramInvocationKind.COMPILED_QUESTION:
            _require_absent(self.base_invocation_id, "base_invocation_id")
        else:
            _require_present(self.base_invocation_id, "base_invocation_id")
        if (self.patch_id is None) != (self.binding_patch_json is None):
            raise ValueError(
                "program invocation patch id and payload must be present together"
            )


@dataclass(frozen=True)
class ProgramInvocationBundleWrite:
    program: AnswerProgramWrite
    invocation: ProgramInvocationWrite


@dataclass(frozen=True)
class ProgramRevisionWrite:
    revision_id: str
    base_program_id: str
    revised_program_id: str
    capability_id: str
    application_json: str


@dataclass(frozen=True)
class ProgramRevisionBundleWrite:
    program: AnswerProgramWrite
    revision: ProgramRevisionWrite


@dataclass(frozen=True)
class RunStepWrite:
    step_id: str
    run_id: str
    sequence: int
    step_key: RunStepKey
    kind: RunStepKind
    attempt: int | None = None
    scope_type: RunStepScopeType | None = None
    scope_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    input_summary_json: JsonObject = field(default_factory=dict)
    output_summary_json: JsonObject = field(default_factory=dict)
    error_json: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _canonicalize_optional_str(self, "scope_id")


@dataclass(frozen=True)
class ModelCallWrite:
    model_call_id: str
    run_id: str
    step_id: str
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
    model_subcalls_json: JsonArray = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelCallUsageWrite:
    usage_id: str
    run_id: str
    model_call_id: str
    usage_kind: ModelUsageKind
    quantity: int
    unit: ModelUsageUnit
    provider_usage_key: str
    cost_micros: int | None = None
    currency: str = ""
    price_basis_json: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCallAuditWrite:
    model_call: ModelCallWrite
    usage_rows: tuple[ModelCallUsageWrite, ...] = ()
    artifacts: tuple["RunArtifactWrite", ...] = ()

    def __post_init__(self) -> None:
        _require_same_run_id(
            self.model_call.run_id,
            (self.model_call, *self.usage_rows, *self.artifacts),
        )
        for usage in self.usage_rows:
            if usage.model_call_id != self.model_call.model_call_id:
                raise ValueError("model call usage must reference the model call")
        for artifact in self.artifacts:
            if artifact.model_call_id != self.model_call.model_call_id:
                raise ValueError("model call artifacts must reference the model call")
            if artifact.step_id != self.model_call.step_id:
                raise ValueError("model call artifacts must use the model call step")


@dataclass(frozen=True)
class CatalogEndpointWrite:
    catalog_endpoint_id: str
    run_id: str
    catalog_endpoint_key: str
    endpoint_name: str
    framework_kind: str | StrEnum
    source_namespace_kind: str | StrEnum
    source_namespace_path_json: tuple[str, ...]
    route_method: str
    route_path_template: str
    handler_ref: str
    route_name: str = ""
    api_schema_operation_id: str = ""
    domain_resource_names_json: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_uuid(self.catalog_endpoint_id, "catalog_endpoint_id")
        _require_present(self.run_id, "run_id")
        _require_present(self.catalog_endpoint_key, "catalog_endpoint_key")
        _require_present(self.endpoint_name, "endpoint_name")
        _require_present(self.route_method, "route_method")
        _require_present(self.route_path_template, "route_path_template")
        _require_present(self.handler_ref, "handler_ref")
        object.__setattr__(
            self,
            "framework_kind",
            _token_value(self.framework_kind, "framework_kind"),
        )
        object.__setattr__(
            self,
            "source_namespace_kind",
            _token_value(self.source_namespace_kind, "source_namespace_kind"),
        )


@dataclass(frozen=True)
class SourceReadWrite:
    source_read_id: str
    run_id: str
    step_id: str
    catalog_endpoint_id: str
    status: SourceReadStatus
    args_json: JsonObject = field(default_factory=dict)
    row_count: int | None = None
    completeness_json: JsonObject = field(default_factory=dict)
    response_hash: str = ""
    artifact_id: str | None = None
    error_json: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_present(self.catalog_endpoint_id, "catalog_endpoint_id")
        if self.status is SourceReadStatus.SUCCEEDED:
            if self.row_count is None:
                raise ValueError("successful source reads require row_count")
            _require_present(self.response_hash, "response_hash")
            if not self.completeness_json:
                raise ValueError("successful source reads require completeness_json")


@dataclass(frozen=True)
class RunArtifactWrite:
    artifact_id: str
    run_id: str
    step_id: str
    artifact_kind: ArtifactKind
    content_hash: str
    content_type: str
    size_bytes: int
    model_call_id: str | None = None
    content: str | None = None
    storage_ref: str | None = None

    def __post_init__(self) -> None:
        has_content = self.content is not None
        has_storage_ref = self.storage_ref is not None
        if has_content == has_storage_ref:
            raise ValueError("artifacts require exactly one of content or storage_ref")
        if self.storage_ref == "":
            raise ValueError("storage_ref cannot be blank")
        if (
            self.artifact_kind.value in ARTIFACT_KINDS_REQUIRING_MODEL_CALL
            and self.model_call_id is None
        ):
            raise ValueError(
                f"{self.artifact_kind.value} artifacts require model_call_id"
            )


@dataclass(frozen=True)
class RunResultWrite:
    run_result_id: str
    run_id: str
    result_kind: RunResultKind


@dataclass(frozen=True)
class RuntimeErrorWrite:
    runtime_error_detail_id: str
    run_id: str
    run_result_id: str
    error_kind: RuntimeErrorKind
    message: str
    failed_step_id: str | None = None


@dataclass(frozen=True)
class RuntimeErrorResultWrite:
    result: RunResultWrite
    error: RuntimeErrorWrite

    def __post_init__(self) -> None:
        if self.result.result_kind is not RunResultKind.RUNTIME_ERROR:
            raise ValueError(
                "runtime error terminal writes require runtime_error result"
            )
        if self.result.run_id != self.error.run_id:
            raise ValueError("runtime error result and detail must use the same run_id")
        if self.result.run_result_id != self.error.run_result_id:
            raise ValueError(
                "runtime error result and detail must use the same run_result_id"
            )


@dataclass(frozen=True)
class AnsweredRunResultWrite:
    result: RunResultWrite
    requested_facts: tuple["RequestedFactWrite", ...]
    fact_results: tuple["FactResultWrite", ...]
    proof_graphs: tuple["ExecutionProofGraphWrite", ...]
    answer: "AnswerWrite"
    outputs: tuple["AnswerOutputWrite", ...]
    presentations: tuple["AnswerPresentationWrite", ...] = ()
    memory_artifacts: tuple["MemoryArtifactWrite", ...] = ()

    def __post_init__(self) -> None:
        if self.result.result_kind is not RunResultKind.ANSWERED:
            raise ValueError("answered terminal writes require answered result")
        _require_same_run_id(
            self.result.run_id,
            (
                *self.requested_facts,
                *self.fact_results,
                *self.proof_graphs,
                self.answer,
                *self.outputs,
                *self.presentations,
                *self.memory_artifacts,
            ),
        )
        if self.answer.run_result_id != self.result.run_result_id:
            raise ValueError("answer must reference the answered run result")


@dataclass(frozen=True)
class FactualTerminalRunResultWrite:
    result: RunResultWrite
    requested_facts: tuple["RequestedFactWrite", ...]
    fact_results: tuple["FactResultWrite", ...]
    proof_graphs: tuple["ExecutionProofGraphWrite", ...] = ()
    memory_artifacts: tuple["MemoryArtifactWrite", ...] = ()

    def __post_init__(self) -> None:
        if self.result.result_kind is not RunResultKind.FACTUAL_TERMINAL:
            raise ValueError("factual terminal writes require factual_terminal result")
        _require_same_run_id(
            self.result.run_id,
            (
                *self.requested_facts,
                *self.fact_results,
                *self.proof_graphs,
                *self.memory_artifacts,
            ),
        )
        answered = [
            item.fact_result_id
            for item in self.fact_results
            if item.result_kind is FactResultKind.ANSWERED
        ]
        if answered:
            raise ValueError("factual terminal fact results cannot be answered")


@dataclass(frozen=True)
class RequestedFactWrite:
    requested_fact_id: str
    run_id: str
    produced_by_step_id: str
    fact_key: str
    answer_expression_family: str
    description: str = ""
    requested_fact_json: JsonObject = field(default_factory=dict)
    answer_requests_json: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class FactResultWrite:
    fact_result_id: str
    run_id: str
    requested_fact_id: str
    produced_by_step_id: str
    result_kind: FactResultKind
    evidence_refs_json: list[str] = field(default_factory=list)
    payload_schema: str = ""
    payload_schema_rev: int | None = None
    payload_json: JsonObject | None = None

    def __post_init__(self) -> None:
        _validate_versioned_payload(
            payload_json=self.payload_json,
            payload_schema=self.payload_schema,
            payload_schema_rev=self.payload_schema_rev,
            label="fact result",
        )


@dataclass(frozen=True)
class MemoryArtifactWrite:
    memory_artifact_id: str
    run_id: str
    produced_by_step_id: str
    source_kind: MemoryArtifactSourceKind
    payload_schema: str
    payload_schema_rev: int
    payload_json: JsonObject
    requested_fact_id: str | None = None
    fact_result_id: str | None = None

    def __post_init__(self) -> None:
        _canonicalize_optional_str(self, "requested_fact_id")
        _canonicalize_optional_str(self, "fact_result_id")
        _validate_versioned_payload(
            payload_json=self.payload_json,
            payload_schema=self.payload_schema,
            payload_schema_rev=self.payload_schema_rev,
            label="memory artifact",
        )
        if self.payload_json.get("sourceKind") != self.source_kind.value:
            raise ValueError(
                "memory artifact payload sourceKind must match source_kind"
            )
        if self.payload_json.get("artifactId") != self.memory_artifact_id:
            raise ValueError("memory artifact payload artifactId must match row id")
        _require_present(self.payload_json.get("outcome"), "outcome")
        if self.source_kind is MemoryArtifactSourceKind.REQUESTED_FACT:
            _require_present(self.requested_fact_id, "requested_fact_id")
            _require_absent(self.fact_result_id, "fact_result_id")
            _require_no_memory_addresses(self.payload_json, self.source_kind)
            return
        if self.source_kind is MemoryArtifactSourceKind.FACT_RESULT:
            _require_present(self.fact_result_id, "fact_result_id")
            _require_absent(self.requested_fact_id, "requested_fact_id")
            _require_memory_addresses(self.payload_json, self.source_kind)
            return
        if self.source_kind is MemoryArtifactSourceKind.RUN_TERMINAL:
            _require_absent(self.requested_fact_id, "requested_fact_id")
            _require_absent(self.fact_result_id, "fact_result_id")
            _require_memory_addresses(self.payload_json, self.source_kind)
            return
        _require_absent(self.requested_fact_id, "requested_fact_id")
        _require_absent(self.fact_result_id, "fact_result_id")
        _require_memory_addresses(self.payload_json, self.source_kind)


@dataclass(frozen=True)
class ClarificationRequestWrite:
    clarification_id: str
    run_id: str
    payload_json: JsonObject
    step_id: str
    need: ClarificationNeed = field(init=False)
    reason: ClarificationReason = field(init=False)

    def __post_init__(self) -> None:
        if not self.payload_json:
            raise ValueError("clarification request requires payload")
        parsed = clarification_from_payload(self.payload_json)
        if parsed.id != self.clarification_id:
            raise ValueError("clarification request identity must match payload")
        object.__setattr__(
            self,
            "need",
            ClarificationNeed(_required_payload_text(self.payload_json, "need")),
        )
        object.__setattr__(
            self,
            "reason",
            ClarificationReason(_required_payload_text(self.payload_json, "reason")),
        )
        if not self.step_id.strip():
            raise ValueError("clarification request requires producing step")


@dataclass(frozen=True)
class ClarificationResponseWrite:
    response_id: str
    run_id: str
    clarification_id: str
    evidence_ref: str
    source_message_ref: str = ""
    selected_option_id: str = ""
    response_text: str = ""


@dataclass(frozen=True)
class AnswerWrite:
    answer_id: str
    run_id: str
    run_result_id: str


@dataclass(frozen=True)
class AnswerOutputWrite:
    answer_output_id: str
    run_id: str
    answer_id: str
    fact_result_id: str
    output_key: str
    value_kind: AnswerValueKind
    value_json: JsonObject
    proof_node_refs_json: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerPresentationWrite:
    presentation_id: str
    run_id: str
    answer_id: str
    presentation_kind: PresentationKind
    render_step_id: str
    client_key: PresentationClientKey = PresentationClientKey.DEFAULT
    locale: str = "default"
    rendered_value: str | None = None
    payload_schema: str = ""
    payload_schema_rev: int | None = None
    payload_json: JsonObject | None = None

    def __post_init__(self) -> None:
        has_rendered_value = self.rendered_value is not None
        has_payload_json = self.payload_json is not None
        if has_rendered_value == has_payload_json:
            raise ValueError(
                "answer presentation requires exactly one of rendered_value or payload_json"
            )
        if has_rendered_value:
            _require_absent(self.payload_schema, "payload_schema")
            if self.payload_schema_rev is not None:
                raise ValueError("payload_schema_rev must be absent")
        else:
            _validate_versioned_payload(
                payload_json=self.payload_json,
                payload_schema=self.payload_schema,
                payload_schema_rev=self.payload_schema_rev,
                label="answer presentation",
            )


@dataclass(frozen=True)
class ExecutionProofGraphWrite:
    proof_graph_id: str
    run_id: str
    fact_result_id: str
    compile_step_id: str
    payload_schema: str
    payload_schema_rev: int
    payload_json: JsonObject
    execute_step_id: str | None = None

    def __post_init__(self) -> None:
        _canonicalize_optional_str(self, "execute_step_id")


class LineageRecorderConflict(ValueError):
    """Raised when an idempotent lineage write conflicts with existing lineage."""


def _require_present(value: str | None, field_name: str) -> None:
    if value is None or value == "":
        raise ValueError(f"{field_name} is required")


def _require_token(value: str, field_name: str) -> None:
    _require_present(value, field_name)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string token")
    if value == "unknown":
        raise ValueError(f"{field_name} cannot be unknown")
    if value != value.lower() or any(
        not (char.isalnum() or char == "_") for char in value
    ):
        raise ValueError(f"{field_name} must be a lowercase snake_case token")


def _token_value(value: str | StrEnum, field_name: str) -> str:
    if isinstance(value, StrEnum):
        value = value.value
    _require_token(value, field_name)
    return value


def _require_uuid(value: str, field_name: str) -> None:
    _require_present(value, field_name)
    try:
        UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a UUID string") from exc


def _require_absent(value: str | None, field_name: str) -> None:
    if value is not None and value != "":
        raise ValueError(f"{field_name} must be absent")


def _canonicalize_optional_str(instance: object, field_name: str) -> None:
    if getattr(instance, field_name) == "":
        object.__setattr__(instance, field_name, None)


def _require_optional_nonempty_str(value: str | None, field_name: str) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when present")


def _required_payload_text(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"clarification payload requires {key}")
    return value


def _require_same_run_id(run_id: str, writes: tuple[object, ...]) -> None:
    for write in writes:
        if getattr(write, "run_id", None) != run_id:
            raise ValueError("answered terminal writes must use one run_id")


def _validate_versioned_payload(
    *,
    payload_json: JsonObject | None,
    payload_schema: str,
    payload_schema_rev: int | None,
    label: str,
) -> None:
    if payload_json is None:
        _require_absent(payload_schema, "payload_schema")
        if payload_schema_rev is not None:
            raise ValueError("payload_schema_rev must be absent")
        return
    _require_present(payload_schema, "payload_schema")
    if payload_schema_rev is None:
        raise ValueError(f"{label} payload requires payload_schema_rev")


def _require_memory_addresses(
    payload_json: JsonObject, source_kind: MemoryArtifactSourceKind
) -> None:
    addresses = payload_json.get("addresses")
    if not isinstance(addresses, list) or not addresses:
        raise ValueError(f"{source_kind.value} memory artifacts require addresses")


def _require_no_memory_addresses(
    payload_json: JsonObject, source_kind: MemoryArtifactSourceKind
) -> None:
    if payload_json.get("addresses"):
        raise ValueError(f"{source_kind.value} memory artifacts cannot carry addresses")
