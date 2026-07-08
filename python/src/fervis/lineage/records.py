"""Framework-neutral lineage persistence record specs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

from fervis.lineage.enums import (
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
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunStepScopeType,
    RunTriggerKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lookup.clarification import ClarificationNeed, ClarificationReason
from fervis.lineage.recorder import (
    AnswerOutputWrite,
    AnswerPresentationWrite,
    AnswerWrite,
    CatalogEndpointWrite,
    ClarificationRequestWrite,
    ClarificationResponseWrite,
    ConversationWrite,
    ExecutionProofGraphWrite,
    FactResultWrite,
    MemoryArtifactWrite,
    ModelCallUsageWrite,
    ModelCallWrite,
    QuestionRunWrite,
    QuestionWrite,
    RequestedFactWrite,
    RunArtifactWrite,
    RunResultWrite,
    RunStepWrite,
    RuntimeErrorWrite,
    SourceReadWrite,
)


T = TypeVar("T")


@dataclass(frozen=True)
class LineageRowField:
    write_name: str
    storage_name: str | None = None
    enum_type: type[StrEnum] | None = None
    none_as_blank: bool = False
    json_array: bool = False
    json_object: bool = False
    json_value: bool = False

    @property
    def storage_field(self) -> str:
        return self.storage_name or self.write_name

    def to_storage(self, write: object) -> Any:
        value = getattr(write, self.write_name)
        if value is None and self.none_as_blank:
            return ""
        if self.enum_type is not None:
            if not isinstance(value, self.enum_type):
                raise TypeError(f"{self.write_name} requires {self.enum_type.__name__}")
            return value.value
        if isinstance(value, StrEnum):
            raise TypeError(f"{self.write_name} does not accept enum values")
        if self.json_array or self.json_object or self.json_value:
            return _normalize_json_value(value)
        return value

    def from_storage(self, record: object) -> Any:
        value = getattr(record, self.storage_field)
        if self.none_as_blank and value == "":
            value = None
        if value is not None and self.enum_type is not None:
            return self.enum_type(value)
        if self.json_array and isinstance(value, list):
            return [dict(item) if isinstance(item, dict) else item for item in value]
        return value


def _normalize_json_value(value: Any) -> Any:
    if hasattr(value, "to_storage_dict"):
        return _normalize_json_value(value.to_storage_dict())
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class LineageRowFieldExpectation:
    target_field: str
    expected_value: Any
    label: str


@dataclass(frozen=True)
class LineageRowReference:
    target_key: str
    lookup: dict[str, Any]
    label: str
    field_expectations: tuple[LineageRowFieldExpectation, ...] = ()


@dataclass(frozen=True)
class LineageRow:
    key: str
    identity: dict[str, Any]
    values: dict[str, Any]
    conflict_lookup: dict[str, Any]
    same_run_refs: tuple[LineageRowReference, ...] = ()

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            field: value
            for field, value in self.values.items()
            if field not in self.identity
        }

    @property
    def storage_fields(self) -> tuple[str, ...]:
        return tuple(self.values)


@dataclass(frozen=True)
class SameRunFieldMatch:
    source_write_name: str
    target_write_name: str
    label: str


@dataclass(frozen=True)
class SameRunReference:
    target_key: str
    write_name: str
    label: str
    required: bool = True
    field_matches: tuple[SameRunFieldMatch, ...] = ()


@dataclass(frozen=True)
class LineageRowSpec(Generic[T]):
    key: str
    label: str
    write_type: type[T]
    identity: LineageRowField
    fields: tuple[LineageRowField, ...]
    conflict_lookup: tuple[str, ...] = ()
    same_run_refs: tuple[SameRunReference, ...] = ()

    def identity_value(self, write: T) -> Any:
        return self.identity.to_storage(write)

    def identity_lookup(self, write: T) -> dict[str, Any]:
        return {self.identity.storage_field: self.identity_value(write)}

    def defaults(self, write: T) -> dict[str, Any]:
        return {
            field.storage_field: field.to_storage(write)
            for field in self.fields
            if field.storage_field != self.identity.storage_field
        }

    def values(self, write: T) -> dict[str, Any]:
        return {
            self.identity.storage_field: self.identity_value(write)
        } | self.defaults(write)

    @property
    def storage_fields(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.identity.storage_field,
                    *(field.storage_field for field in self.fields),
                )
            )
        )

    def conflict_lookup_values(self, write: T) -> dict[str, Any]:
        return {
            self._field_by_write_name(
                field_name
            ).storage_field: self._field_by_write_name(field_name).to_storage(write)
            for field_name in self.conflict_lookup
        }

    def to_row(self, write: T) -> LineageRow:
        run_id = getattr(write, "run_id", None)
        same_run_refs: list[LineageRowReference] = []
        if run_id not in (None, ""):
            for reference in self.same_run_refs:
                value = getattr(write, reference.write_name)
                if value in (None, ""):
                    if reference.required:
                        raise ValueError(f"{reference.label} reference is missing")
                    continue
                same_run_refs.append(
                    LineageRowReference(
                        target_key=reference.target_key,
                        lookup={
                            RECORD_SPECS_BY_KEY[
                                reference.target_key
                            ].identity.storage_field: value,
                            "run_id": run_id,
                        },
                        label=reference.label,
                        field_expectations=tuple(
                            self._field_expectation(reference.target_key, match, write)
                            for match in reference.field_matches
                        ),
                    )
                )
        return LineageRow(
            key=self.key,
            identity=self.identity_lookup(write),
            values=self.values(write),
            conflict_lookup=(
                self.conflict_lookup_values(write) if self.conflict_lookup else {}
            ),
            same_run_refs=tuple(same_run_refs),
        )

    def from_storage(self, record: object) -> T:
        kwargs = {
            self.identity.write_name: self.identity.from_storage(record),
            **{
                field.write_name: field.from_storage(record)
                for field in self.fields
                if field.storage_field != self.identity.storage_field
            },
        }
        return self.write_type(**kwargs)

    def _field_by_write_name(self, field_name: str) -> LineageRowField:
        if self.identity.write_name == field_name:
            return self.identity
        for field in self.fields:
            if field.write_name == field_name:
                return field
        raise ValueError(f"{self.key} has no field {field_name!r}")

    def _field_expectation(
        self,
        target_key: str,
        match: SameRunFieldMatch,
        write: T,
    ) -> LineageRowFieldExpectation:
        source_field = self._field_by_write_name(match.source_write_name)
        target_field = RECORD_SPECS_BY_KEY[target_key]._field_by_write_name(
            match.target_write_name
        )
        return LineageRowFieldExpectation(
            target_field=target_field.storage_field,
            expected_value=source_field.to_storage(write),
            label=match.label,
        )


def field(
    write_name: str,
    *,
    storage_name: str | None = None,
    enum_type: type[StrEnum] | None = None,
    none_as_blank: bool = False,
    json_array: bool = False,
    json_object: bool = False,
    json_value: bool = False,
) -> LineageRowField:
    return LineageRowField(
        write_name=write_name,
        storage_name=storage_name,
        enum_type=enum_type,
        none_as_blank=none_as_blank,
        json_array=json_array,
        json_object=json_object,
        json_value=json_value,
    )


CONVERSATION = LineageRowSpec(
    key="conversation",
    label="conversation",
    write_type=ConversationWrite,
    identity=field("conversation_id"),
    fields=(
        field("tenant_id"),
        field("read_context_ref", json_object=True),
        field("origin_kind", enum_type=ConversationOriginKind),
        field("parent_conversation_id"),
        field("forked_after_question_id"),
        field("forked_after_run_id"),
        field("origin_ref"),
    ),
)

QUESTION = LineageRowSpec(
    key="question",
    label="question",
    write_type=QuestionWrite,
    identity=field("question_id"),
    fields=(
        field("conversation_id"),
        field("conversation_sequence"),
        field("origin_message_ref"),
        field("original_question"),
    ),
    conflict_lookup=("conversation_id", "conversation_sequence"),
)

QUESTION_RUN = LineageRowSpec(
    key="question_run",
    label="run",
    write_type=QuestionRunWrite,
    identity=field("run_id"),
    fields=(
        field("question_id"),
        field("run_number"),
        field("trigger_kind", enum_type=RunTriggerKind),
        field("previous_run_id"),
        field("trigger_clarification_response_run_id"),
        field("trigger_clarification_response_id", none_as_blank=True),
        field("integrated_question"),
        field("adapter_ref"),
        field("runtime_version"),
    ),
    conflict_lookup=("question_id", "run_number"),
)

RUN_STEP = LineageRowSpec(
    key="run_step",
    label="step",
    write_type=RunStepWrite,
    identity=field("step_id"),
    fields=(
        field("run_id"),
        field("sequence"),
        field("step_key", enum_type=RunStepKey),
        field("attempt"),
        field("scope_type", enum_type=RunStepScopeType, none_as_blank=True),
        field("scope_id", none_as_blank=True),
        field("kind", enum_type=RunStepKind),
        field("started_at"),
        field("finished_at"),
        field("input_summary_json", json_value=True),
        field("output_summary_json", json_value=True),
        field("error_json", json_value=True),
    ),
    conflict_lookup=("run_id", "sequence"),
)

MODEL_CALL = LineageRowSpec(
    key="model_call",
    label="model call",
    write_type=ModelCallWrite,
    identity=field("model_call_id"),
    fields=(
        field("run_id"),
        field("step_id"),
        field("call_index"),
        field("provider"),
        field("model_key"),
        field("provider_request_id"),
        field("status", enum_type=ModelCallStatus),
        field("finish_reason"),
        field("duration_ms"),
        field("reasoning_effort"),
        field("reasoning_budget_tokens"),
        field("max_output_tokens"),
        field("prompt_chars"),
        field("schema_chars"),
        field("tool_spec_chars"),
        field("submitted_payload_chars"),
        field("raw_output_chars"),
        field("model_subcalls_json", json_array=True),
    ),
    same_run_refs=(SameRunReference("run_step", "step_id", "model call step"),),
    conflict_lookup=("run_id", "step_id", "call_index"),
)

MODEL_CALL_USAGE = LineageRowSpec(
    key="model_call_usage",
    label="model call usage",
    write_type=ModelCallUsageWrite,
    identity=field("usage_id"),
    fields=(
        field("run_id"),
        field("model_call_id"),
        field("usage_kind", enum_type=ModelUsageKind),
        field("quantity"),
        field("unit", enum_type=ModelUsageUnit),
        field("provider_usage_key"),
        field("cost_micros"),
        field("currency"),
        field("price_basis_json", json_value=True),
    ),
    same_run_refs=(
        SameRunReference("model_call", "model_call_id", "usage model call"),
    ),
)

CATALOG_ENDPOINT = LineageRowSpec(
    key="catalog_endpoint",
    label="catalog endpoint",
    write_type=CatalogEndpointWrite,
    identity=field("catalog_endpoint_id"),
    fields=(
        field("catalog_endpoint_id"),
        field("run_id"),
        field("catalog_endpoint_key"),
        field("endpoint_name"),
        field("framework_kind"),
        field("source_namespace_kind"),
        field("source_namespace_path_json", json_array=True),
        field("route_method"),
        field("route_path_template"),
        field("route_name"),
        field("api_schema_operation_id"),
        field("handler_ref"),
        field("domain_resource_names_json", json_array=True),
    ),
    conflict_lookup=("run_id", "catalog_endpoint_key"),
    same_run_refs=(SameRunReference("question_run", "run_id", "catalog endpoint run"),),
)

SOURCE_READ = LineageRowSpec(
    key="source_read",
    label="source read",
    write_type=SourceReadWrite,
    identity=field("source_read_id"),
    fields=(
        field("run_id"),
        field("step_id"),
        field("catalog_endpoint_id"),
        field("args_json", json_value=True),
        field("status", enum_type=SourceReadStatus),
        field("row_count"),
        field("completeness_json", json_value=True),
        field("response_hash"),
        field("artifact_id"),
        field("error_json", json_value=True),
    ),
    same_run_refs=(
        SameRunReference("run_step", "step_id", "source read step"),
        SameRunReference(
            "catalog_endpoint",
            "catalog_endpoint_id",
            "source read catalog endpoint",
        ),
        SameRunReference(
            "run_artifact",
            "artifact_id",
            "source read artifact",
            required=False,
            field_matches=(
                SameRunFieldMatch(
                    source_write_name="step_id",
                    target_write_name="step_id",
                    label="source read artifact step",
                ),
            ),
        ),
    ),
)

RUN_ARTIFACT = LineageRowSpec(
    key="run_artifact",
    label="artifact",
    write_type=RunArtifactWrite,
    identity=field("artifact_id"),
    fields=(
        field("run_id"),
        field("step_id"),
        field("model_call_id"),
        field("artifact_kind", enum_type=ArtifactKind),
        field("content_hash"),
        field("content"),
        field("storage_ref"),
        field("content_type"),
        field("size_bytes"),
    ),
    same_run_refs=(
        SameRunReference("run_step", "step_id", "artifact step"),
        SameRunReference(
            "model_call",
            "model_call_id",
            "artifact model call",
            required=False,
            field_matches=(
                SameRunFieldMatch(
                    source_write_name="step_id",
                    target_write_name="step_id",
                    label="artifact model call step",
                ),
            ),
        ),
    ),
)

RUN_RESULT = LineageRowSpec(
    key="run_result",
    label="run result",
    write_type=RunResultWrite,
    identity=field("run_result_id"),
    fields=(field("run_id"), field("result_kind", enum_type=RunResultKind)),
    conflict_lookup=("run_id",),
)

RUNTIME_ERROR = LineageRowSpec(
    key="runtime_error_detail",
    label="runtime error detail",
    write_type=RuntimeErrorWrite,
    identity=field("runtime_error_detail_id"),
    fields=(
        field("run_id"),
        field("run_result_id"),
        field("failed_step_id"),
        field("error_kind", enum_type=RuntimeErrorKind),
        field("message"),
    ),
    same_run_refs=(
        SameRunReference("run_result", "run_result_id", "runtime error result"),
        SameRunReference(
            "run_step",
            "failed_step_id",
            "runtime error step",
            required=False,
        ),
    ),
    conflict_lookup=("run_id", "run_result_id"),
)

REQUESTED_FACT = LineageRowSpec(
    key="requested_fact",
    label="requested fact",
    write_type=RequestedFactWrite,
    identity=field("requested_fact_id"),
    fields=(
        field("run_id"),
        field("produced_by_step_id"),
        field("fact_key"),
        field("description"),
        field("answer_expression_family"),
        field("requested_fact_json", json_value=True),
        field("answer_requests_json", json_value=True),
    ),
    conflict_lookup=("run_id", "produced_by_step_id", "fact_key"),
    same_run_refs=(
        SameRunReference("run_step", "produced_by_step_id", "requested fact step"),
    ),
)

FACT_RESULT = LineageRowSpec(
    key="fact_result",
    label="fact result",
    write_type=FactResultWrite,
    identity=field("fact_result_id"),
    fields=(
        field("run_id"),
        field("requested_fact_id"),
        field("produced_by_step_id"),
        field("result_kind", enum_type=FactResultKind),
        field("evidence_refs_json", json_value=True),
        field("payload_schema"),
        field("payload_schema_rev"),
        field("payload_json", json_value=True),
    ),
    same_run_refs=(
        SameRunReference(
            "requested_fact", "requested_fact_id", "fact result requested fact"
        ),
        SameRunReference("run_step", "produced_by_step_id", "fact result step"),
    ),
)

MEMORY_ARTIFACT = LineageRowSpec(
    key="memory_artifact",
    label="memory artifact",
    write_type=MemoryArtifactWrite,
    identity=field("memory_artifact_id"),
    fields=(
        field("run_id"),
        field("produced_by_step_id"),
        field("source_kind", enum_type=MemoryArtifactSourceKind),
        field("requested_fact_id"),
        field("fact_result_id"),
        field("payload_schema"),
        field("payload_schema_rev"),
        field("payload_json", json_value=True),
    ),
    same_run_refs=(
        SameRunReference("run_step", "produced_by_step_id", "memory artifact step"),
        SameRunReference(
            "requested_fact",
            "requested_fact_id",
            "memory artifact requested fact",
            required=False,
        ),
        SameRunReference(
            "fact_result",
            "fact_result_id",
            "memory artifact fact result",
            required=False,
        ),
    ),
)

CLARIFICATION_REQUEST = LineageRowSpec(
    key="clarification_request",
    label="clarification request",
    write_type=ClarificationRequestWrite,
    identity=field("clarification_id"),
    fields=(
        field("run_id"),
        field("fact_result_id"),
        field("step_id"),
        field("need", enum_type=ClarificationNeed),
        field("reason", enum_type=ClarificationReason),
        field("payload_json", json_value=True),
    ),
    same_run_refs=(
        SameRunReference(
            "fact_result",
            "fact_result_id",
            "clarification fact result",
            required=False,
        ),
        SameRunReference("run_step", "step_id", "clarification step", required=False),
    ),
)

CLARIFICATION_RESPONSE = LineageRowSpec(
    key="clarification_response",
    label="clarification response",
    write_type=ClarificationResponseWrite,
    identity=field("response_id"),
    fields=(
        field("run_id"),
        field("clarification_id"),
        field("source_message_ref"),
        field("selected_option_id"),
        field("response_text"),
        field("evidence_ref"),
    ),
    same_run_refs=(
        SameRunReference(
            "clarification_request",
            "clarification_id",
            "clarification response request",
        ),
    ),
)

ANSWER = LineageRowSpec(
    key="answer",
    label="answer",
    write_type=AnswerWrite,
    identity=field("answer_id"),
    fields=(field("run_id"), field("run_result_id")),
    same_run_refs=(SameRunReference("run_result", "run_result_id", "answer result"),),
    conflict_lookup=("run_id", "run_result_id"),
)

ANSWER_OUTPUT = LineageRowSpec(
    key="answer_output",
    label="answer output",
    write_type=AnswerOutputWrite,
    identity=field("answer_output_id"),
    fields=(
        field("run_id"),
        field("answer_id"),
        field("fact_result_id"),
        field("output_key"),
        field("value_kind", enum_type=AnswerValueKind),
        field("value_json", json_value=True),
        field("proof_node_refs_json", json_value=True),
    ),
    conflict_lookup=("run_id", "answer_id", "output_key"),
    same_run_refs=(
        SameRunReference("answer", "answer_id", "answer output answer"),
        SameRunReference("fact_result", "fact_result_id", "answer output fact result"),
    ),
)

ANSWER_PRESENTATION = LineageRowSpec(
    key="answer_presentation",
    label="answer presentation",
    write_type=AnswerPresentationWrite,
    identity=field("presentation_id"),
    fields=(
        field("run_id"),
        field("answer_id"),
        field("client_key", enum_type=PresentationClientKey),
        field("locale"),
        field("presentation_kind", enum_type=PresentationKind),
        field("rendered_value"),
        field("render_step_id"),
        field("payload_schema"),
        field("payload_schema_rev"),
        field("payload_json", json_value=True),
    ),
    conflict_lookup=(
        "run_id",
        "answer_id",
        "client_key",
        "locale",
        "presentation_kind",
    ),
    same_run_refs=(
        SameRunReference("answer", "answer_id", "answer presentation answer"),
        SameRunReference(
            "run_step", "render_step_id", "answer presentation render step"
        ),
    ),
)

EXECUTION_PROOF_GRAPH = LineageRowSpec(
    key="execution_proof_graph",
    label="execution proof graph",
    write_type=ExecutionProofGraphWrite,
    identity=field("proof_graph_id"),
    fields=(
        field("run_id"),
        field("fact_result_id"),
        field("compile_step_id"),
        field("execute_step_id"),
        field("payload_schema"),
        field("payload_schema_rev"),
        field("payload_json", json_value=True),
    ),
    conflict_lookup=("run_id", "fact_result_id"),
    same_run_refs=(
        SameRunReference("fact_result", "fact_result_id", "proof graph fact result"),
        SameRunReference("run_step", "compile_step_id", "proof graph compile step"),
        SameRunReference(
            "run_step",
            "execute_step_id",
            "proof graph execute step",
            required=False,
        ),
    ),
)

ALL_RECORD_SPECS = (
    CONVERSATION,
    QUESTION,
    QUESTION_RUN,
    RUN_STEP,
    MODEL_CALL,
    MODEL_CALL_USAGE,
    CATALOG_ENDPOINT,
    SOURCE_READ,
    RUN_ARTIFACT,
    RUN_RESULT,
    RUNTIME_ERROR,
    REQUESTED_FACT,
    FACT_RESULT,
    MEMORY_ARTIFACT,
    CLARIFICATION_REQUEST,
    CLARIFICATION_RESPONSE,
    ANSWER,
    ANSWER_OUTPUT,
    ANSWER_PRESENTATION,
    EXECUTION_PROOF_GRAPH,
)

RECORD_SPECS_BY_KEY = {spec.key: spec for spec in ALL_RECORD_SPECS}
