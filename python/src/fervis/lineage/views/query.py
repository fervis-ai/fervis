"""Framework-neutral lineage view query contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from fervis.lineage.enums import (
    AnswerValueKind,
    FactResultKind,
    PresentationClientKey,
    PresentationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    QuestionRunKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lineage.memory_artifacts import MemoryArtifactRow
from fervis.lookup.clarification import ClarificationNeed, ClarificationReason

JsonObject = dict[str, object]


@dataclass(frozen=True)
class ConversationRow:
    conversation_id: str
    tenant_id: str


@dataclass(frozen=True)
class QuestionRow:
    question_id: str
    conversation_id: str
    conversation_sequence: int
    original_question: str


@dataclass(frozen=True)
class RunRow:
    run_id: str
    question_id: str
    run_number: int
    kind: QuestionRunKind
    trigger_kind: RunTriggerKind
    base_run_id: str | None = None
    trigger_clarification_response_id: str | None = None


@dataclass(frozen=True)
class AnswerProgramRow:
    program_id: str
    schema_revision: int
    canonical_json: str = ""


@dataclass(frozen=True)
class BindingPatchRow:
    patch_id: str
    canonical_json: str = ""


@dataclass(frozen=True)
class ProgramInvocationRow:
    invocation_id: str
    run_id: str
    program_id: str
    bindings_json: str = ""
    patch: BindingPatchRow | None = None
    revision_id: str | None = None


@dataclass(frozen=True)
class ProgramRevisionRow:
    revision_id: str
    base_program_id: str
    revised_program_id: str
    capability_id: str
    application_json: str = ""


@dataclass(frozen=True)
class StepRow:
    step_id: str
    run_id: str
    sequence: int
    step_key: RunStepKey
    kind: RunStepKind
    input_summary_json: JsonObject = field(default_factory=dict)
    output_summary_json: JsonObject = field(default_factory=dict)
    error_json: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class RunResultRow:
    run_result_id: str
    run_id: str
    result_kind: RunResultKind


@dataclass(frozen=True)
class RuntimeErrorRow:
    runtime_error_detail_id: str
    run_id: str
    run_result_id: str
    failed_step_id: str | None
    error_kind: RuntimeErrorKind
    message: str


@dataclass(frozen=True)
class ClarificationRequestRow:
    clarification_id: str
    run_id: str
    need: ClarificationNeed
    reason: ClarificationReason
    payload_json: JsonObject
    fact_result_id: str | None = None
    step_id: str | None = None


@dataclass(frozen=True)
class ClarificationResponseRow:
    response_id: str
    run_id: str
    clarification_id: str
    evidence_ref: str
    source_message_ref: str = ""
    selected_option_id: str = ""
    response_text: str = ""


@dataclass(frozen=True)
class RequestedFactRow:
    requested_fact_id: str
    run_id: str
    produced_by_step_id: str
    fact_key: str
    description: str
    answer_expression_family: str
    requested_fact_json: JsonObject = field(default_factory=dict)
    answer_requests_json: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class FactResultRow:
    fact_result_id: str
    run_id: str
    requested_fact_id: str
    produced_by_step_id: str
    result_kind: FactResultKind
    evidence_refs_json: tuple[str, ...] = ()
    payload_schema: str = ""
    payload_schema_rev: int | None = None
    payload_json: JsonObject | None = None


@dataclass(frozen=True)
class AnswerRow:
    answer_id: str
    run_id: str
    run_result_id: str


@dataclass(frozen=True)
class AnswerOutputRow:
    answer_output_id: str
    run_id: str
    answer_id: str
    fact_result_id: str
    output_key: str
    value_kind: AnswerValueKind
    value_json: JsonObject
    proof_node_refs_json: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPresentationRow:
    presentation_id: str
    run_id: str
    answer_id: str
    client_key: PresentationClientKey
    locale: str
    presentation_kind: PresentationKind
    render_step_id: str
    rendered_value: str = ""
    payload_schema: str = ""
    payload_schema_rev: int | None = None
    payload_json: JsonObject | None = None


@dataclass(frozen=True)
class CatalogEndpointRow:
    catalog_endpoint_id: str
    run_id: str
    catalog_endpoint_key: str
    endpoint_name: str
    framework_kind: str
    source_namespace_kind: str
    source_namespace_path_json: tuple[str, ...]
    route_method: str
    route_path_template: str
    route_name: str = ""
    api_schema_operation_id: str = ""
    handler_ref: str = ""
    domain_resource_names_json: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceReadRow:
    source_read_id: str
    run_id: str
    step_id: str
    catalog_endpoint_id: str
    args_json: JsonObject
    status: SourceReadStatus
    row_count: int | None = None
    completeness_json: JsonObject = field(default_factory=dict)
    response_hash: str = ""
    artifact_id: str | None = None
    error_json: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ProofGraphRow:
    proof_graph_id: str
    run_id: str
    fact_result_id: str
    compile_step_id: str
    execute_step_id: str | None
    payload_schema: str
    payload_schema_rev: int
    payload_json: JsonObject


@dataclass(frozen=True)
class LineageRows:
    conversations: tuple[ConversationRow, ...] = ()
    questions: tuple[QuestionRow, ...] = ()
    runs: tuple[RunRow, ...] = ()
    answer_programs: tuple[AnswerProgramRow, ...] = ()
    program_invocations: tuple[ProgramInvocationRow, ...] = ()
    program_revisions: tuple[ProgramRevisionRow, ...] = ()
    steps: tuple[StepRow, ...] = ()
    run_results: tuple[RunResultRow, ...] = ()
    runtime_errors: tuple[RuntimeErrorRow, ...] = ()
    clarification_requests: tuple[ClarificationRequestRow, ...] = ()
    clarification_responses: tuple[ClarificationResponseRow, ...] = ()
    requested_facts: tuple[RequestedFactRow, ...] = ()
    fact_results: tuple[FactResultRow, ...] = ()
    memory_artifacts: tuple[MemoryArtifactRow, ...] = ()
    answers: tuple[AnswerRow, ...] = ()
    answer_outputs: tuple[AnswerOutputRow, ...] = ()
    answer_presentations: tuple[AnswerPresentationRow, ...] = ()
    catalog_endpoints: tuple[CatalogEndpointRow, ...] = ()
    source_reads: tuple[SourceReadRow, ...] = ()
    proof_graphs: tuple[ProofGraphRow, ...] = ()


class LineageQueryPort(Protocol):
    def run_id_for_answer(self, answer_id: str) -> str | None: ...

    def run_by_id(self, run_id: str) -> RunRow | None: ...

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]: ...

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]: ...

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]: ...

    def lineage_rows_for_run_ids(self, run_ids: tuple[str, ...]) -> LineageRows: ...
