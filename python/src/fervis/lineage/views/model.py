"""Framework-neutral lineage view model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from fervis.lineage.enums import ContributionOrigin
from fervis.lineage.step_summary import StepSummaryDetail


class LineageRootKind(StrEnum):
    ANSWER = "answer"
    RUN = "run"
    QUESTION = "question"
    CONVERSATION = "conversation"


@dataclass(frozen=True)
class ArtifactInspectionView:
    artifact_kind: str
    artifact_id: str
    size_bytes: int


@dataclass(frozen=True)
class ObservabilityNoticeView:
    kind: str
    severity: str
    message: str
    run_ids: tuple[str, ...] = ()
    details: dict[str, object] | None = None


@dataclass(frozen=True)
class ModelCallInspectionView:
    model_call_id: str
    run_id: str
    step_id: str
    step_key: str
    step_sequence: int
    call_index: int
    provider: str
    model_key: str
    status: str
    prompt_chars: int
    schema_chars: int
    tool_spec_chars: int
    artifacts: tuple[ArtifactInspectionView, ...] = ()


@dataclass(frozen=True)
class ContributionView:
    origin: ContributionOrigin
    label: str
    node_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputLineageResultView:
    fact_result_id: str
    requested_fact_id: str
    fact_description: str
    explicit: tuple[str, ...] = ()
    derived: tuple[str, ...] = ()
    contextual: tuple[str, ...] = ()
    applied: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    proof_handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputLineageView:
    root_kind: LineageRootKind
    root_id: str
    results: tuple[InputLineageResultView, ...]


@dataclass(frozen=True)
class StepDecisionView:
    step_key: str
    lines: tuple[str, ...]
    detail: StepSummaryDetail = StepSummaryDetail.COMPACT
    is_explanation: bool = False


@dataclass(frozen=True)
class SemanticRequestedFactView:
    requested_fact_id: str
    description: str


@dataclass(frozen=True)
class SemanticKnownInputView:
    input_id: str
    text: str
    kind: str
    role: str = ""
    description: str = ""
    resolved_value_text: str = ""


@dataclass(frozen=True)
class SemanticResolverCandidateView:
    input_id: str
    resolver_read_id: str
    resolver_label: str
    basis: str = ""


@dataclass(frozen=True)
class SemanticGroundingResultView:
    input_id: str
    input_text: str
    resolver_read_id: str
    resolver_label: str
    matched_field: str
    matched_value: str
    matched_label: str


@dataclass(frozen=True)
class SemanticInterpretedInputView:
    input_id: str
    input_text: str
    kind: str
    value: str
    label: str = ""
    detail: str = ""


@dataclass(frozen=True)
class SemanticConversationClauseView:
    current_clause_text: str
    current_value_text: str
    resolved_frame_text: str
    resolved_clause_text: str


@dataclass(frozen=True)
class StepSemanticView:
    requested_facts: tuple[SemanticRequestedFactView, ...] = ()
    known_inputs: tuple[SemanticKnownInputView, ...] = ()
    resolver_candidates: tuple[SemanticResolverCandidateView, ...] = ()
    grounding_results: tuple[SemanticGroundingResultView, ...] = ()
    interpreted_inputs: tuple[SemanticInterpretedInputView, ...] = ()
    conversation_clauses: tuple[SemanticConversationClauseView, ...] = ()


@dataclass(frozen=True)
class StepView:
    step_id: str
    step_key: str
    sequence: int
    fact_refs: tuple[str, ...] = ()
    decisions: tuple[StepDecisionView, ...] = ()
    semantic: StepSemanticView = field(default_factory=StepSemanticView)
    error: dict[str, object] | None = None


@dataclass(frozen=True)
class CatalogEndpointView:
    catalog_endpoint_id: str
    catalog_endpoint_key: str
    endpoint_name: str
    framework_kind: str
    source_namespace_kind: str
    source_namespace_path: tuple[str, ...] = ()
    route_method: str = ""
    route_path_template: str = ""
    route_name: str = ""
    api_schema_operation_id: str = ""
    handler_ref: str = ""
    domain_resource_names: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        namespace = "/".join(self.source_namespace_path)
        if namespace:
            return f"{namespace}/{self.endpoint_name}"
        return self.endpoint_name


@dataclass(frozen=True)
class SourceReadView:
    source_read_id: str
    step_id: str
    catalog_endpoint: CatalogEndpointView
    args: dict[str, object]
    row_count: int | None
    response_hash: str
    status: str
    completeness: dict[str, object]
    artifact_id: str | None = None
    error: dict[str, object] | None = None

    @property
    def endpoint_name(self) -> str:
        return self.catalog_endpoint.endpoint_name


@dataclass(frozen=True)
class RuntimeErrorView:
    runtime_error_detail_id: str
    error_kind: str
    message: str
    failed_step_id: str | None = None
    failed_step_key: str | None = None


@dataclass(frozen=True)
class ClarificationRequestView:
    clarification_id: str
    basis: str
    question_text: str
    requested_fact_id: str = ""
    known_input_id: str = ""
    fact_result_id: str | None = None
    step_id: str | None = None
    options: tuple[dict[str, object], ...] = ()
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClarificationResponseView:
    response_id: str
    clarification_id: str
    evidence_ref: str
    source_message_ref: str = ""
    selected_option_id: str = ""
    response_text: str = ""


@dataclass(frozen=True)
class ProofEndpointArgView:
    handle: str
    arg_name: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofAppliedInputView:
    handle: str
    label: str
    action: str
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofComputationLinkView:
    source: str
    target: str
    role: str


@dataclass(frozen=True)
class ExecutionProofView:
    proof_graph_id: str
    evidence_handles: tuple[str, ...]
    endpoint_args: tuple[ProofEndpointArgView, ...]
    computation_links: tuple[ProofComputationLinkView, ...]
    computation_summaries: tuple[str, ...]
    debug_evidence_handles: tuple[str, ...]
    debug_computation_links: tuple[str, ...]
    contributions: tuple[ContributionView, ...]
    applied_inputs: tuple[ProofAppliedInputView, ...]
    source_reads: tuple[SourceReadView, ...] = ()


@dataclass(frozen=True)
class AnswerOutputView:
    fact_result_id: str
    output_key: str
    value_kind: str
    value: str
    value_json: dict[str, object]
    proof_node_refs: tuple[str, ...]
    proof: ExecutionProofView | None = None


@dataclass(frozen=True)
class AnswerPresentationView:
    presentation_id: str
    client_key: str
    locale: str
    presentation_kind: str
    render_step_id: str
    value: str


@dataclass(frozen=True)
class AnswerView:
    answer_id: str
    outputs: tuple[AnswerOutputView, ...]
    presentations: tuple[AnswerPresentationView, ...] = ()


@dataclass(frozen=True)
class MemoryArtifactView:
    memory_artifact_id: str
    source_kind: str
    payload_schema: str
    payload_schema_rev: int
    outcome: str = ""
    source_question: str = ""
    source_answer: str = ""
    address_summaries: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactResultView:
    fact_result_id: str
    produced_by_step_id: str
    result_kind: str
    steps: tuple[StepView, ...] = ()
    proof: ExecutionProofView | None = None
    memory_artifacts: tuple[MemoryArtifactView, ...] = ()


@dataclass(frozen=True)
class RequestedFactView:
    requested_fact_id: str
    produced_by_step_id: str
    fact_key: str
    description: str
    steps: tuple[StepView, ...]
    fact_results: tuple[FactResultView, ...]
    answer_outputs: tuple[AnswerOutputView, ...]
    memory_artifacts: tuple[MemoryArtifactView, ...] = ()


@dataclass(frozen=True)
class RunView:
    run_id: str
    run_number: int
    trigger_kind: str
    result_kind: str
    activated_memory_ids: tuple[str, ...]
    requested_facts: tuple[RequestedFactView, ...]
    answers: tuple[AnswerView, ...]
    source_reads: tuple[SourceReadView, ...]
    steps: tuple[StepView, ...]
    runtime_errors: tuple[RuntimeErrorView, ...] = ()
    memory_artifacts: tuple[MemoryArtifactView, ...] = ()
    trigger_clarification_response_run_id: str | None = None
    trigger_clarification_response_id: str | None = None
    clarification_requests: tuple[ClarificationRequestView, ...] = ()
    clarification_responses: tuple[ClarificationResponseView, ...] = ()


@dataclass(frozen=True)
class TimelineRequestedFactView:
    requested_fact_id: str
    fact_key: str
    description: str


@dataclass(frozen=True)
class TimelineFactResultView:
    fact_result_id: str
    requested_fact_id: str
    result_kind: str
    proof: ExecutionProofView | None = None
    memory_artifacts: tuple[MemoryArtifactView, ...] = ()


@dataclass(frozen=True)
class TimelineAnswerOutputView:
    fact_result_id: str
    output_key: str
    value_kind: str
    value: str
    value_json: dict[str, object]
    proof_node_refs: tuple[str, ...]
    proof: ExecutionProofView | None = None


@dataclass(frozen=True)
class TimelineStepView:
    step_id: str
    step_key: str
    sequence: int
    decisions: tuple[StepDecisionView, ...] = ()
    semantic: StepSemanticView = field(default_factory=StepSemanticView)
    model_calls: tuple[ModelCallInspectionView, ...] = ()
    source_reads: tuple[SourceReadView, ...] = ()
    requested_facts: tuple[TimelineRequestedFactView, ...] = ()
    fact_results: tuple[TimelineFactResultView, ...] = ()
    answer_outputs: tuple[TimelineAnswerOutputView, ...] = ()
    answer_presentations: tuple[AnswerPresentationView, ...] = ()
    clarifications: tuple[ClarificationRequestView, ...] = ()
    runtime_errors: tuple[RuntimeErrorView, ...] = ()


@dataclass(frozen=True)
class TimelineRunView:
    run_id: str
    run_number: int
    trigger_kind: str
    result_kind: str
    activated_memory_ids: tuple[str, ...]
    memory_artifacts: tuple[MemoryArtifactView, ...]
    steps: tuple[TimelineStepView, ...]
    trigger_clarification_response_run_id: str | None = None
    trigger_clarification_response_id: str | None = None
    clarification_responses: tuple[ClarificationResponseView, ...] = ()


@dataclass(frozen=True)
class TimelineQuestionView:
    question_id: str
    conversation_id: str
    text: str
    runs: tuple[TimelineRunView, ...]


@dataclass(frozen=True)
class LineageTimelineView:
    root_kind: LineageRootKind
    root_id: str
    questions: tuple[TimelineQuestionView, ...]
    observability_notices: tuple[ObservabilityNoticeView, ...] = ()


@dataclass(frozen=True)
class QuestionView:
    question_id: str
    conversation_id: str
    text: str
    runs: tuple[RunView, ...]


@dataclass(frozen=True)
class LineageView:
    root_kind: LineageRootKind
    root_id: str
    questions: tuple[QuestionView, ...]
