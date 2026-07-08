"""Lookup runtime pipeline."""

import json
from dataclasses import dataclass, field
from typing import Any

from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.errors import ErrorCode
from fervis.observability.event_contracts import EventPayloadKey
from fervis.lookup.relation_catalog import validate_relation_catalog
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRequest,
    CatalogSelectionResult,
    ResolverCatalogSelectionRequest,
    select_relation_catalog,
    select_resolver_relation_catalog,
)
from fervis.lookup.conversation_resolution import (
    ConversationResolutionGenerationError,
    ConversationResolutionKind,
    ConversationResolutionRequest,
    conversation_resolution_overlay_from,
    generate_conversation_resolution,
)
from fervis.lookup.plan_execution.authorized_sources import (
    AuthorizedExecutionSources,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.runner import execute_fact_plan
from fervis.lookup.grounding.resolution import (
    ground_question_inputs,
    GroundingSourceReadError,
)
from fervis.lookup.grounding.turn import GroundingGenerationError
from fervis.lookup.memory.projection import project_lookup_memory
from fervis.lookup.memory.outcomes import fact_value_memory_addresses
from fervis.lookup.outcomes.model import (
    FactResult,
    NeedsClarification,
)
from fervis.lookup.outcomes.answerability import classify_plan_impossible
from fervis.lookup.clarification import (
    AmbiguousQuestionInterpretation,
    Clarification,
    ClarificationOption,
    clarify,
)
from fervis.lookup.plan_selection import (
    BoundPlanSelectionSet,
    PlanSelectionGenerationError,
    PlanSelectionRequest,
    PlanSelectionSet,
    generate_plan_selection,
)
from fervis.lookup.fact_plan.fact_plan import (
    FactPlan,
    PlanClarification,
    PlanImpossible,
)
from fervis.lookup.fact_plan.values import FactValue
from fervis.lookup.fact_planning.request import FactPlanRequest
from fervis.lookup.fact_planning.turn import (
    FactPlanGenerationError,
    generate_pattern_fact_plan,
)
from fervis.lookup.turn_prompts.context import active_clarification_context
from fervis.lookup.query_enrichment import (
    QueryEnrichmentGenerationError,
    QueryEnrichmentRequest,
    generate_query_enrichment,
)
from fervis.lookup.question_contract import (
    QuestionContractGenerationError,
    QuestionContractNeedsClarification,
    QuestionContractRequest,
    generate_question_contract,
)
from fervis.lookup.read_eligibility import (
    READ_ELIGIBILITY_RECALL_READS_PER_FACT,
    ReadEligibilityGenerationError,
    ReadEligibilityRequest,
    filter_catalog_selection_for_read_eligibility,
    generate_read_eligibility,
    prepare_catalog_selection_for_read_eligibility,
)
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupRuntimePorts,
)
from fervis.lookup.orchestration.result import LookupResult, RunStatus
from fervis.lookup.source_binding import (
    SourceBindingGenerationError,
    SourceBindingPlan,
    SourceBindingRequest,
    SourceCandidateDiscoveryRequest,
    generate_source_binding,
    source_candidate_discovery_payload,
)
from fervis.lookup.source_binding.role_selection import (
    bound_plan_selection_for_source_binding,
    plan_selection_uses_only_values,
    value_only_source_binding_plan,
)
from fervis.lookup.memory.available_values import (
    active_memory_operation_values,
    active_memory_reference_values,
)
from fervis.lookup.memory.projection import (
    ConversationMemoryProjectionOverflow,
    project_conversation_memory_cards,
)
from fervis.memory.conversation_context import expand_activated_memory_cards
from fervis.memory.projection import fact_artifacts_from_context
from fervis.lookup.lineage.source_reads import SourceReadLineageScope
from fervis.lineage.recorder import (
    CatalogEndpointWrite,
    RunStepWrite,
    SourceReadWrite,
)
from .model_turn_events import _model_turn_event_payload
from fervis.lookup.lineage.steps import (
    compile_step_id,
    execution_step_id,
    lineage_error_json,
    lineage_model_turn_output_summary,
    model_turn_step_id,
    record_compile_step,
    record_execution_step,
    record_model_turn_audit,
    record_model_turn_step,
    record_step_source_context,
)
from fervis.lookup.lineage.results import (
    LineagePersistenceUnavailable,
    RuntimeErrorTerminal,
    record_runtime_error_lineage,
    runtime_error_terminal_result,
)
from fervis.lookup.lineage.step_summaries import add_grounding_result_semantics
from .result_synthesis import _synthesize_result
from .terminal_results import (
    _execution_failure_payload,
    _grounding_issue_fact_result,
    _plan_clarification_fact_result,
    _plan_validation_failed_result,
    _question_contract_clarification_fact_result,
    _status_for_fact_result,
)
from .limits import _limit_before_next_model_turn, _merge_usage


@dataclass
class _LookupPipelineState:
    request: LookupRequest
    ports: LookupRuntimePorts
    memory: Any
    provider: str
    model_key: str
    memory_card_projection: Any = None
    activated_memory: Any = None
    conversation_turn: Any = None
    conversation_resolution: Any = None
    conversation_resolution_overlay: Any = None
    question_turn: Any = None
    question_contract: Any = None
    full_catalog: Any = None
    query_enrichment_turn: Any = None
    query_enrichment_usage: dict[str, Any] | None = None
    catalog_selection: CatalogSelectionResult | None = None
    resolver_catalog_selection: Any = None
    read_eligibility_turn: Any = None
    read_eligibility_usage: dict[str, Any] | None = None
    plan_selection_turn_number: int = 3
    plan_selection_turn: Any = None
    plan_selection_outcome: Any = None
    catalog: Any = None
    grounding: Any = None
    grounding_usage: dict[str, Any] | None = None
    source_binding_turn_number: int = 3
    source_binding_turn: Any = None
    source_binding_outcome: Any = None
    fact_plan_request: FactPlanRequest | None = None
    bound_plan_selection: Any = None
    pattern_plan_turn_number: int = 5
    plan_turn: Any = None


@dataclass
class _SourceReadLineageBuffer:
    catalog_endpoints: list[CatalogEndpointWrite] = field(default_factory=list)
    source_reads: list[SourceReadWrite] = field(default_factory=list)

    def record_catalog_endpoint(
        self,
        catalog_endpoint: CatalogEndpointWrite,
    ) -> CatalogEndpointWrite:
        self.catalog_endpoints.append(catalog_endpoint)
        return catalog_endpoint

    def record_source_read(self, source_read: SourceReadWrite) -> SourceReadWrite:
        self.source_reads.append(source_read)
        return source_read


@dataclass
class _SourceReadLineage:
    source_read_lineage: SourceReadLineageScope | None = None
    source_read_buffer: _SourceReadLineageBuffer | None = None

    @property
    def source_reads(self) -> tuple[SourceReadWrite, ...]:
        if self.source_read_buffer is None:
            return ()
        return tuple(self.source_read_buffer.source_reads)

    @property
    def catalog_endpoints(self) -> tuple[CatalogEndpointWrite, ...]:
        if self.source_read_buffer is None:
            return ()
        return _dedupe_catalog_endpoints(self.source_read_buffer.catalog_endpoints)


def run_lookup_question(
    request: LookupRequest,
    ports: LookupRuntimePorts,
) -> LookupResult:
    try:
        memory_card_projection = project_conversation_memory_cards(
            request.conversation_context,
            current_question=request.question,
        )
    except ConversationMemoryProjectionOverflow:
        return _runtime_error_terminal_from_ports(
            request=request,
            ports=ports,
            error_code=ErrorCode.PLANNING_FAILED,
            message="conversation memory projection exceeded the prompt budget",
            usage={},
        )
    state = _LookupPipelineState(
        request=request,
        ports=ports,
        memory=project_lookup_memory(request.conversation_context),
        memory_card_projection=memory_card_projection,
        provider=str(request.provider_preferences.get("provider") or ""),
        model_key=str(request.provider_preferences.get("modelKey") or ""),
    )
    try:
        for phase in (
            _run_conversation_resolution_phase,
            _run_question_contract_phase,
            _run_query_enrichment_and_catalog_phase,
            _run_grounding_phase,
            _run_read_eligibility_phase,
            _run_plan_selection_phase,
            _run_source_binding_phase,
            _run_planning_phase,
        ):
            result = phase(state)
            if result is not None:
                return result
        return _run_execution_phase(state)
    except LineagePersistenceUnavailable:
        return RuntimeErrorTerminal(
            run_id=request.run_id,
            error_code=ErrorCode.LINEAGE_PERSISTENCE_FAILED,
            message=ErrorCode.LINEAGE_PERSISTENCE_FAILED,
        ).lookup_result()


def _runtime_error_terminal(
    state: _LookupPipelineState,
    *,
    error_code: str,
    message: str,
    usage: dict[str, Any],
    failed_step_id: str | None = None,
) -> LookupResult:
    return _runtime_error_terminal_from_ports(
        request=state.request,
        ports=state.ports,
        error_code=error_code,
        message=message,
        usage=usage,
        failed_step_id=failed_step_id,
    )


def _runtime_error_terminal_from_ports(
    *,
    request: LookupRequest,
    ports: LookupRuntimePorts,
    error_code: str,
    message: str,
    usage: dict[str, Any],
    failed_step_id: str | None = None,
) -> LookupResult:
    sink = ports.lineage_step_sink
    return runtime_error_terminal_result(
        RuntimeErrorTerminal(
            run_id=request.run_id,
            failed_step_id=failed_step_id,
            error_code=error_code,
            message=message,
            usage=usage,
        ),
        recorder=sink.recorder if sink is not None else None,
        lineage_required=getattr(ports, "lineage_required", False),
    )


def _emit_progress(
    state: _LookupPipelineState,
    *,
    stage: str,
    message: str,
) -> None:
    sink = state.ports.progress_sink
    if sink is None:
        return
    sink.emit(
        {
            "event": "run.progress",
            "run_id": state.request.run_id,
            "stage": stage,
            "message": message,
        }
    )


def _run_conversation_resolution_phase(
    state: _LookupPipelineState,
) -> LookupResult | None:
    context_sources = tuple(
        getattr(state.memory_card_projection, "context_sources", ()) or ()
    )
    context_frames = tuple(
        getattr(state.memory_card_projection, "context_frames", ()) or ()
    )
    if (
        not context_sources
        and active_clarification_context(
            state.request.conversation_context,
            current_question=state.request.question,
        )
        is None
    ):
        return None
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    _emit_progress(
        state,
        stage="conversation_resolution",
        message="resolving conversation context",
    )
    try:
        state.conversation_turn = generate_conversation_resolution(
            request=ConversationResolutionRequest(
                question=state.request.question,
                conversation_context=state.request.conversation_context,
                host=state.request.host,
                context_sources=context_sources,
                context_frames=context_frames,
            ),
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except ConversationResolutionGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.CONVERSATION_RESOLUTION,
            turn=1,
            exc=exc,
            usage=exc.usage,
        )
    _append_model_turn_completed(
        state,
        phase=ModelTurnPurpose.CONVERSATION_RESOLUTION,
        turn=1,
        model_turn=state.conversation_turn,
    )
    state.conversation_resolution = state.conversation_turn.result.outcome
    state.conversation_resolution_overlay = conversation_resolution_overlay_from(
        state.conversation_resolution,
        memory_projection=state.memory_card_projection,
    )
    if (
        state.conversation_resolution.resolution
        == ConversationResolutionKind.NEEDS_CLARIFICATION
    ):
        fact_result = _conversation_resolution_ambiguity_fact_result(
            state.conversation_resolution.unresolved
        )
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=fact_result,
            status=RunStatus.NEEDS_CLARIFICATION,
            usage=state.conversation_turn.usage,
            question_contract=None,
            grounded_values=(),
            question_contract_step_id=model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.CONVERSATION_RESOLUTION,
                turn=1,
            ),
        )
    if (
        state.conversation_resolution.resolution
        == ConversationResolutionKind.STANDALONE
        and active_clarification_context(
            state.request.conversation_context,
            current_question=state.request.question,
        )
        is not None
    ):
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLANNING_FAILED,
            message="clarification response was not connected to the active clarification",
            usage=state.conversation_turn.usage,
        )
    activation_error = _activate_selected_memory(state)
    if activation_error is not None:
        return activation_error
    return None


def _run_question_contract_phase(state: _LookupPipelineState) -> LookupResult | None:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    _emit_progress(
        state,
        stage="question_contract",
        message="normalizing requested fact",
    )
    try:
        state.question_turn = generate_question_contract(
            request=QuestionContractRequest(
                current_question=state.request.question,
                conversation_context=state.request.conversation_context,
                conversation_resolution_overlay=state.conversation_resolution_overlay,
                host=state.request.host,
            ),
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except QuestionContractGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.QUESTION_CONTRACT,
            turn=_question_turn_number(state),
            exc=exc,
            usage=_phase_usage(state, exc.usage),
        )
    _append_model_turn_completed(
        state,
        phase=ModelTurnPurpose.QUESTION_CONTRACT,
        turn=_question_turn_number(state),
        model_turn=state.question_turn,
    )
    outcome = state.question_turn.result.outcome
    if isinstance(outcome, QuestionContractNeedsClarification):
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=_question_contract_clarification_fact_result(outcome),
            status=RunStatus.NEEDS_CLARIFICATION,
            usage=_phase_usage(state),
            question_contract=None,
            grounded_values=(),
            question_contract_step_id=model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=_question_turn_number(state),
            ),
        )
    state.question_contract = outcome
    state.full_catalog = validate_relation_catalog(
        state.ports.relation_catalog_port.build_relation_catalog()
    )
    return None


def _run_query_enrichment_and_catalog_phase(
    state: _LookupPipelineState,
) -> LookupResult | None:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    _emit_progress(
        state,
        stage="query_enrichment",
        message="matching question terms to API resources",
    )
    try:
        state.query_enrichment_turn = generate_query_enrichment(
            request=QueryEnrichmentRequest(
                question=state.request.question,
                conversation_context=state.request.conversation_context,
                requested_facts=state.question_contract.requested_facts,
                relation_catalog=state.full_catalog,
                conversation_resolution_overlay=state.conversation_resolution_overlay,
                host=state.request.host,
            ),
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except QueryEnrichmentGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.QUERY_ENRICHMENT,
            turn=_query_enrichment_turn_number(state),
            exc=exc,
            usage=_phase_usage(state, exc.usage),
        )
    _append_model_turn_completed(
        state,
        phase=ModelTurnPurpose.QUERY_ENRICHMENT,
        turn=_query_enrichment_turn_number(state),
        model_turn=state.query_enrichment_turn,
    )
    state.query_enrichment_usage = state.query_enrichment_turn.usage
    state.resolver_catalog_selection = select_resolver_relation_catalog(
        ResolverCatalogSelectionRequest(
            relation_catalog=state.full_catalog,
            entity_target_catalog_search_terms=(
                state.query_enrichment_turn.result.entity_target_catalog_search_terms
            ),
        )
    )
    return None


def _run_grounding_phase(state: _LookupPipelineState) -> LookupResult | None:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    _emit_progress(
        state,
        stage="grounding",
        message="grounding question inputs",
    )
    grounding_lineage = _grounding_source_lineage(state)
    try:
        state.grounding = ground_question_inputs(
            question=state.request.question,
            question_contract=state.question_contract,
            full_catalog=state.full_catalog,
            resolver_catalog=state.resolver_catalog_selection.relation_catalog,
            data_access_port=state.ports.data_access_port,
            runtime_values=state.request.runtime_values,
            conversation_context=state.request.conversation_context,
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
            resolver_selections=(
                state.resolver_catalog_selection.entity_target_selections
            ),
            active_memory_ids=_active_memory_ids(state),
            conversation_resolution_overlay=state.conversation_resolution_overlay,
            source_read_lineage=grounding_lineage.source_read_lineage,
            host=state.request.host,
        )
    except GroundingGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.GROUNDING,
            turn=_grounding_turn_number(state),
            exc=exc,
            usage=_phase_usage(state, exc.usage),
        )
    except GroundingSourceReadError as exc:
        grounding_step = None
        if exc.turn is not None:
            grounding_step = _append_model_turn_completed(
                state,
                phase=ModelTurnPurpose.GROUNDING,
                turn=_grounding_turn_number(state),
                model_turn=exc.turn,
            )
            record_step_source_context(
                state.ports,
                step=grounding_step,
                catalog_endpoints=grounding_lineage.catalog_endpoints,
                source_reads=grounding_lineage.source_reads,
            )
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.FRAMEWORK_ADAPTER_FAILED,
            message=str(exc),
            usage=_phase_usage(state, exc.usage),
            failed_step_id=(
                grounding_step.step_id if grounding_step is not None else None
            ),
        )
    if state.grounding.turn is not None:
        grounding_step = _append_model_turn_completed(
            state,
            phase=ModelTurnPurpose.GROUNDING,
            turn=_grounding_turn_number(state),
            model_turn=state.grounding.turn,
        )
        record_step_source_context(
            state.ports,
            step=grounding_step,
            catalog_endpoints=grounding_lineage.catalog_endpoints,
            source_reads=grounding_lineage.source_reads,
        )
    state.grounding_usage = state.grounding.usage
    _select_answer_reads_for_eligibility(state)
    if state.grounding.ledger.issues:
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=_grounding_issue_fact_result(state.grounding.ledger.issues),
            status=RunStatus.NEEDS_CLARIFICATION,
            usage=_phase_usage(state, state.grounding_usage),
            question_contract=state.question_contract,
            grounded_values=state.grounding.ledger.values,
            question_contract_step_id=model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=_question_turn_number(state),
            ),
        )
    return None


def _run_read_eligibility_phase(state: _LookupPipelineState) -> LookupResult | None:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    _emit_progress(
        state,
        stage="read_eligibility",
        message="selecting candidate reads",
    )
    try:
        state.read_eligibility_turn = generate_read_eligibility(
            request=ReadEligibilityRequest(
                question=state.request.question,
                question_contract=state.question_contract,
                requested_facts=state.question_contract.requested_facts,
                catalog_selection=state.catalog_selection,
                conversation_context=state.request.conversation_context,
                conversation_resolution_overlay=state.conversation_resolution_overlay,
                available_values=_catalog_available_values_for_state(state),
                host=state.request.host,
            ),
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except ReadEligibilityGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.READ_ELIGIBILITY,
            turn=_read_eligibility_turn_number(state),
            exc=exc,
            usage=_phase_usage(state, exc.usage),
        )
    _append_model_turn_completed(
        state,
        phase=ModelTurnPurpose.READ_ELIGIBILITY,
        turn=_read_eligibility_turn_number(state),
        model_turn=state.read_eligibility_turn,
    )
    state.read_eligibility_usage = state.read_eligibility_turn.usage
    state.catalog_selection = filter_catalog_selection_for_read_eligibility(
        catalog_selection=state.catalog_selection,
        read_eligibility=state.read_eligibility_turn.result,
    )
    state.catalog = validate_relation_catalog(state.catalog_selection.relation_catalog)
    return None


def _select_answer_reads_for_eligibility(
    state: _LookupPipelineState,
) -> None:
    state.catalog_selection = select_relation_catalog(
        CatalogSelectionRequest(
            relation_catalog=state.full_catalog,
            requested_facts=state.question_contract.requested_facts,
            max_reads_per_fact=READ_ELIGIBILITY_RECALL_READS_PER_FACT,
            resource_name_matches=(
                state.query_enrichment_turn.result.requested_fact_resource_name_matches
            ),
            active_memory_signals=(),
            available_values=_catalog_available_values_for_state(state),
        )
    )
    state.catalog_selection = prepare_catalog_selection_for_read_eligibility(
        catalog_selection=state.catalog_selection,
        full_catalog=state.full_catalog,
        max_reads_per_fact=READ_ELIGIBILITY_RECALL_READS_PER_FACT,
    )
    state.catalog = validate_relation_catalog(state.catalog_selection.relation_catalog)


def _run_source_binding_phase(state: _LookupPipelineState) -> LookupResult | None:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    state.source_binding_turn_number = state.plan_selection_turn_number + 1
    if _selected_plan_uses_only_values(state):
        assert isinstance(state.plan_selection_outcome, PlanSelectionSet)
        state.source_binding_outcome = value_only_source_binding_plan(
            state.plan_selection_outcome,
            requested_facts=state.question_contract.requested_facts,
        )
        _set_fact_plan_request_from_source_binding(state)
        return None
    if not isinstance(state.plan_selection_outcome, PlanSelectionSet):
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message="plan selection did not produce a valid source-selection set",
            usage=_plan_selection_usage(state),
        )
    source_binding_request = _source_binding_request_for_state(
        state,
        plan_selection=state.plan_selection_outcome,
    )
    _emit_progress(
        state,
        stage="source_binding",
        message="selecting source read",
    )
    try:
        state.source_binding_turn = generate_source_binding(
            request=source_binding_request,
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except SourceBindingGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.SOURCE_BINDING,
            turn=state.source_binding_turn_number,
            exc=exc,
            usage=_phase_usage(state, exc.usage),
        )
    for index, subturn in enumerate(state.source_binding_turn.subturns):
        _append_model_turn_completed(
            state,
            phase=ModelTurnPurpose.SOURCE_BINDING,
            turn=state.source_binding_turn_number + index,
            model_turn=subturn,
        )
    state.source_binding_outcome = state.source_binding_turn.result.outcome
    terminal = _source_binding_terminal_result(state)
    if terminal is not None:
        return terminal
    if not isinstance(state.source_binding_outcome, SourceBindingPlan):
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message="source binding did not produce a valid binding plan",
            usage=_source_binding_usage(state),
        )
    _set_fact_plan_request_from_source_binding(state)
    return None


def _set_fact_plan_request_from_source_binding(state: _LookupPipelineState) -> None:
    assert isinstance(state.source_binding_outcome, SourceBindingPlan)
    state.fact_plan_request = FactPlanRequest(
        question=state.request.question,
        question_contract=state.question_contract,
        relation_catalog=state.catalog,
        bound_sources=state.source_binding_outcome.bound_sources,
        same_scope_relation_catalog=state.full_catalog,
        memory_inputs=_active_memory_prompt_context(state),
        catalog_selection=state.catalog_selection,
        available_values=_available_values_for_state(state),
        available_value_uses=state.grounding.ledger.uses,
        conversation_context=state.request.conversation_context,
        host=state.request.host,
    )


def _selected_plan_uses_only_values(state: _LookupPipelineState) -> bool:
    if not isinstance(state.plan_selection_outcome, PlanSelectionSet):
        return False
    return plan_selection_uses_only_values(state.plan_selection_outcome)


def _run_plan_selection_phase(state: _LookupPipelineState) -> LookupResult | None:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    state.plan_selection_turn_number = _source_binding_base_turn_number(state) + 1
    _emit_progress(
        state,
        stage="plan_selection",
        message="choosing answer strategy",
    )
    try:
        state.plan_selection_turn = generate_plan_selection(
            request=PlanSelectionRequest(
                question=state.request.question,
                question_contract=state.question_contract,
                requested_facts=state.question_contract.requested_facts,
                relation_catalog=state.catalog,
                source_candidate_payload=source_candidate_discovery_payload(
                    _source_candidate_discovery_request_for_state(state)
                ),
                conversation_context=state.request.conversation_context,
                conversation_resolution_overlay=state.conversation_resolution_overlay,
                host=state.request.host,
            ),
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except PlanSelectionGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.PLAN_SELECTION,
            turn=state.plan_selection_turn_number,
            exc=exc,
            usage=_phase_usage(state, exc.usage),
        )
    _append_model_turn_completed(
        state,
        phase=ModelTurnPurpose.PLAN_SELECTION,
        turn=state.plan_selection_turn_number,
        model_turn=state.plan_selection_turn,
    )
    state.plan_selection_outcome = state.plan_selection_turn.result.outcome
    if isinstance(state.plan_selection_outcome, PlanImpossible):
        return _verified_impossible_result(
            state,
            state.plan_selection_outcome,
            usage=_plan_selection_usage(state),
        )
    if not isinstance(state.plan_selection_outcome, PlanSelectionSet):
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message="plan selection did not produce a valid source-selection set",
            usage=_plan_selection_usage(state),
        )
    return None


def _source_binding_request_for_state(
    state: _LookupPipelineState,
    *,
    plan_selection: PlanSelectionSet,
    available_values: tuple[FactValue, ...] | None = None,
) -> SourceBindingRequest:
    return SourceBindingRequest(
        question=state.request.question,
        question_contract=state.question_contract,
        requested_facts=state.question_contract.requested_facts,
        relation_catalog=state.catalog,
        same_scope_relation_catalog=state.full_catalog,
        memory_inputs=_active_memory_prompt_context(state),
        active_memory_ids=tuple(_active_memory_prompt_ids(state)),
        catalog_selection=state.catalog_selection,
        available_values=(
            available_values
            if available_values is not None
            else _source_binding_available_values_for_state(state)
        ),
        available_value_uses=state.grounding.ledger.uses,
        read_eligibility=(
            state.read_eligibility_turn.result
            if state.read_eligibility_turn is not None
            else None
        ),
        plan_selection=plan_selection,
        conversation_context=state.request.conversation_context,
        conversation_resolution_overlay=state.conversation_resolution_overlay,
        host=state.request.host,
    )


def _source_candidate_discovery_request_for_state(
    state: _LookupPipelineState,
) -> SourceCandidateDiscoveryRequest:
    return SourceCandidateDiscoveryRequest(
        question=state.request.question,
        question_contract=state.question_contract,
        requested_facts=state.question_contract.requested_facts,
        relation_catalog=state.catalog,
        same_scope_relation_catalog=state.full_catalog,
        memory_inputs=_active_memory_prompt_context(state),
        active_memory_ids=tuple(_active_memory_prompt_ids(state)),
        catalog_selection=state.catalog_selection,
        available_values=_available_values_for_state(state),
        available_value_uses=state.grounding.ledger.uses,
        read_eligibility=(
            state.read_eligibility_turn.result
            if state.read_eligibility_turn is not None
            else None
        ),
        conversation_context=state.request.conversation_context,
        conversation_resolution_overlay=state.conversation_resolution_overlay,
        host=state.request.host,
    )


def _run_planning_phase(state: _LookupPipelineState) -> LookupResult | None:
    state.pattern_plan_turn_number = (
        state.source_binding_turn_number + _source_binding_model_turn_count(state)
    )
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    state.bound_plan_selection = _bound_plan_selection_from_plan_selection(state)
    if state.bound_plan_selection is None:
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message="source binding did not match the selected source plan",
            usage=_source_binding_usage(state),
        )
    _emit_progress(
        state,
        stage="fact_planning",
        message="building answer plan",
    )
    try:
        state.plan_turn = generate_pattern_fact_plan(
            request=state.fact_plan_request,
            plan_selection=state.bound_plan_selection,
            model_port=state.ports.planner_model_port,
            provider=state.provider,
            model_key=state.model_key,
            max_thinking_tokens=state.request.max_thinking_tokens,
        )
    except FactPlanGenerationError as exc:
        return _model_turn_failure_result(
            state,
            phase=ModelTurnPurpose.PATTERN_FACT_PLANNING,
            turn=state.pattern_plan_turn_number,
            exc=exc,
            usage=_merge_usage(_source_binding_usage(state), exc.usage),
        )
    _append_model_turn_completed(
        state,
        phase=ModelTurnPurpose.PATTERN_FACT_PLANNING,
        turn=state.pattern_plan_turn_number,
        model_turn=state.plan_turn,
    )
    return _pattern_plan_terminal_result(state)


def _bound_plan_selection_from_plan_selection(
    state: _LookupPipelineState,
) -> BoundPlanSelectionSet | None:
    if not isinstance(state.plan_selection_outcome, PlanSelectionSet):
        return None
    if not isinstance(state.source_binding_outcome, SourceBindingPlan):
        return None
    requested_facts = tuple(
        getattr(getattr(state, "question_contract", None), "requested_facts", ())
    )
    if not requested_facts:
        return None
    return bound_plan_selection_for_source_binding(
        state.plan_selection_outcome,
        state.source_binding_outcome,
        requested_facts=requested_facts,
    )


def _run_execution_phase(state: _LookupPipelineState) -> LookupResult:
    execution_sources = _authorized_execution_sources(state)
    execution_lineage = _execution_lineage(state)
    _emit_progress(
        state,
        stage="execution",
        message="reading source",
    )
    try:
        execution = execute_fact_plan(
            plan=state.plan_turn.plan,
            question_contract=state.question_contract,
            catalog=execution_sources.relation_catalog,
            catalog_selection=state.catalog_selection,
            data_access_port=state.ports.data_access_port,
            memory=state.memory,
            available_values=_available_values_for_state(state),
            available_value_uses=state.grounding.ledger.uses,
            authorized_sources=execution_sources,
            source_read_lineage=execution_lineage.source_read_lineage,
        )
    except VerificationError as exc:
        payload = _execution_failure_payload(
            request=state.request,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            exc=exc,
        )
        failed_step = _record_execution_failure_step(
            state,
            payload,
            execution_lineage,
        )
        record_runtime_error_lineage(
            request=state.request,
            ports=state.ports,
            failed_step_id=failed_step.step_id if failed_step is not None else None,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message=str(exc),
        )
        return LookupResult(
            status=RunStatus.FAILED,
            error=ErrorCode.PLAN_VALIDATION_FAILED,
            usage=_pattern_plan_usage(state),
        )
    except Exception as exc:
        payload = _execution_failure_payload(
            request=state.request,
            error_code=ErrorCode.FACT_PLAN_EXECUTION_FAILED,
            exc=exc,
        )
        failed_step = _record_execution_failure_step(
            state,
            payload,
            execution_lineage,
        )
        record_runtime_error_lineage(
            request=state.request,
            ports=state.ports,
            failed_step_id=failed_step.step_id if failed_step is not None else None,
            error_code=ErrorCode.FACT_PLAN_EXECUTION_FAILED,
            message=str(exc),
        )
        return LookupResult(
            status=RunStatus.FAILED,
            error=ErrorCode.FACT_PLAN_EXECUTION_FAILED,
            usage=_pattern_plan_usage(state),
        )
    _record_execution_lineage_step(state, execution, execution_lineage)
    if execution.issue is not None or execution.fact_result is None:
        error_code = (
            execution.issue.kind.value
            if execution.issue is not None
            else ErrorCode.FACT_PLAN_EXECUTION_FAILED
        )
        record_runtime_error_lineage(
            request=state.request,
            ports=state.ports,
            failed_step_id=execution_step_id(state.ports),
            error_code=error_code,
            message=(
                execution.issue.message
                if execution.issue is not None
                else ErrorCode.FACT_PLAN_EXECUTION_FAILED
            ),
        )
        return LookupResult(
            status=RunStatus.FAILED,
            error=error_code,
            usage=_pattern_plan_usage(state),
        )
    return _synthesize_result(
        request=state.request,
        ports=state.ports,
        fact_result=execution.fact_result,
        status=_status_for_fact_result(execution.fact_result),
        usage=_pattern_plan_usage(state),
        question_contract=state.question_contract,
        grounded_values=state.grounding.ledger.values,
        extra_fact_addresses=fact_value_memory_addresses(state.grounding.ledger.values),
        known_input_step_id=_known_input_step_id(state),
        question_contract_step_id=model_turn_step_id(
            state.ports,
            purpose=ModelTurnPurpose.QUESTION_CONTRACT,
            turn=_question_turn_number(state),
        ),
        compile_step_id=compile_step_id(state.ports),
        execute_step_id=execution_step_id(state.ports),
        proof_graph=execution.proof_graph,
        answer_plan=execution.answer_plan,
        proof_node_refs_by_render_output_id=(
            execution.proof_node_refs_by_render_output_id
        ),
        conversation_resolution_activation=_conversation_resolution_activation(state),
    )


def _source_binding_terminal_result(
    state: _LookupPipelineState,
) -> LookupResult | None:
    if isinstance(state.source_binding_outcome, PlanClarification):
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=_plan_clarification_fact_result(
                state.source_binding_outcome,
                catalog=state.catalog,
                memory_relations=state.memory.relations,
            ),
            status=RunStatus.NEEDS_CLARIFICATION,
            usage=_source_binding_usage(state),
            question_contract=state.question_contract,
            grounded_values=state.grounding.ledger.values,
            question_contract_step_id=model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=_question_turn_number(state),
            ),
            conversation_resolution_activation=_conversation_resolution_activation(
                state
            ),
        )
    if isinstance(state.source_binding_outcome, PlanImpossible):
        return _verified_impossible_result(
            state,
            state.source_binding_outcome,
            usage=_source_binding_usage(state),
        )
    return None


def _pattern_plan_terminal_result(state: _LookupPipelineState) -> LookupResult | None:
    verified_plan = state.plan_turn.plan
    plan_outcome = verified_plan.outcome
    if isinstance(plan_outcome, (PlanClarification, PlanImpossible)):
        try:
            verified_plan = _verify_plan(state, verified_plan)
            plan_outcome = verified_plan.outcome
        except VerificationError as exc:
            return _plan_validation_failed_result(
                request=state.request,
                ports=state.ports,
                usage=_pattern_plan_usage(state),
                exc=exc,
            )
    if isinstance(plan_outcome, PlanClarification):
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=_plan_clarification_fact_result(
                plan_outcome,
                catalog=state.catalog,
                memory_relations=state.memory.relations,
            ),
            status=RunStatus.NEEDS_CLARIFICATION,
            usage=_pattern_plan_usage(state),
            question_contract=state.question_contract,
            grounded_values=state.grounding.ledger.values,
            question_contract_step_id=model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=_question_turn_number(state),
            ),
            conversation_resolution_activation=_conversation_resolution_activation(
                state
            ),
        )
    if isinstance(plan_outcome, PlanImpossible):
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=classify_plan_impossible(
                plan_outcome,
                question_contract=state.question_contract,
            ),
            status=RunStatus.COMPLETED,
            usage=_pattern_plan_usage(state),
            question_contract=state.question_contract,
            grounded_values=state.grounding.ledger.values,
            question_contract_step_id=model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=_question_turn_number(state),
            ),
            conversation_resolution_activation=_conversation_resolution_activation(
                state
            ),
        )
    if not plan_outcome.fulfillment:
        return _plan_validation_failed_result(
            request=state.request,
            ports=state.ports,
            usage=_pattern_plan_usage(state),
            exc=VerificationError("answer plan requires fulfillment"),
        )
    return None


def _verified_impossible_result(
    state: _LookupPipelineState,
    outcome: PlanImpossible,
    *,
    usage: dict[str, Any],
) -> LookupResult:
    try:
        verified_impossible = _verify_plan(state, FactPlan(outcome=outcome)).outcome
    except VerificationError as exc:
        return _plan_validation_failed_result(
            request=state.request,
            ports=state.ports,
            usage=usage,
            exc=exc,
        )
    return _synthesize_result(
        request=state.request,
        ports=state.ports,
        fact_result=classify_plan_impossible(
            verified_impossible,
            question_contract=state.question_contract,
        ),
        status=RunStatus.COMPLETED,
        usage=usage,
        question_contract=state.question_contract,
        grounded_values=state.grounding.ledger.values,
        question_contract_step_id=model_turn_step_id(
            state.ports,
            purpose=ModelTurnPurpose.QUESTION_CONTRACT,
            turn=_question_turn_number(state),
        ),
        conversation_resolution_activation=_conversation_resolution_activation(state),
    )


def _verify_plan(state: _LookupPipelineState, plan: FactPlan) -> FactPlan:
    from fervis.lookup.plan_execution.verification import verify_fact_plan

    execution_sources = _authorized_execution_sources(state)
    return verify_fact_plan(
        plan,
        question_contract=state.question_contract,
        catalog=execution_sources.relation_catalog,
        catalog_selection=state.catalog_selection,
        available_values=_available_values_for_state(state),
        available_value_uses=state.grounding.ledger.uses,
        memory_relations=state.memory.relations,
        authorized_sources=execution_sources,
    )


def _authorized_execution_sources(
    state: _LookupPipelineState,
) -> AuthorizedExecutionSources:
    relation_sources = (
        _source_binding_relation_sources(state.source_binding_outcome)
        if isinstance(state.source_binding_outcome, SourceBindingPlan)
        else ()
    )
    if state.full_catalog is None:
        return AuthorizedExecutionSources.from_catalog_selection(
            state.catalog_selection
        )
    return AuthorizedExecutionSources.from_pipeline_sources(
        full_catalog=state.full_catalog,
        catalog_selection=state.catalog_selection,
        relation_sources=relation_sources,
    )


def _source_binding_relation_sources(
    source_binding: SourceBindingPlan,
) -> tuple[Any, ...]:
    output: list[Any] = []
    for bound_source in source_binding.bound_sources:
        if bound_source.source is not None:
            output.append(bound_source.source)
        output.extend(bound_source.source_invocations)
    return tuple(output)


def _available_values_for_state(state: _LookupPipelineState) -> tuple[FactValue, ...]:
    return _dedupe_fact_values(
        (
            *state.grounding.ledger.values,
            *active_memory_operation_values(
                memory=state.memory,
                active_memory_ids=_active_memory_ids(state),
            ),
        )
    )


def _source_binding_available_values_for_state(
    state: _LookupPipelineState,
) -> tuple[FactValue, ...]:
    grounded_values = (
        tuple(state.grounding.ledger.values) if state.grounding is not None else ()
    )
    return _dedupe_fact_values(grounded_values)


def _catalog_available_values_for_state(
    state: _LookupPipelineState,
) -> tuple[FactValue, ...]:
    grounded_values = (
        tuple(state.grounding.ledger.values) if state.grounding is not None else ()
    )
    return _dedupe_fact_values(
        (
            *grounded_values,
            *_active_memory_reference_values_for_state(state),
        )
    )


def _active_memory_reference_values_for_state(
    state: _LookupPipelineState,
) -> tuple[FactValue, ...]:
    return active_memory_reference_values(
        memory=state.memory,
        active_memory_ids=_active_memory_ids(state),
    )


def _dedupe_fact_values(values: tuple[FactValue, ...]) -> tuple[FactValue, ...]:
    output: list[FactValue] = []
    seen: set[str] = set()
    for value in values:
        if value.id in seen:
            continue
        seen.add(value.id)
        output.append(value)
    return tuple(output)


def _active_memory_ids(state: _LookupPipelineState) -> frozenset[str]:
    activated_memory = state.activated_memory
    if activated_memory is None:
        return frozenset()
    return frozenset(getattr(activated_memory, "by_memory_id", {}) or {})


def _conversation_resolution_activation(
    state: _LookupPipelineState,
) -> dict[str, Any]:
    resolution = state.conversation_resolution
    if resolution is None:
        return {}
    return dict(resolution.activation_payload())


def _activate_selected_memory(state: _LookupPipelineState) -> LookupResult | None:
    used_memory_ids = tuple(
        getattr(state.conversation_resolution, "used_memory_ids", ()) or ()
    )
    if not used_memory_ids:
        state.activated_memory = None
        return None
    try:
        state.activated_memory = expand_activated_memory_cards(
            artifacts=fact_artifacts_from_context(state.request.conversation_context),
            memory_cards=dict(state.memory_card_projection.private_cards or {}),
            used_memory_ids=used_memory_ids,
        )
    except ValueError:
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLANNING_FAILED,
            message="selected conversation memory could not be activated",
            usage=_conversation_usage(state) or {},
        )
    return None


def _conversation_resolution_ambiguity_fact_result(
    unresolved: Any,
) -> FactResult:
    clarifications = _conversation_resolution_clarifications(unresolved)
    return FactResult(outcome=NeedsClarification(clarifications=clarifications))


def _conversation_resolution_clarifications(
    unresolved: Any,
) -> tuple[Clarification, ...]:
    if getattr(unresolved, "unresolved_kind", "") == "multiple_meanings":
        option_labels = tuple(
            str(getattr(item, "integrated_question", "") or "")
            for item in getattr(unresolved, "candidate_interpretations", ()) or ()
            if str(getattr(item, "integrated_question", "") or "")
        )
        return (
            clarify(
                AmbiguousQuestionInterpretation(
                    clarification_id="conversation_resolution_ambiguous_1",
                    requested_fact_id="conversation_resolution",
                    source_text="",
                    options=tuple(
                        ClarificationOption(id=item, label=item)
                        for item in option_labels
                    ),
                    proof_refs=("conversation_resolution:unresolved",),
                )
            ),
        )
    return (
        clarify(
            AmbiguousQuestionInterpretation(
                clarification_id="conversation_resolution_ambiguous_1",
                requested_fact_id="conversation_resolution",
                source_text=_unresolved_text(unresolved),
                options=(
                    ClarificationOption(
                        id=_unresolved_text(unresolved),
                        label=_unresolved_text(unresolved),
                    ),
                ),
                proof_refs=("conversation_resolution:unresolved",),
            )
        ),
    )


def _unresolved_text(item: Any) -> str:
    return str(getattr(item, "why_unresolved", item) or "")


def _active_memory_prompt_context(state: _LookupPipelineState) -> dict[str, Any]:
    active_ids = _active_memory_prompt_ids(state)
    if not active_ids:
        return {}
    context = dict(state.memory.prompt_context or {})
    output: dict[str, Any] = {}
    relations = [
        relation
        for relation in context.get("memoryRelations") or ()
        if isinstance(relation, dict) and str(relation.get("id") or "") in active_ids
    ]
    if relations:
        output["memoryRelations"] = relations
    values = [
        value
        for value in context.get("memoryValues") or ()
        if isinstance(value, dict)
        and (
            str(value.get("id") or "") in active_ids
            or str(value.get("sourceRelationId") or "") in active_ids
        )
    ]
    if values:
        output["memoryValues"] = values
    outcomes = [
        outcome
        for outcome in context.get("memoryOutcomes") or ()
        if isinstance(outcome, dict) and str(outcome.get("id") or "") in active_ids
    ]
    if outcomes:
        output["memoryOutcomes"] = outcomes
    return output


def _active_memory_prompt_ids(state: _LookupPipelineState) -> frozenset[str]:
    active_ids = _active_memory_ids(state)
    if not active_ids:
        return frozenset()
    context = dict(state.memory.prompt_context or {})
    return active_ids | _active_source_relation_ids(
        context.get("memoryValues"),
        active_ids=active_ids,
    )


def _active_source_relation_ids(
    memory_values: Any,
    *,
    active_ids: frozenset[str],
) -> frozenset[str]:
    relation_ids: set[str] = set()
    for value in memory_values or ():
        if not isinstance(value, dict):
            continue
        if str(value.get("id") or "") not in active_ids:
            continue
        relation_id = str(value.get("sourceRelationId") or "").strip()
        if relation_id:
            relation_ids.add(relation_id)
    return frozenset(relation_ids)


def _question_turn_number(state: _LookupPipelineState) -> int:
    return 2 if state.conversation_turn is not None else 1


def _query_enrichment_turn_number(state: _LookupPipelineState) -> int:
    return _question_turn_number(state) + 1


def _grounding_turn_number(state: _LookupPipelineState) -> int:
    return _query_enrichment_turn_number(state) + 1


def _known_input_step_id(state: _LookupPipelineState) -> str | None:
    if state.grounding.turn is not None:
        return model_turn_step_id(
            state.ports,
            purpose=ModelTurnPurpose.GROUNDING,
            turn=_grounding_turn_number(state),
        )
    return model_turn_step_id(
        state.ports,
        purpose=ModelTurnPurpose.QUESTION_CONTRACT,
        turn=_question_turn_number(state),
    )


def _source_binding_base_turn_number(state: _LookupPipelineState) -> int:
    return _read_eligibility_turn_number(state)


def _read_eligibility_turn_number(state: _LookupPipelineState) -> int:
    return (
        _grounding_turn_number(state) + 1
        if state.grounding.turn is not None
        else _grounding_turn_number(state)
    )


def _conversation_usage(state: _LookupPipelineState) -> dict[str, Any] | None:
    if state.conversation_turn is None:
        return None
    return state.conversation_turn.usage


def _append_model_turn_completed(
    state: _LookupPipelineState,
    *,
    phase: ModelTurnPurpose,
    turn: int,
    model_turn: Any,
) -> RunStepWrite | None:
    payload = _model_turn_event_payload(
        request=state.request,
        phase=phase,
        turn=turn,
        provider=state.provider,
        model_key=state.model_key,
        system_prompt=model_turn.artifact.system_prompt,
        prompt_text=model_turn.artifact.prompt_text,
        usage=model_turn.usage,
        duration_ms=model_turn.duration_ms,
        tool_specs=model_turn.artifact.tool_specs,
        schema=model_turn.artifact.provider_schema,
        submitted_payload=model_turn.artifact.submitted_payload,
        raw_output=model_turn.artifact.raw_output,
        parsed_payload=model_turn.artifact.parsed_payload,
        derived_payload=model_turn.artifact.derived_payload,
        selected_tool_name=model_turn.artifact.selected_tool_name,
    )
    step = record_model_turn_step(
        state.ports,
        purpose=phase,
        turn=turn,
        prompt_chars=payload.get(EventPayloadKey.PROMPT_CHARS),
        schema_chars=payload.get(EventPayloadKey.SCHEMA_CHARS),
        output_summary_json=_model_turn_output_summary(
            state,
            phase=phase,
            payload=payload,
        ),
    )
    record_model_turn_audit(
        state.ports,
        step=step,
        provider=state.provider,
        model_key=state.model_key,
        artifact=model_turn.artifact,
        usage=model_turn.usage,
        duration_ms=model_turn.duration_ms,
        succeeded=True,
    )
    return step


def _model_turn_output_summary(
    state: _LookupPipelineState,
    *,
    phase: ModelTurnPurpose,
    payload: dict[str, Any],
) -> dict[str, object]:
    summary = lineage_model_turn_output_summary(payload)
    if (
        phase == ModelTurnPurpose.GROUNDING
        and state.grounding is not None
        and state.question_contract is not None
    ):
        return add_grounding_result_semantics(
            summary,
            ledger=state.grounding.ledger,
            question_contract=state.question_contract,
        )
    return summary


def _model_turn_failure_result(
    state: _LookupPipelineState,
    *,
    phase: ModelTurnPurpose,
    turn: int,
    exc: Any,
    usage: dict[str, Any],
) -> LookupResult:
    payload = _model_turn_event_payload(
        request=state.request,
        phase=phase,
        turn=turn,
        provider=state.provider,
        model_key=state.model_key,
        system_prompt=exc.artifact.system_prompt,
        prompt_text=exc.artifact.prompt_text,
        usage=exc.usage,
        duration_ms=exc.duration_ms,
        tool_specs=exc.artifact.tool_specs,
        schema=exc.artifact.provider_schema,
        submitted_payload=exc.artifact.submitted_payload,
        raw_output=exc.artifact.raw_output,
        parsed_payload=exc.artifact.parsed_payload,
        derived_payload=exc.artifact.derived_payload,
        selected_tool_name=exc.artifact.selected_tool_name,
        error_code=exc.error_code,
        error_class=(
            exc.__cause__.__class__.__name__
            if exc.__cause__ is not None
            else exc.__class__.__name__
        ),
        error_context=exc.error_context,
    )
    failed_step = record_model_turn_step(
        state.ports,
        purpose=phase,
        turn=turn,
        prompt_chars=payload.get(EventPayloadKey.PROMPT_CHARS),
        schema_chars=payload.get(EventPayloadKey.SCHEMA_CHARS),
        error_json=lineage_error_json(payload),
    )
    try:
        record_model_turn_audit(
            state.ports,
            step=failed_step,
            provider=state.provider,
            model_key=state.model_key,
            artifact=exc.artifact,
            usage=exc.usage,
            duration_ms=exc.duration_ms,
            succeeded=False,
        )
    except LineagePersistenceUnavailable:
        pass
    try:
        record_runtime_error_lineage(
            request=state.request,
            ports=state.ports,
            failed_step_id=failed_step.step_id if failed_step is not None else None,
            error_code=exc.error_code,
            message=_model_turn_error_message(exc.error_code, exc.error_context),
        )
    except LineagePersistenceUnavailable:
        return LookupResult(
            status=RunStatus.FAILED,
            error=exc.error_code,
            usage=usage,
        )
    return LookupResult(
        status=RunStatus.FAILED,
        error=exc.error_code,
        usage=usage,
    )


def _model_turn_error_message(
    error_code: str,
    error_context: dict[str, Any],
) -> str:
    if not error_context:
        return error_code
    return f"{error_code}: {json.dumps(error_context, sort_keys=True)}"


def _record_execution_lineage_step(
    state: _LookupPipelineState,
    execution: Any,
    execution_lineage: _SourceReadLineage,
) -> None:
    execution_payload = {
        EventPayloadKey.RUN_ID: state.request.run_id,
        EventPayloadKey.RELATION_COUNT: len(execution.relations),
    }
    if execution.issue is not None:
        execution_payload.update(
            {
                EventPayloadKey.ERROR_CODE: execution.issue.kind.value,
                EventPayloadKey.ERROR_CLASS: execution.issue.__class__.__name__,
                EventPayloadKey.ERROR_CONTEXT: execution.issue.message,
            }
        )
    record_compile_step(
        state.ports,
        proof_node_count=len(execution.proof_graph.nodes),
        proof_edge_count=len(execution.proof_graph.edges),
    )
    record_execution_step(
        state.ports,
        relation_count=len(execution.relations),
        proof_refs=tuple(str(item) for item in execution.proof_refs),
        error_json=lineage_error_json(execution_payload),
        catalog_endpoints=execution_lineage.catalog_endpoints,
        source_reads=execution_lineage.source_reads,
    )


def _record_execution_failure_step(
    state: _LookupPipelineState,
    payload: dict[str, Any],
    execution_lineage: _SourceReadLineage,
) -> RunStepWrite | None:
    return record_execution_step(
        state.ports,
        error_json=lineage_error_json(payload),
        catalog_endpoints=execution_lineage.catalog_endpoints,
        source_reads=execution_lineage.source_reads,
    )


def _execution_lineage(state: _LookupPipelineState) -> _SourceReadLineage:
    return _source_read_lineage_for_step(state, step_id=execution_step_id(state.ports))


def _grounding_source_lineage(state: _LookupPipelineState) -> _SourceReadLineage:
    return _source_read_lineage_for_step(
        state,
        step_id=model_turn_step_id(
            state.ports,
            purpose=ModelTurnPurpose.GROUNDING,
            turn=_grounding_turn_number(state),
        ),
    )


def _source_read_lineage_for_step(
    state: _LookupPipelineState,
    *,
    step_id: str | None,
) -> _SourceReadLineage:
    if step_id is None:
        return _SourceReadLineage()
    source_read_buffer = _SourceReadLineageBuffer()
    return _SourceReadLineage(
        source_read_buffer=source_read_buffer,
        source_read_lineage=SourceReadLineageScope(
            run_id=state.request.run_id,
            step_id=step_id,
            recorder=source_read_buffer,
        ),
    )


def _source_binding_usage(state: _LookupPipelineState) -> dict[str, Any]:
    if state.source_binding_turn is None:
        return _plan_selection_usage(state)
    return _merge_usage(
        _plan_selection_usage(state),
        state.source_binding_turn.usage,
    )


def _dedupe_catalog_endpoints(
    catalog_endpoints: list[CatalogEndpointWrite],
) -> tuple[CatalogEndpointWrite, ...]:
    by_run_key: dict[tuple[str, str], CatalogEndpointWrite] = {}
    for catalog_endpoint in catalog_endpoints:
        key = (catalog_endpoint.run_id, catalog_endpoint.catalog_endpoint_key)
        existing = by_run_key.get(key)
        if existing is None:
            by_run_key[key] = catalog_endpoint
            continue
        if existing != catalog_endpoint:
            raise ValueError(
                "conflicting catalog endpoint lineage for "
                f"{catalog_endpoint.catalog_endpoint_key!r} in run "
                f"{catalog_endpoint.run_id!r}"
            )
    return tuple(by_run_key.values())


def _source_binding_model_turn_count(state: _LookupPipelineState) -> int:
    if state.source_binding_turn is None:
        return 0
    subturns = getattr(state.source_binding_turn, "subturns", ()) or ()
    return len(subturns) or 1


def _phase_usage(
    state: _LookupPipelineState,
    *items: dict[str, Any] | None,
) -> dict[str, Any]:
    return _merge_usage(
        _conversation_usage(state),
        getattr(getattr(state, "question_turn", None), "usage", None),
        getattr(state, "query_enrichment_usage", None),
        getattr(state, "grounding_usage", None),
        getattr(state, "read_eligibility_usage", None),
        *items,
    )


def _plan_selection_usage(state: _LookupPipelineState) -> dict[str, Any]:
    return _phase_usage(
        state,
        getattr(getattr(state, "plan_selection_turn", None), "usage", None),
    )


def _pattern_plan_usage(state: _LookupPipelineState) -> dict[str, Any]:
    return _merge_usage(
        _source_binding_usage(state),
        state.plan_turn.usage,
    )
