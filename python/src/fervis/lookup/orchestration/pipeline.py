"""Lookup runtime pipeline."""

import json
from dataclasses import dataclass
from typing import Any

from fervis.model_io.turns import ModelTurnPurpose
from fervis.lineage.enums import ArtifactKind, ProgramInvocationKind
from fervis.lookup.errors import ErrorCode
from fervis.observability.event_contracts import EventPayloadKey
from fervis.lookup.relation_catalog import parse_relation_catalog
from fervis.lookup.conversation_resolution import (
    CompiledConversationResolution,
    ConversationResolution,
    ConversationResolutionGenerationError,
    ConversationResolutionRequest,
    ConversationResolutionTurnResult,
    compile_conversation_resolution,
    conversation_resolution_context_sources,
    generate_conversation_resolution,
)
from fervis.lookup.conversation_resolution.callable_frames import (
    callable_frame_bindings,
)
from fervis.lookup.conversation_resolution.model import UnresolvedResolution
from fervis.lookup.plan_execution.authorized_sources import (
    AuthorizedExecutionSources,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.persistence import (
    ProgramInvocationBinding,
)
from fervis.lookup.memory.projection import project_lookup_memory
from fervis.lookup.memory.outcomes import fact_value_memory_addresses
from fervis.lookup.outcomes.model import (
    FactResult,
    NeedsClarification,
)
from fervis.lookup.clarification import (
    AmbiguousQuestionInterpretation,
    Clarification,
    clarify,
)
from fervis.lookup.clarification.model import (
    ConversationResolutionResponse,
    ConversationInterpretationCandidate,
    ConversationInterpretationEvidence,
    QuestionContractResponse,
)
from fervis.lookup.answer_program.values import (
    FactValue,
)
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.question_contract.request import QuestionContractRequest
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupRuntimePorts,
)
from fervis.lookup.orchestration.result import LookupResult, RunStatus
from fervis.lookup.memory.projection import (
    ConversationMemoryProjectionOverflow,
    project_conversation_memory_cards,
)
from fervis.memory.conversation_context import (
    ConversationMemoryCardProjection,
    ExpandedActivatedMemory,
    expand_activated_memory_cards,
)
from fervis.memory.projection import fact_artifacts_from_context
from fervis.lineage.recorder import (
    RunArtifactWrite,
    RunStepWrite,
)
from .model_turn_events import _model_turn_event_payload
from fervis.lookup.lineage.steps import (
    lineage_error_json,
    lineage_model_turn_output_summary,
    model_turn_step_id,
    record_model_turn_audit,
    record_model_turn_step,
)
from fervis.lookup.lineage.results import (
    LineagePersistenceUnavailable,
    RuntimeErrorTerminal,
    record_runtime_error_lineage,
    runtime_error_terminal_result,
)
from .result_synthesis import _synthesize_result
from .program_execution import ProgramExecutionPorts, run_answer_program_execution
from .terminal_results import (
    semantic_clarification_fact_result,
)
from .limits import _limit_before_next_model_turn, _merge_usage
from .question_execution import (
    CompileQuestionExecution,
    ContinuePriorRequestExecution,
    fold_question_execution,
    parse_question_execution,
)
from .semantic_compilation import (
    SemanticCompilationClarification,
    SemanticCompilationRequest,
    SemanticCompilationSuccess,
    SemanticCompilationTurnError,
    compile_semantic_question,
    resolve_semantic_continuation_arguments,
)
from fervis.lookup.grounding import IdentityExecutionClarification


@dataclass
class _LookupPipelineState:
    request: LookupRequest
    ports: LookupRuntimePorts
    memory: Any
    provider: str
    model_key: str
    memory_card_projection: ConversationMemoryCardProjection
    activated_memory: ExpandedActivatedMemory | None = None
    conversation_turn: ConversationResolutionTurnResult | None = None
    conversation_resolution: ConversationResolution | None = None
    compiled_conversation_resolution: CompiledConversationResolution | None = None
    full_catalog: Any = None
    semantic_compilation: SemanticCompilationSuccess | None = None
    semantic_usage: dict[str, Any] | None = None
    semantic_turn_numbers: dict[ModelTurnPurpose, list[int]] | None = None


class _RunLimitReached(Exception):
    def __init__(self, result: LookupResult):
        super().__init__("run limit reached")
        self.result = result


@dataclass
class _SemanticTurnRecorder:
    state: _LookupPipelineState
    next_turn: int
    usage: dict[str, Any]
    turn_numbers: dict[ModelTurnPurpose, list[int]]

    def __call__(self, purpose: ModelTurnPurpose, model_turn: Any) -> None:
        _append_model_turn_completed(
            self.state,
            phase=purpose,
            turn=self.next_turn,
            model_turn=model_turn,
        )
        self.turn_numbers.setdefault(purpose, []).append(self.next_turn)
        self.usage = _merge_usage(self.usage, model_turn.usage)
        self.next_turn += 1
        failure = _limit_before_next_model_turn(
            self.state.ports,
            self.state.request.run_id,
        )
        if failure is not None:
            raise _RunLimitReached(failure)


def run_lookup_question(
    request: LookupRequest,
    ports: LookupRuntimePorts,
) -> LookupResult:
    try:
        memory_card_projection = project_conversation_memory_cards(
            request.conversation_context,
            prior_program_invocations=ports.prior_program_invocations,
            conversation_id=str(
                request.user_context.get("conversationId") or ""
            ).strip(),
            tenant_id=request.tenant_id,
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
        conversation_result = _run_conversation_resolution_phase(state)
        if conversation_result is not None:
            return conversation_result
        try:
            execution = parse_question_execution(
                resolution=state.compiled_conversation_resolution,
                memory_projection=state.memory_card_projection,
                prior_program_invocations=state.ports.prior_program_invocations,
                conversation_id=str(
                    state.request.user_context.get("conversationId") or ""
                ).strip(),
                tenant_id=state.request.tenant_id,
            )
        except ValueError:
            return _runtime_error_terminal(
                state,
                error_code=ErrorCode.PLAN_VALIDATION_FAILED,
                message="question execution did not match its persisted contract",
                usage=_phase_usage(state),
            )
        return fold_question_execution(
            execution,
            compile_question=lambda selected: _run_compile_question_execution(
                state,
                selected,
            ),
            continue_prior_request=lambda selected: (
                _run_continue_prior_request_execution(state, selected)
            ),
        )
    except LineagePersistenceUnavailable:
        return RuntimeErrorTerminal(
            run_id=request.run_id,
            error_code=ErrorCode.LINEAGE_PERSISTENCE_FAILED,
            message=ErrorCode.LINEAGE_PERSISTENCE_FAILED,
        ).lookup_result()


def _run_compile_question_execution(
    state: _LookupPipelineState,
    execution: CompileQuestionExecution,
) -> LookupResult:
    state.compiled_conversation_resolution = execution.resolution
    return _run_semantic_compile_question(state)


def _run_semantic_compile_question(state: _LookupPipelineState) -> LookupResult:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    state.full_catalog = parse_relation_catalog(
        state.ports.relation_catalog_port.build_relation_catalog()
    )
    recorder = _SemanticTurnRecorder(
        state=state,
        next_turn=2 if state.conversation_turn is not None else 1,
        usage={},
        turn_numbers={},
    )

    try:
        outcome = compile_semantic_question(
            _semantic_compilation_request(state),
            on_turn=recorder,
        )
    except _RunLimitReached as exc:
        return exc.result
    except SemanticCompilationTurnError as exc:
        return _model_turn_failure_result(
            state,
            phase=exc.purpose,
            turn=recorder.next_turn,
            exc=exc.failure,
            usage=_merge_usage(recorder.usage, exc.failure.usage),
        )
    except (ValueError, VerificationError) as exc:
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message=str(exc),
            usage=recorder.usage,
        )
    state.semantic_usage = recorder.usage
    state.semantic_turn_numbers = recorder.turn_numbers
    if isinstance(outcome, SemanticCompilationClarification):
        return _semantic_compilation_clarification_result(state, outcome)
    state.semantic_compilation = outcome
    _record_source_contract_snapshot(state, outcome)
    return _run_semantic_execution_phase(state)


def _record_source_contract_snapshot(
    state: _LookupPipelineState,
    outcome: SemanticCompilationSuccess,
) -> None:
    sink = state.ports.lineage_step_sink
    if sink is None:
        return
    read_eligibility_turns = (state.semantic_turn_numbers or {}).get(
        ModelTurnPurpose.READ_ELIGIBILITY,
        [],
    )
    if not read_eligibility_turns:
        raise LineagePersistenceUnavailable(
            "source contract snapshot requires Read Eligibility lineage"
        )
    step_id = model_turn_step_id(
        state.ports,
        purpose=ModelTurnPurpose.READ_ELIGIBILITY,
        turn=read_eligibility_turns[-1],
    )
    if step_id is None:
        raise LineagePersistenceUnavailable(
            "source contract snapshot requires a lineage step"
        )
    snapshot = outcome.source_contract_snapshot
    content = snapshot.content
    sink.recorder.record_artifact(
        RunArtifactWrite(
            artifact_id=snapshot.ref,
            run_id=state.request.run_id,
            step_id=step_id,
            artifact_kind=ArtifactKind.SOURCE_CONTRACT,
            content_hash=snapshot.ref.rsplit(":", 1)[-1],
            content_type="application/json",
            size_bytes=len(content.encode("utf-8")),
            content=content,
        )
    )


def _semantic_compilation_request(
    state: _LookupPipelineState,
) -> SemanticCompilationRequest:
    return SemanticCompilationRequest(
        run_id=state.request.run_id,
        question=state.request.question,
        question_contract_request=QuestionContractRequest(
            current_question=state.request.question,
            conversation_context=state.request.conversation_context,
            conversation_resolution=(
                state.compiled_conversation_resolution
                if state.compiled_conversation_resolution is not None
                and state.compiled_conversation_resolution.uses_prior_context
                else None
            ),
            host=state.request.host,
            clarification_responses=tuple(
                response
                for response in state.request.clarification_responses
                if isinstance(response, QuestionContractResponse)
            ),
        ),
        full_catalog=state.full_catalog,
        memory_relations=state.memory.relations,
        data_access_port=state.ports.data_access_port,
        model_port=state.ports.planner_model_port,
        provider=state.provider,
        max_thinking_tokens=state.request.max_thinking_tokens,
        max_catalog_reads_per_fact=state.request.max_catalog_reads_per_fact,
        runtime_values=state.request.runtime_values,
        conversation_context=state.request.conversation_context,
        host=state.request.host,
        clarification_responses=state.request.clarification_responses,
    )


def _run_semantic_execution_phase(state: _LookupPipelineState) -> LookupResult:
    compiled = state.semantic_compilation
    if compiled is None:
        raise VerificationError("semantic execution requires compiled facts")
    execution_sources = AuthorizedExecutionSources.from_program(
        full_catalog=state.full_catalog,
        program=compiled.compilation.answer_program,
    )
    question_turns = (state.semantic_turn_numbers or {}).get(
        ModelTurnPurpose.QUESTION_CONTRACT,
        [],
    )
    known_input_turns = (state.semantic_turn_numbers or {}).get(
        ModelTurnPurpose.READ_ELIGIBILITY, []
    ) or (state.semantic_turn_numbers or {}).get(ModelTurnPurpose.GROUNDING, [])
    grounded_values = tuple(value.typed_value for value in compiled.canonical_values)
    return run_answer_program_execution(
        request=state.request,
        ports=ProgramExecutionPorts(
            data_access_port=state.ports.data_access_port,
            memory=state.memory,
            lineage_step_sink=state.ports.lineage_step_sink,
            lineage_required=state.ports.lineage_required,
        ),
        program=compiled.compilation.answer_program,
        bindings=compiled.compilation.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=execution_sources.relation_catalog,
            authorized_sources=execution_sources,
            catalog_selection=compiled.catalog_selection,
            memory_relations=state.memory.relations,
            authority_ref=state.request.authority_ref,
            expression_values=_answer_program_expression_values(state),
            expression_types={"ANCHOR_TIMEZONE": "string"},
        ),
        invocation_binding=_program_invocation_binding(state),
        question_contract_step_id=(
            model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=question_turns[0],
            )
            if question_turns
            else ""
        )
        or "",
        usage=state.semantic_usage or {},
        grounded_values=grounded_values,
        extra_fact_addresses=fact_value_memory_addresses(grounded_values),
        known_input_step_id=(
            model_turn_step_id(
                state.ports,
                purpose=(
                    ModelTurnPurpose.READ_ELIGIBILITY
                    if (state.semantic_turn_numbers or {}).get(
                        ModelTurnPurpose.READ_ELIGIBILITY
                    )
                    else ModelTurnPurpose.GROUNDING
                ),
                turn=known_input_turns[-1],
            )
            if known_input_turns
            else None
        ),
        conversation_resolution_activation=_conversation_resolution_activation(state),
    )


def _semantic_compilation_clarification_result(
    state: _LookupPipelineState,
    outcome: SemanticCompilationClarification,
) -> LookupResult:
    question_turns = (state.semantic_turn_numbers or {}).get(
        ModelTurnPurpose.QUESTION_CONTRACT,
        [],
    )
    return _synthesize_result(
        request=state.request,
        ports=state.ports,
        fact_result=semantic_clarification_fact_result(
            outcome.cause,
            contract=outcome.question_contract,
        ),
        status=RunStatus.NEEDS_CLARIFICATION,
        usage=state.semantic_usage or {},
        question_contract=outcome.question_contract,
        grounded_values=tuple(
            value.typed_value for value in outcome.canonical_values
        ),
        question_contract_step_id=(
            model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.QUESTION_CONTRACT,
                turn=question_turns[-1],
            )
            if question_turns
            else ""
        )
        or "",
    )


def _run_continue_prior_request_execution(
    state: _LookupPipelineState,
    execution: ContinuePriorRequestExecution,
) -> LookupResult:
    limit_failure = _limit_before_next_model_turn(state.ports, state.request.run_id)
    if limit_failure is not None:
        return limit_failure
    state.full_catalog = parse_relation_catalog(
        state.ports.relation_catalog_port.build_relation_catalog()
    )
    recorder = _SemanticTurnRecorder(
        state=state,
        next_turn=2,
        usage={},
        turn_numbers={},
    )

    try:
        grounded_values = resolve_semantic_continuation_arguments(
            execution.frame,
            _semantic_compilation_request(state),
            on_turn=recorder,
        )
    except _RunLimitReached as exc:
        return exc.result
    except SemanticCompilationTurnError as exc:
        return _model_turn_failure_result(
            state,
            phase=exc.purpose,
            turn=recorder.next_turn,
            exc=exc.failure,
            usage=_merge_usage(recorder.usage, exc.failure.usage),
        )
    except (ValueError, VerificationError) as exc:
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message=str(exc),
            usage=recorder.usage,
        )
    state.semantic_usage = recorder.usage
    state.semantic_turn_numbers = recorder.turn_numbers
    if isinstance(grounded_values, IdentityExecutionClarification):
        program = execution.frame.program
        question_contract = QuestionContract(
            inputs=program.inputs,
            requested_facts=program.fact_template,
            input_denotations=program.input_denotations,
        )
        return _synthesize_result(
            request=state.request,
            ports=state.ports,
            fact_result=semantic_clarification_fact_result(
                grounded_values,
                contract=question_contract,
            ),
            status=RunStatus.NEEDS_CLARIFICATION,
            usage=recorder.usage,
            question_contract=question_contract,
            grounded_values=tuple(execution.frame.certified_argument_values),
            question_contract_step_id="",
        )
    return _run_continue_prior_request_program(
        state,
        execution,
        grounded_values=grounded_values,
    )


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
    context_sources = state.memory_card_projection.context_sources
    context_frames = state.memory_card_projection.context_frames
    conversation_responses = tuple(
        response
        for response in state.request.clarification_responses
        if isinstance(response, ConversationResolutionResponse)
    )
    if not context_sources and not conversation_responses:
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
        resolution_request = ConversationResolutionRequest(
            question=state.request.question,
            conversation_context=state.request.conversation_context,
            host=state.request.host,
            context_sources=context_sources,
            context_frames=context_frames,
            clarification_responses=conversation_responses,
        )
        state.conversation_turn = generate_conversation_resolution(
            request=resolution_request,
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
    if state.conversation_resolution.needs_clarification:
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
    try:
        state.compiled_conversation_resolution = compile_conversation_resolution(
            state.conversation_resolution,
            memory_projection=state.memory_card_projection,
            context_sources=conversation_resolution_context_sources(resolution_request),
        )
    except ValueError:
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLANNING_FAILED,
            message="conversation resolution could not be compiled",
            usage=state.conversation_turn.usage,
        )
    activation_error = _activate_selected_memory(state)
    if activation_error is not None:
        return activation_error
    return None


def _run_continue_prior_request_program(
    state: _LookupPipelineState,
    execution: ContinuePriorRequestExecution,
    *,
    grounded_values: tuple[FactValue, ...],
) -> LookupResult:
    prepared = execution.frame
    try:
        bindings = callable_frame_bindings(
            prepared,
            grounded_values=grounded_values,
        )
    except ValueError:
        return _runtime_error_terminal(
            state,
            error_code=ErrorCode.PLAN_VALIDATION_FAILED,
            message="callable prior frame arguments could not be bound",
            usage=_phase_usage(state),
        )
    execution_sources = AuthorizedExecutionSources.from_program(
        full_catalog=state.full_catalog,
        program=prepared.program,
    )
    return run_answer_program_execution(
        request=state.request,
        ports=ProgramExecutionPorts(
            data_access_port=state.ports.data_access_port,
            memory=state.memory,
            lineage_step_sink=state.ports.lineage_step_sink,
            lineage_required=state.ports.lineage_required,
        ),
        program=prepared.program,
        bindings=bindings,
        environment=ExecutionEnvironment(
            catalog=execution_sources.relation_catalog,
            authorized_sources=execution_sources,
            memory_relations=state.memory.relations,
            authority_ref=state.request.authority_ref,
            expression_values=_answer_program_expression_values(state),
            expression_types={"ANCHOR_TIMEZONE": "string"},
        ),
        invocation_binding=_program_invocation_binding(state),
        question_contract_step_id=(
            model_turn_step_id(
                state.ports,
                purpose=ModelTurnPurpose.CONVERSATION_RESOLUTION,
                turn=1,
            )
            or ""
        ),
        usage=_phase_usage(state),
        grounded_values=grounded_values,
        extra_fact_addresses=fact_value_memory_addresses(grounded_values),
        known_input_step_id=_continue_prior_request_known_input_step_id(state),
        conversation_resolution_activation=_conversation_resolution_activation(state),
        invocation_kind=ProgramInvocationKind.CONTINUE_PRIOR_REQUEST,
        base_invocation_id=prepared.base.invocation.invocation_id,
    )


def _answer_program_expression_values(
    state: _LookupPipelineState,
) -> dict[str, str]:
    runtime_values = state.request.runtime_values
    return {
        "ANCHOR_TIMEZONE": runtime_values.timezone if runtime_values is not None else ""
    }


def _conversation_resolution_activation(
    state: _LookupPipelineState,
) -> dict[str, Any]:
    resolution = state.conversation_resolution
    payload = dict(resolution.activation_payload()) if resolution is not None else {}
    compiled = state.compiled_conversation_resolution
    if compiled is not None:
        payload["conversation_resolution_context"] = compiled.to_prompt_payload()
    return payload


def _activate_selected_memory(state: _LookupPipelineState) -> LookupResult | None:
    resolution = state.conversation_resolution
    if resolution is None:
        return None
    used_memory_ids = resolution.used_memory_ids
    if not used_memory_ids:
        state.activated_memory = None
        return None
    try:
        state.activated_memory = expand_activated_memory_cards(
            artifacts=fact_artifacts_from_context(state.request.conversation_context),
            memory_projection=state.memory_card_projection,
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
    unresolved: UnresolvedResolution,
) -> FactResult:
    clarifications = _conversation_resolution_clarifications(unresolved)
    return FactResult(outcome=NeedsClarification(clarifications=clarifications))


def _conversation_resolution_clarifications(
    unresolved: UnresolvedResolution,
) -> tuple[Clarification, ...]:
    if unresolved.unresolved_kind == "multiple_meanings":
        return (
            clarify(
                AmbiguousQuestionInterpretation(
                    clarification_id="conversation_resolution_ambiguous_1",
                    requested_fact_id="conversation_resolution",
                    source_text="",
                    candidates=tuple(
                        ConversationInterpretationCandidate(
                            id=f"interpretation_{index}",
                            contextualized_question=item.contextualized_question,
                            source_evidence=tuple(
                                ConversationInterpretationEvidence(
                                    source_id=evidence.source_id,
                                    exact_source_texts=evidence.exact_source_texts,
                                )
                                for evidence in item.context_evidence
                            ),
                        )
                        for index, item in enumerate(
                            unresolved.candidate_interpretations, start=1
                        )
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
                accepts_free_text=True,
                proof_refs=("conversation_resolution:unresolved",),
            )
        ),
    )


def _unresolved_text(item: UnresolvedResolution) -> str:
    return item.why_unresolved


def _continue_prior_request_known_input_step_id(
    state: _LookupPipelineState,
) -> str | None:
    semantic_turns = state.semantic_turn_numbers or {}
    for purpose in (ModelTurnPurpose.READ_ELIGIBILITY, ModelTurnPurpose.GROUNDING):
        turns = semantic_turns.get(purpose, [])
        if turns:
            return model_turn_step_id(
                state.ports,
                purpose=purpose,
                turn=turns[-1],
            )
    return model_turn_step_id(
        state.ports,
        purpose=ModelTurnPurpose.CONVERSATION_RESOLUTION,
        turn=1,
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
        output_summary_json=lineage_model_turn_output_summary(payload),
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


def _program_invocation_binding(
    state: _LookupPipelineState,
) -> ProgramInvocationBinding | None:
    binding = state.ports.program_invocation_binding
    if binding is None and state.ports.lineage_required:
        raise LineagePersistenceUnavailable(
            "answer program invocation persistence is unavailable"
        )
    return binding


def _phase_usage(
    state: _LookupPipelineState,
    *items: dict[str, Any] | None,
) -> dict[str, Any]:
    return _merge_usage(
        _conversation_usage(state),
        getattr(state, "semantic_usage", None),
        *items,
    )
