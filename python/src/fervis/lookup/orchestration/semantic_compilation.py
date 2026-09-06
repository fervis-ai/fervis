"""Sole semantic compilation sequence from question text to AnswerProgram."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.available_sources import (
    SourceContractSnapshot,
    build_available_source_catalog,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.answer_program.values import LiteralType
from fervis.lookup.clarification.context import clarification_response_ref
from fervis.lookup.clarification.model import (
    ClarificationOwnerResponse,
    GroundingIdentityResponse,
    SourceBindingCatalogInputResponse,
)
from fervis.lookup.conversation_resolution.callable_frames import CallableFrameProgram
from fervis.lookup.fact_compilation import (
    FactCompilationResult,
    compile_verified_source_strategies,
)
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceCatalog,
    build_api_row_source_catalog,
    build_row_source_catalog,
)
from fervis.lookup.runtime_values import RuntimeValueContext
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.grounding import (
    CanonicalInputValue,
    IdentityExecutionClarification,
    IdentityExecutionFailureReason,
    SemanticGroundingRequest,
    SemanticGroundingResult,
    deterministic_scalar_values,
    grounding_partitions,
    reference_grounding_tasks,
    time_grounding_tasks,
    build_canonical_input_ledger,
    parse_semantic_grounding,
    reference_binding_options,
    SemanticGroundingTurnPrompt,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.plan_execution.authorized_sources import AuthorizedExecutionSources
from fervis.lookup.source_binding.candidates import candidate_source_strategy
from fervis.lookup.query_enrichment import (
    SemanticQueryEnrichmentRequest,
    reference_input_recall_tasks,
    semantic_recall_buckets,
    parse_semantic_query_enrichment,
    SemanticQueryEnrichmentTurnPrompt,
)
from fervis.lookup.question_contract import (
    InputTerm,
    QuestionContract,
    QuestionContractNeedsClarification,
    QuestionContractRequest,
    RequestedFactSemanticIndex,
    analyze_requested_fact,
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
    SemanticQuestionContractTurnPrompt,
    SemanticQuestionFrameTurnPrompt,
)
from fervis.lookup.read_eligibility import (
    IdentityRouteSelection,
    SemanticReadEligibilityRequest,
    SemanticReadEligibilityResult,
    combine_semantic_read_eligibility_results,
    parse_semantic_read_eligibility,
    SemanticReadEligibilityTurnPrompt,
    execute_identity_selection,
)
from fervis.lookup.relation_catalog.selection.batches import (
    combine_catalog_selection_batches,
    next_catalog_selection_batch,
)
from fervis.lookup.relation_catalog import RelationCatalog, RelationDataAccessPort
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionResult,
    SemanticCatalogSelectionRequest,
    select_resolver_reads,
    select_semantic_relation_catalog,
)
from fervis.lookup.semantic_turn import SemanticTurnResult, generate_semantic_turn
from fervis.lookup.semantic_turn import SemanticTurnGenerationError
from fervis.lookup.source_binding.membership import SourceMembershipTurnPrompt, membership_surfaces, parse_source_membership
from fervis.lookup.source_binding.parser import empty_input_binding_payload
from fervis.lookup.source_binding import (
    CatalogProvidedValue,
    SemanticSourceBindingRequest,
    compile_source_binding_plan,
    SemanticSourceBindingTurnPrompt,
    SemanticSourceRealizationTurnPrompt,
    compile_source_realization,
    SourceRealizationUnavailable,
    SourceStrategyVerificationFailure,
    VerifiedSourceStrategy,
    verify_source_strategy,
    source_binding_clarification,
    source_required_inputs_are_satisfiable,
)
from fervis.lookup.turn_prompts import (
    HostPromptContext,
    TurnPromptContext,
    build_turn_prompt_context,
)
from fervis.model_io.turns import ModelTurnPurpose


SemanticTurnObserver = Callable[
    [ModelTurnPurpose, SemanticTurnResult[Any]],
    None,
]


@dataclass(frozen=True)
class SemanticCompilationRequest:
    run_id: str
    question: str
    question_contract_request: QuestionContractRequest
    full_catalog: RelationCatalog
    memory_relations: tuple[RelationRows, ...]
    data_access_port: RelationDataAccessPort
    model_port: Any
    provider: str
    max_thinking_tokens: int
    max_catalog_reads_per_fact: int
    runtime_values: RuntimeValueContext | None
    conversation_context: dict[str, Any]
    host: HostPromptContext
    clarification_responses: tuple[ClarificationOwnerResponse, ...] = ()


@dataclass(frozen=True)
class SemanticCompilationSuccess:
    question_contract: QuestionContract
    canonical_values: tuple[CanonicalInputValue, ...]
    catalog_selection: CatalogSelectionResult
    source_contract_snapshot: SourceContractSnapshot
    compilation: FactCompilationResult


@dataclass(frozen=True)
class SemanticCompilationClarification:
    cause: object
    question_contract: QuestionContract | None = None
    canonical_values: tuple[CanonicalInputValue, ...] = ()


@dataclass(frozen=True)
class SemanticCompilationImpossible:
    question_contract: QuestionContract
    canonical_values: tuple[CanonicalInputValue, ...]
    blocked_fact_ids: tuple[str, ...]
    source_contract_snapshot: SourceContractSnapshot
    reviewed_read_ids: tuple[str, ...]
    failed_requirement_refs: tuple[str, ...] = ()


SemanticCompilationOutcome = (
    SemanticCompilationSuccess | SemanticCompilationClarification | SemanticCompilationImpossible
)


def resolve_semantic_continuation_arguments(
    frame: CallableFrameProgram,
    request: SemanticCompilationRequest,
    *,
    on_turn: SemanticTurnObserver | None = None,
) -> tuple[FactValue, ...] | IdentityExecutionClarification:
    """Resolve only changed bindings for one verified persisted program."""

    context = build_turn_prompt_context(
        current_question=request.question,
        conversation_context=request.conversation_context,
        host=request.host,
    )
    input_by_ref = {item.id: item for item in frame.program.inputs}
    input_by_ref.update({item.id: item for item in frame.changed_inputs})
    denotation_by_ref = {
        item.input_ref: item for item in frame.program.input_denotations
    }
    denotation_by_ref.update(
        {item.input_ref: item for item in frame.changed_input_denotations}
    )
    indexes = tuple(
        analyze_requested_fact(
            fact,
            inputs=input_by_ref,
            input_denotations=denotation_by_ref,
        )
        for fact in frame.program.fact_template
    )
    response_values = _grounding_response_values(
        indexes=indexes,
        responses=request.clarification_responses,
    )
    response_use_refs = {
        use_ref for value in response_values for use_ref in value.use_refs
    }
    changed_use_refs = {
        use_ref for argument in frame.arguments for use_ref in argument.input_use_refs
    }
    certified_by_input = {
        value.known_input_id: value for value in frame.certified_argument_values
    }
    unresolved_use_sites = tuple(
        use
        for index in indexes
        for use in index.input_use_sites
        if use.use_ref in changed_use_refs
        and use.use_ref not in response_use_refs
        and use.input_ref not in certified_by_input
    )
    partitions = grounding_partitions(unresolved_use_sites)
    expected_identities = frame.expected_input_identities
    all_resolver_sources = build_row_source_catalog(request.full_catalog)
    options_by_input = {
        partition.input_ref: reference_binding_options(
            input_id=partition.input_ref,
            resolver_catalog=request.full_catalog,
            resolver_row_sources=all_resolver_sources,
            expected_identity=expected_identities.get(partition.input_ref),
        )
        for partition in partitions
        if partition.expected_set_ref is not None
        or partition.reference_fact_ref is not None
    }
    options_by_use_ref = {
        use_ref: options_by_input.get(partition.input_ref, ())
        for partition in partitions
        for use_ref in partition.use_refs
    }
    reference_tasks = reference_grounding_tasks(
        partitions,
        resolver_options_by_use_ref=options_by_use_ref,
        denoted_instance_kinds_by_input_ref=_denoted_instance_kinds(indexes),
    )
    resolver_read_ids = {
        option.candidate.resolver_read_id
        for task in reference_tasks
        for option in task.options
    }
    resolver_catalog = RelationCatalog(
        reads=tuple(
            read for read in request.full_catalog.reads if read.id in resolver_read_ids
        )
    )
    time_tasks = time_grounding_tasks(partitions, inputs=input_by_ref)
    set_origins = {
        ref: term.origin
        for index in indexes
        for ref, term in index.term_by_ref.items()
        if ref.kind.value == "set"
    }
    for partition in partitions:
        if partition.expected_set_ref is None:
            continue
        denotation = denotation_by_ref[partition.input_ref]
        instance_kind = denotation.denoted_instance_kind
        if instance_kind is None:
            raise ValueError("identity grounding requires an instance kind")
        set_origins[partition.expected_set_ref] = SourceOrigin(
            SourceOriginKind.CONVERSATION_RESOLUTION,
            instance_kind,
            resolved_input_ref=partition.input_ref,
        )
    grounding_request = SemanticGroundingRequest(
        question=request.question,
        inputs=frame.changed_inputs,
        tasks=reference_tasks,
        set_origins=set_origins,
        resolver_catalog=resolver_catalog,
        time_tasks=time_tasks,
        runtime_date=(
            request.runtime_values.runtime_date
            if request.runtime_values is not None
            else ""
        ),
        timezone=(
            request.runtime_values.timezone
            if request.runtime_values is not None
            else "timezone.utc"
        ),
    )
    if reference_tasks or time_tasks:
        grounding_result = _turn(
            ModelTurnPurpose.GROUNDING,
            prompt=SemanticGroundingTurnPrompt(grounding_request),
            context=context,
            parse=lambda payload: parse_semantic_grounding(
                payload, request=grounding_request
            ),
            request=request,
            on_turn=on_turn,
        ).result
    else:
        grounding_result = SemanticGroundingResult((), ())
    canonical_values = [
        CanonicalInputValue(
            canonical_value_id=value.id,
            input_ref=input_ref,
            use_refs=next(
                argument.input_use_refs
                for argument in frame.arguments
                if argument.input_ref == input_ref
            ),
            typed_value=value,
            certification_refs=value.proof_refs,
        )
        for input_ref, value in certified_by_input.items()
    ]
    canonical_values.extend(response_values)
    canonical_values.extend(
        deterministic_scalar_values(partitions, inputs=input_by_ref)
    )
    canonical_values.extend(grounding_result.canonical_values)
    if grounding_result.identity_tasks:
        answer_catalog = AuthorizedExecutionSources.from_program(
            full_catalog=request.full_catalog,
            program=frame.program,
        ).relation_catalog
        eligibility_request = SemanticReadEligibilityRequest(
            indexes=indexes,
            source_catalog=build_row_source_catalog(answer_catalog),
            answer_catalog=answer_catalog,
            identity_tasks=grounding_result.identity_tasks,
            resolver_catalog=resolver_catalog,
        )
        eligibility = _read_eligibility_turn(
            eligibility_request,
            context=context,
            request=request,
            on_turn=on_turn,
        )
        tasks = {item.task_ref: item for item in grounding_result.identity_tasks}
        for outcome in eligibility.identity_outcomes:
            if not isinstance(outcome, IdentityRouteSelection):
                return IdentityExecutionClarification(
                    task_ref=outcome.task_ref,
                    input_ref=tasks[outcome.task_ref].input_ref,
                    use_refs=tasks[outcome.task_ref].use_refs,
                    reason=IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT,
                    evidence_refs=outcome.evidence_refs,
                )
            task = tasks[outcome.task_ref]
            resolved = execute_identity_selection(
                task=task,
                selection=outcome,
                input_term=input_by_ref[task.input_ref],
                full_catalog=request.full_catalog,
                data_access_port=request.data_access_port,
                source_read_key_prefix="semantic_continuation_identity_resolution",
            )
            if isinstance(resolved, IdentityExecutionClarification):
                return resolved
            canonical_values.append(resolved.canonical_value)
    canonical_ledger = build_canonical_input_ledger(
        tuple(canonical_values),
        required_use_refs=tuple(changed_use_refs),
    )
    return tuple(value.typed_value for value in canonical_ledger)


@dataclass(frozen=True)
class SemanticCompilationTurnError(Exception):
    purpose: ModelTurnPurpose
    failure: SemanticTurnGenerationError


@dataclass(frozen=True)
class _GroundedQuestion:
    parsed: ParsedSemanticQuestionContract
    catalog_selection: CatalogSelectionResult
    answer_sources: RowSourceCatalog
    resolver_catalog: RelationCatalog
    canonical_values: tuple[CanonicalInputValue, ...]
    grounding_result: SemanticGroundingResult


def compile_semantic_question(
    request: SemanticCompilationRequest,
    *,
    on_turn: SemanticTurnObserver | None = None,
) -> SemanticCompilationOutcome:
    context = build_turn_prompt_context(
        current_question=request.question,
        conversation_context=request.conversation_context,
        host=request.host,
    )
    parsed_question = _parse_question_contract(
        request,
        context=context,
        on_turn=on_turn,
    )
    if not isinstance(parsed_question, ParsedSemanticQuestionContract):
        return SemanticCompilationClarification(cause=parsed_question)
    grounded = _recall_and_ground(
        parsed_question,
        request=request,
        context=context,
        on_turn=on_turn,
    )
    indexes = grounded.parsed.semantic_indexes
    contract = grounded.parsed.contract
    inputs = {item.id: item for item in contract.inputs}

    eligibility_request = SemanticReadEligibilityRequest(
        indexes=indexes,
        source_catalog=grounded.answer_sources,
        answer_catalog=grounded.catalog_selection.relation_catalog,
        identity_tasks=grounded.grounding_result.identity_tasks,
        resolver_catalog=grounded.resolver_catalog,
    )
    eligibility = _read_eligibility_turn(
        eligibility_request,
        context=context,
        request=request,
        on_turn=on_turn,
    )
    canonical_values = _resolve_identity_tasks(
        grounded,
        eligibility=eligibility,
        inputs=inputs,
        request=request,
    )
    if isinstance(canonical_values, SemanticCompilationClarification):
        return canonical_values
    canonical_values = build_canonical_input_ledger(
        canonical_values,
        required_use_refs=tuple(
            use.use_ref for index in indexes for use in index.input_use_sites
        ),
    )
    return _select_bind_and_compile(
        grounded,
        initial_eligibility=eligibility,
        canonical_values=canonical_values,
        context=context,
        request=request,
        on_turn=on_turn,
    )


def _parse_question_contract(
    request: SemanticCompilationRequest,
    *,
    context: TurnPromptContext,
    on_turn: SemanticTurnObserver | None,
) -> ParsedSemanticQuestionContract | QuestionContractNeedsClarification:
    conversation_text = _conversation_input_text(request.question_contract_request)
    frame_turn = _turn(
        ModelTurnPurpose.QUESTION_CONTRACT,
        prompt=SemanticQuestionFrameTurnPrompt(request.question_contract_request),
        context=context,
        parse=lambda payload: parse_semantic_question_frame(
            payload,
            question_context_texts=(request.question,),
            conversation_text_by_resolved_input_ref=conversation_text,
        ),
        request=request,
        on_turn=on_turn,
    )
    question_meaning = frame_turn.result
    if not isinstance(question_meaning, ParsedSemanticQuestionMeaning):
        return question_meaning
    question_turn = _turn(
        ModelTurnPurpose.QUESTION_CONTRACT,
        prompt=SemanticQuestionContractTurnPrompt(
            request.question_contract_request,
            meaning=question_meaning,
        ),
        context=context,
        parse=lambda payload: parse_semantic_question_contract(
            payload,
            meaning=question_meaning,
            question_context_texts=(request.question,),
            conversation_text_by_resolved_input_ref=conversation_text,
        ),
        request=request,
        on_turn=on_turn,
    )
    parsed_question = question_turn.result
    return parsed_question


def _recall_and_ground(
    parsed_question: ParsedSemanticQuestionContract,
    *,
    request: SemanticCompilationRequest,
    context: TurnPromptContext,
    on_turn: SemanticTurnObserver | None,
) -> _GroundedQuestion:
    indexes = parsed_question.semantic_indexes
    contract = parsed_question.contract
    inputs = {item.id: item for item in contract.inputs}
    response_values = _grounding_response_values(
        indexes=indexes,
        responses=request.clarification_responses,
    )
    certified_use_refs = {
        use_ref for value in response_values for use_ref in value.use_refs
    }
    recall_buckets = tuple(
        bucket for index in indexes for bucket in semantic_recall_buckets(index)
    )
    reference_tasks = tuple(
        task
        for task in reference_input_recall_tasks(indexes, inputs=inputs)
        if task.input_use_ref not in certified_use_refs
    )
    from fervis.lookup.query_enrichment.semantic import semantic_recall_requirements
    query_request = SemanticQueryEnrichmentRequest(
        recall_buckets=recall_buckets,
        reference_tasks=reference_tasks,
        resource_names=_resource_names(request.full_catalog),
        requirements=tuple(item for index in indexes for item in semantic_recall_requirements(index)),
    )
    query_turn = _turn(
        ModelTurnPurpose.QUERY_ENRICHMENT,
        prompt=SemanticQueryEnrichmentTurnPrompt(query_request),
        context=context,
        parse=lambda payload: parse_semantic_query_enrichment(
            payload, request=query_request
        ),
        request=request,
        on_turn=on_turn,
    )
    query_result = query_turn.result
    catalog_selection = select_semantic_relation_catalog(
        SemanticCatalogSelectionRequest(
            relation_catalog=request.full_catalog,
            indexes=indexes,
            resource_matches=query_result.recall_bucket_matches,
            max_reads_per_fact=request.max_catalog_reads_per_fact,
        )
    )
    answer_sources = build_row_source_catalog(
        catalog_selection.relation_catalog,
        memory_relations=request.memory_relations,
    )
    resolver_catalog, options_by_use_ref = _resolver_options(
        full_catalog=request.full_catalog,
        reference_tasks=reference_tasks,
        search_terms=query_result.input_resource_search_terms,
    )
    partitions = grounding_partitions(
        tuple(
            use
            for index in indexes
            for use in index.input_use_sites
            if use.use_ref not in certified_use_refs
        )
    )
    grounding_tasks = reference_grounding_tasks(
        partitions,
        resolver_options_by_use_ref=options_by_use_ref,
        denoted_instance_kinds_by_input_ref=_denoted_instance_kinds(indexes),
    )
    time_tasks = time_grounding_tasks(partitions, inputs=inputs)
    grounding_request = SemanticGroundingRequest(
        question=request.question,
        inputs=contract.inputs,
        tasks=grounding_tasks,
        set_origins={
            ref: term.origin
            for index in indexes
            for ref, term in index.term_by_ref.items()
            if ref.kind.value == "set"
        },
        resolver_catalog=resolver_catalog,
        time_tasks=time_tasks,
        runtime_date=(
            request.runtime_values.runtime_date
            if request.runtime_values is not None
            else ""
        ),
        timezone=(
            request.runtime_values.timezone
            if request.runtime_values is not None
            else "timezone.utc"
        ),
    )
    if grounding_tasks or time_tasks:
        grounding_turn = _turn(
            ModelTurnPurpose.GROUNDING,
            prompt=SemanticGroundingTurnPrompt(grounding_request),
            context=context,
            parse=lambda payload: parse_semantic_grounding(
                payload, request=grounding_request
            ),
            request=request,
            on_turn=on_turn,
        )
        grounding_result = grounding_turn.result
    else:
        grounding_result = SemanticGroundingResult((), ())
    canonical_values: tuple[CanonicalInputValue, ...] = (
        *response_values,
        *deterministic_scalar_values(partitions, inputs=inputs),
        *grounding_result.canonical_values,
    )

    return _GroundedQuestion(
        parsed=parsed_question,
        catalog_selection=catalog_selection,
        answer_sources=answer_sources,
        resolver_catalog=resolver_catalog,
        canonical_values=canonical_values,
        grounding_result=grounding_result,
    )


def _resolve_identity_tasks(
    grounded: _GroundedQuestion,
    *,
    eligibility: SemanticReadEligibilityResult,
    inputs: dict[str, InputTerm],
    request: SemanticCompilationRequest,
) -> tuple[CanonicalInputValue, ...] | SemanticCompilationClarification:
    selected_values = list(grounded.canonical_values)
    identity_task_by_ref = {
        item.task_ref: item for item in grounded.grounding_result.identity_tasks
    }
    for outcome in eligibility.identity_outcomes:
        if not isinstance(outcome, IdentityRouteSelection):
            return SemanticCompilationClarification(
                cause=outcome,
                question_contract=grounded.parsed.contract,
                canonical_values=tuple(selected_values),
            )
        task = identity_task_by_ref[outcome.task_ref]
        resolved = execute_identity_selection(
            task=task,
            selection=outcome,
            input_term=inputs[task.input_ref],
            full_catalog=request.full_catalog,
            data_access_port=request.data_access_port,
            source_read_key_prefix="semantic_identity_resolution",
        )
        if isinstance(resolved, IdentityExecutionClarification):
            return SemanticCompilationClarification(
                cause=resolved,
                question_contract=grounded.parsed.contract,
                canonical_values=tuple(selected_values),
            )
        selected_values.append(resolved.canonical_value)
    return tuple(selected_values)


def _select_bind_and_compile(
    grounded: _GroundedQuestion,
    *,
    initial_eligibility: SemanticReadEligibilityResult,
    canonical_values: tuple[CanonicalInputValue, ...],
    context: TurnPromptContext,
    request: SemanticCompilationRequest,
    on_turn: SemanticTurnObserver | None,
) -> SemanticCompilationOutcome:
    indexes = grounded.parsed.semantic_indexes
    contract = grounded.parsed.contract
    (
        catalog_selection,
        answer_sources,
        eligibility,
        source_catalog,
        strategies,
    ) = _prepare_source_candidates(
        initial_catalog_selection=grounded.catalog_selection,
        initial_answer_sources=grounded.answer_sources,
        initial_eligibility=initial_eligibility,
        canonical_values=canonical_values,
        indexes=indexes,
        context=context,
        request=request,
        on_turn=on_turn,
    )
    blocked = tuple(strategy.requested_fact_id for strategy in strategies if not strategy.branches)
    if blocked:
        return SemanticCompilationImpossible(
            question_contract=contract, canonical_values=canonical_values,
            blocked_fact_ids=blocked, source_contract_snapshot=source_catalog.contract_snapshot,
            reviewed_read_ids=tuple(read.id for read in catalog_selection.relation_catalog.reads),
        )
    strategy_by_fact = {item.requested_fact_id: item for item in strategies}
    verified: list[VerifiedSourceStrategy] = []
    for index in indexes:
        use_refs = {item.use_ref for item in index.input_use_sites}
        strategy = strategy_by_fact[index.requested_fact_id]
        selected_source_catalog = source_catalog.select(
            source_refs=frozenset(
                source_ref
                for branch in strategy.branches
                for source_ref in branch.source_refs
            ),
            relation_evidence_refs=frozenset(
                evidence_ref
                for branch in strategy.branches
                for evidence_ref in branch.relation_evidence_refs
            ),
        )
        source_binding_request = SemanticSourceBindingRequest(
            index=index,
            strategy=strategy,
            source_catalog=selected_source_catalog,
            canonical_values=tuple(
                value
                for value in canonical_values
                if set(value.use_refs).intersection(use_refs)
            ),
            catalog_values=_catalog_values(
                requested_fact_id=index.requested_fact_id,
                source_catalog=selected_source_catalog,
                responses=request.clarification_responses,
            ),
        )
        realization = _turn(
            ModelTurnPurpose.SOURCE_REALIZATION,
            prompt=SemanticSourceRealizationTurnPrompt(source_binding_request),
            context=context,
            parse=lambda payload, current=source_binding_request: compile_source_realization(payload, request=current),
            request=request,
            on_turn=on_turn,
        ).result
        if isinstance(realization, SourceRealizationUnavailable):
            return SemanticCompilationImpossible(
                question_contract=contract, canonical_values=canonical_values,
                blocked_fact_ids=(index.requested_fact_id,),
                source_contract_snapshot=source_catalog.contract_snapshot,
                reviewed_read_ids=tuple(read.id for read in catalog_selection.relation_catalog.reads),
                failed_requirement_refs=realization.unmet_requirement_refs,
            )
        source_binding_request = realization.request
        clarification = source_binding_clarification(source_binding_request)
        if clarification is not None:
            return SemanticCompilationClarification(
                cause=clarification,
                question_contract=contract,
                canonical_values=canonical_values,
            )
        surfaces = membership_surfaces(realization)
        if any(surfaces.values()):
            membership = _turn(
                ModelTurnPurpose.SOURCE_BINDING, prompt=SourceMembershipTurnPrompt(realization),
                context=context, parse=lambda payload, current=realization: parse_source_membership(payload, realization=current),
                request=request, on_turn=on_turn,
            ).result
        else:
            membership = parse_source_membership({branch:{} for branch in surfaces}, realization=realization)
        empty_payload = empty_input_binding_payload(membership.realization.request)
        if empty_payload is not None:
            bound_plan = compile_source_binding_plan(empty_payload, membership=membership)
        else:
            bound_plan = _turn(
                ModelTurnPurpose.SOURCE_BINDING,
                prompt=SemanticSourceBindingTurnPrompt(membership),
                context=context,
                parse=lambda payload, current=membership: compile_source_binding_plan(payload, membership=current),
                request=request, on_turn=on_turn,
            ).result
        verification = verify_source_strategy(
            bound_plan,
            request=source_binding_request,
        )
        if isinstance(verification, SourceStrategyVerificationFailure):
            raise ValueError(
                f"source strategy verification failed: {verification.reason.value}"
            )
        verified.append(verification)
    compilation = compile_verified_source_strategies(tuple(verified))
    return SemanticCompilationSuccess(
        question_contract=contract,
        canonical_values=canonical_values,
        catalog_selection=catalog_selection,
        source_contract_snapshot=source_catalog.contract_snapshot,
        compilation=compilation,
    )


def _catalog_values(
    *,
    requested_fact_id: str,
    source_catalog,
    responses: tuple[ClarificationOwnerResponse, ...],
) -> tuple[CatalogProvidedValue, ...]:
    values: list[CatalogProvidedValue] = []
    for response in responses:
        if (
            not isinstance(response, SourceBindingCatalogInputResponse)
            or response.requested_fact_id != requested_fact_id
        ):
            continue
        target = response.target
        source = source_catalog.source(target.row_source_id)
        parameter = next(
            (item for item in source.params if item.param_ref == target.param_ref),
            None,
        )
        if parameter is None or (
            parameter.id,
            parameter.type.value,
            parameter.choices,
        ) != (target.param_id, target.value_type, target.choices):
            raise ValueError(
                "catalog clarification target no longer matches its source"
            )
        literal_type = _catalog_literal_type(target.value_type)
        value_id = f"catalog_value:{response.response_id}"
        proof_ref = clarification_response_ref(response.response_id)
        values.append(
            CatalogProvidedValue(
                catalog_input_ref=response.clarification_id,
                target_ref=target.param_ref,
                value_id=value_id,
                typed_value=FactValue.literal(
                    id=value_id,
                    literal_type=literal_type,
                    value=response.value,
                    proof_refs=(proof_ref,),
                ),
                certification_refs=(proof_ref,),
            )
        )
    return tuple(values)


def _grounding_response_values(
    *,
    indexes,
    responses: tuple[ClarificationOwnerResponse, ...],
) -> tuple[CanonicalInputValue, ...]:
    values: list[CanonicalInputValue] = []
    indexes_by_fact = {item.requested_fact_id: item for item in indexes}
    for response in responses:
        if not isinstance(response, GroundingIdentityResponse):
            continue
        index = indexes_by_fact.get(response.requested_fact_id)
        if index is None:
            raise ValueError("grounding clarification references an unknown fact")
        use_refs = tuple(
            use.use_ref
            for use in index.input_use_sites
            if use.input_ref == response.known_input_id
            and use.reference_fact_ref is not None
        )
        if not use_refs or response.option.key is None:
            raise ValueError("grounding clarification lacks an identity use or key")
        proof_ref = clarification_response_ref(response.response_id)
        value_id = f"grounded_identity:{response.response_id}"
        if response.option.matched_field and response.option.matched_value != "":
            fact_value = FactValue.identity(
                id=value_id,
                known_input_id=response.known_input_id,
                key=response.option.key,
                display_value=response.option.label,
                matched_field_path=response.option.matched_field,
                matched_field_ref=response.option.matched_field,
                matched_value=response.option.matched_value,
                proof_refs=(proof_ref,),
                source_refs=(response.option.resolver_read_id,),
            )
        else:
            fact_value = FactValue.identity(
                id=value_id,
                known_input_id=response.known_input_id,
                key=response.option.key,
                display_value=response.option.label,
                proof_refs=(proof_ref,),
                source_refs=(response.option.resolver_read_id,),
            )
        values.append(
            CanonicalInputValue(
                canonical_value_id=value_id,
                input_ref=response.known_input_id,
                use_refs=use_refs,
                typed_value=fact_value,
                certification_refs=(proof_ref,),
            )
        )
    return tuple(values)


def _catalog_literal_type(value_type: str) -> LiteralType:
    if value_type in {"decimal", "float", "integer", "number"}:
        return LiteralType.NUMBER
    if value_type in {"bool", "boolean"}:
        return LiteralType.BOOLEAN
    return LiteralType.STRING


def _executable_relation_catalog(catalog: RelationCatalog, *, values: tuple[FactValue, ...]) -> RelationCatalog:
    from fervis.lookup.relation_catalog.selection.results import relation_catalog_for_read_ids
    sources = build_api_row_source_catalog(catalog)
    executable = {source.read_id for source in sources.sources
                  if source_required_inputs_are_satisfiable(source, values=values)}
    return relation_catalog_for_read_ids(catalog, read_ids=tuple(read.id for read in catalog.reads if read.id in executable))


def _bound_recall_selection(selection: CatalogSelectionResult, *, full_catalog: RelationCatalog, values: tuple[FactValue, ...]) -> CatalogSelectionResult:
    executable = {read.id for read in _executable_relation_catalog(full_catalog, values=values).reads}
    return replace(selection, requested_fact_selections=tuple(
        replace(fact, unselected_positive_read_ids=tuple(ref for ref in fact.unselected_positive_read_ids if ref in executable))
        for fact in selection.requested_fact_selections
    ))


def _prepare_source_candidates(
    *,
    initial_catalog_selection: CatalogSelectionResult,
    initial_answer_sources,
    initial_eligibility,
    canonical_values: tuple[CanonicalInputValue, ...],
    indexes,
    context: TurnPromptContext,
    request: SemanticCompilationRequest,
    on_turn: SemanticTurnObserver | None,
):
    initial_catalog_selection = _bound_recall_selection(
        initial_catalog_selection, full_catalog=request.full_catalog,
        values=tuple(value.typed_value for value in canonical_values),
    )
    batches = [initial_catalog_selection]
    eligibility_results = [initial_eligibility]
    while True:
        selection = combine_catalog_selection_batches(
            tuple(batches), full_catalog=request.full_catalog
        )
        sources = (
            initial_answer_sources
            if len(batches) == 1
            else build_row_source_catalog(
                selection.relation_catalog, memory_relations=request.memory_relations
            )
        )
        eligibility = combine_semantic_read_eligibility_results(
            tuple(eligibility_results)
        )
        catalog = build_available_source_catalog(
            sources, read_eligibility=eligibility, snapshot_namespace=request.run_id
        )
        source_refs = frozenset(
            source.id
            for source in catalog.sources
            if source_required_inputs_are_satisfiable(
                source, values=tuple(value.typed_value for value in canonical_values)
            )
        )
        catalog = catalog.select(
            source_refs=source_refs,
            relation_evidence_refs=frozenset(
                edge.evidence_ref
                for edge in catalog.relation_evidence
                if {edge.left_source_ref, edge.right_source_ref} <= source_refs
            ),
        )
        strategies = tuple(
            candidate_source_strategy(index, catalog, canonical_values)
            for index in indexes
        )
        result = selection, sources, eligibility, catalog, tuple(strategies)
        # Type-compatible rows are not evidence of semantic coverage. Review
        # every recalled batch before asking realization to choose its sources.
        next_batch = next_catalog_selection_batch(
            catalog_selection=selection,
            full_catalog=request.full_catalog,
            max_reads_per_fact=request.max_catalog_reads_per_fact,
        )
        if next_batch is None:
            return result
        next_request = SemanticReadEligibilityRequest(
            indexes=indexes,
            source_catalog=build_api_row_source_catalog(next_batch.relation_catalog),
            answer_catalog=next_batch.relation_catalog,
            identity_tasks=(),
            resolver_catalog=RelationCatalog(reads=()),
        )
        eligibility_results.append(
            _read_eligibility_turn(
                next_request, context=context, request=request, on_turn=on_turn
            )
        )
        batches.append(next_batch)


def _read_eligibility_turn(
    eligibility_request: SemanticReadEligibilityRequest,
    *,
    context: TurnPromptContext,
    request: SemanticCompilationRequest,
    on_turn: SemanticTurnObserver | None,
):
    return _turn(
        ModelTurnPurpose.READ_ELIGIBILITY,
        prompt=SemanticReadEligibilityTurnPrompt(eligibility_request),
        context=context,
        parse=lambda payload: parse_semantic_read_eligibility(
            payload, request=eligibility_request
        ),
        request=request,
        on_turn=on_turn,
    ).result


def _turn(
    purpose: ModelTurnPurpose,
    *,
    prompt,
    context: TurnPromptContext,
    parse,
    request: SemanticCompilationRequest,
    on_turn: SemanticTurnObserver | None,
):
    try:
        result = generate_semantic_turn(
            prompt=prompt,
            context=context,
            parse=parse,
            model_port=request.model_port,
            provider=request.provider,
            max_thinking_tokens=request.max_thinking_tokens,
        )
    except SemanticTurnGenerationError as exc:
        raise SemanticCompilationTurnError(purpose, exc) from exc
    if on_turn is not None:
        on_turn(purpose, result)
    return result


def _conversation_input_text(
    request: QuestionContractRequest,
) -> dict[str, str]:
    resolution = request.conversation_resolution
    if resolution is None:
        return {}
    return resolution.question_contract_input_text_by_ref()


def _resolver_options(
    *,
    full_catalog: RelationCatalog,
    reference_tasks,
    search_terms,
):
    terms_by_use = {
        item.input_use_ref: item.catalog_search_terms for item in search_terms
    }
    selected_by_use = {
        task.input_use_ref: select_resolver_reads(
            full_catalog,
            catalog_search_terms=terms_by_use[task.input_use_ref],
            limit=3,
        )
        for task in reference_tasks
    }
    resolver_reads = tuple(
        {read.id: read for reads in selected_by_use.values() for read in reads}.values()
    )
    resolver_catalog = RelationCatalog(reads=resolver_reads)
    options_by_use = {}
    for task in reference_tasks:
        task_catalog = RelationCatalog(reads=selected_by_use[task.input_use_ref])
        options_by_use[task.input_use_ref] = reference_binding_options(
            input_id=task.input_ref,
            resolver_catalog=task_catalog,
            resolver_row_sources=build_row_source_catalog(task_catalog),
            expected_identity=None,
        )
    return resolver_catalog, options_by_use


def _denoted_instance_kinds(
    indexes: tuple[RequestedFactSemanticIndex, ...],
) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for index in indexes:
        for input_ref, denotation in index.input_denotation_by_ref.items():
            kind = denotation.denoted_instance_kind
            if kind is None:
                continue
            prior = kinds.get(input_ref)
            if prior is not None and prior != kind:
                raise ValueError("one input has conflicting denoted instance kinds")
            kinds[input_ref] = kind
    return kinds


def _resource_names(catalog: RelationCatalog) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(name for read in catalog.reads for name in read.resource_names)
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
