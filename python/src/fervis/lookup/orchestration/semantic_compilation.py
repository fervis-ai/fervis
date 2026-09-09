"""Sole semantic compilation sequence from question text to AnswerProgram."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace, field
from typing import Any

from fervis.lookup.available_sources import (
    SourceContractSnapshot,
    build_available_source_catalog,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.clarification.context import clarification_response_ref
from fervis.lookup.clarification.model import (
    ClarificationOwnerResponse,
    GroundingIdentityResponse,
)
from fervis.lookup.conversation_resolution.callable_frames import CallableFrameProgram
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.relation_catalog.row_sources import (
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
from fervis.lookup.query_enrichment import (
    SemanticQueryEnrichmentRequest,
    reference_input_recall_tasks,
    parse_semantic_query_enrichment,
    SemanticQueryEnrichmentTurnPrompt,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    QueryQuestionContract,
    QuestionContractRequest,
    RequestedFactSemanticIndex,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_frame,
    SemanticQuestionFrameTurnPrompt,
)
from fervis.lookup.read_eligibility import (
    IdentityRouteSelection,
    SemanticReadEligibilityRequest,
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
from fervis.lookup.turn_prompts import (
    HostPromptContext,
    TurnPromptContext,
    build_turn_prompt_context,
)
from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.source_reads.access_model import ReadAccessCatalog
from fervis.lookup.question_contract.grounding_context import GroundingFactContext
from fervis.lookup.question_contract.model import IntentQuestionContract
from fervis.lookup.lineage.source_reads import SourceReadLineageScope



from fervis.lookup.question_contract.grounding_context import intent_grounding_contexts
from fervis.lookup.query_enrichment.semantic import SemanticRecallBucket, SemanticRecallBucketKind
from fervis.lookup.relation_catalog.selection.selector.semantic_selection import (
    FactRecallBuckets,
)
from fervis.lookup.source_reads.representation import inspect_selected_representations
from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt, parse_query_answer, QueryUnavailable
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.parameters import query_parameter_menu, with_catalog_choices, with_reference_arguments, without_input_parameters
from fervis.lookup.relational_sql.compiler import compile_query_answer, combine_query_answers
from fervis.lookup.answer_program.values import ValueComponent
from .reference_queries import reference_input_values, plan_fact_references, reference_prerequisites


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
    identity_read_lineage: SourceReadLineageScope | None = None
    representation_observer: Callable | None = None
    discovery_failures: list[ValueError] = field(default_factory=list)
    read_access: ReadAccessCatalog = ReadAccessCatalog()
    validation_failure_observer: Callable[[ModelTurnPurpose, SemanticTurnGenerationError], None] | None = None


@dataclass(frozen=True)
class CompiledAnswer:
    answer_program: AnswerProgram
    initial_bindings: BindingSet


@dataclass(frozen=True)
class SemanticCompilationSuccess:
    question_contract: QuestionContract | QueryQuestionContract
    canonical_values: tuple[CanonicalInputValue, ...]
    catalog_selection: CatalogSelectionResult
    source_contract_snapshot: SourceContractSnapshot
    compilation: CompiledAnswer


@dataclass(frozen=True)
class SemanticCompilationClarification:
    cause: object
    question_contract: QuestionContract | QueryQuestionContract | IntentQuestionContract | None = None
    canonical_values: tuple[CanonicalInputValue, ...] = ()


@dataclass(frozen=True)
class SemanticCompilationImpossible:
    question_contract: QuestionContract | QueryQuestionContract | IntentQuestionContract
    canonical_values: tuple[CanonicalInputValue, ...]
    blocked_fact_ids: tuple[str, ...]
    source_contract_snapshot: SourceContractSnapshot
    reviewed_read_ids: tuple[str, ...]
    failed_requirement_refs: tuple[str, ...] = ()


SemanticCompilationOutcome = (
    SemanticCompilationSuccess
    | SemanticCompilationClarification
    | SemanticCompilationImpossible
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
    from fervis.lookup.question_contract.grounding_context import grounding_fact_context
    contexts = tuple(grounding_fact_context(fact,inputs=input_by_ref,input_denotations=denotation_by_ref)
                     for fact in frame.program.fact_template)
    indexes = tuple(context.graph_index for context in contexts if context.graph_index is not None)
    response_values = _grounding_response_values(
        indexes=contexts,
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
        for index in contexts
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
        if partition.requires_identity_resolution
    }
    options_by_use_ref = {
        use_ref: options_by_input.get(partition.input_ref, ())
        for partition in partitions
        for use_ref in partition.use_refs
    }
    reference_tasks = reference_grounding_tasks(
        partitions,
        resolver_options_by_use_ref=options_by_use_ref,
        denoted_instance_kinds_by_input_ref=_denoted_instance_kinds(contexts),
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
    read_access = _discover_read_access(
        tuple(resolver_read_ids), request=request, context=context, on_turn=on_turn,
    )
    request = replace(request, read_access=read_access)
    time_tasks = time_grounding_tasks(partitions, inputs=input_by_ref)
    set_origins = {
        ref: term.origin
        for index in contexts
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
        read_access=read_access,
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
            fact_contexts=contexts,
            source_catalog=build_row_source_catalog(answer_catalog),
            answer_catalog=answer_catalog,
            identity_tasks=grounding_result.identity_tasks,
            resolver_catalog=resolver_catalog,
            read_access=read_access,
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
                source_read_key_prefix=f"semantic_continuation_identity_resolution:{task.task_ref}",
                source_read_lineage=request.identity_read_lineage,
                read_access=read_access,
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
















def _grounding_response_values(
    *,
    indexes,
    responses: tuple[ClarificationOwnerResponse, ...],
) -> tuple[CanonicalInputValue, ...]:
    values: list[CanonicalInputValue] = []
    indexes_by_fact = {item.requested_fact_id: item for item in indexes}
    for response in responses:
        if not isinstance(response, GroundingIdentityResponse) or response.reference_operand:
            continue
        index = indexes_by_fact.get(response.requested_fact_id)
        if index is None:
            raise ValueError("grounding clarification references an unknown fact")
        use_refs = tuple(
            use.use_ref
            for use in index.input_use_sites
            if use.input_ref == response.known_input_id
            and (use.is_identity_reference or use.reference_fact_ref is not None or use.identity_set_ref is not None)
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
                source_refs=(response.option.resolver_read_id,) if response.option.resolver_read_id else (),
            )
        else:
            fact_value = FactValue.identity(
                id=value_id,
                known_input_id=response.known_input_id,
                key=response.option.key,
                display_value=response.option.label,
                proof_refs=(proof_ref,),
                source_refs=(response.option.resolver_read_id,) if response.option.resolver_read_id else (),
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
    parsed_payload_for_result=None,
):
    from fervis.lookup.turn_prompts.correction import run_with_correction
    try:
        result = run_with_correction(prompt,
            lambda current: generate_semantic_turn(
                prompt=current, context=context, parse=parse,
                model_port=request.model_port, provider=request.provider,
                max_thinking_tokens=request.max_thinking_tokens,
            ),
            (lambda failure: request.validation_failure_observer(purpose,failure))
            if request.validation_failure_observer is not None else None,
        )
    except SemanticTurnGenerationError as exc:
        raise SemanticCompilationTurnError(purpose, exc) from exc
    if parsed_payload_for_result is not None:
        result = replace(result, artifact=replace(result.artifact, parsed_payload=parsed_payload_for_result(result.result)))
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




def _denoted_instance_kinds(
    indexes: tuple[RequestedFactSemanticIndex | GroundingFactContext, ...],
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




def _discover_read_access(read_ids, *, request, context, on_turn):
    from collections import deque
    from fervis.lookup.source_reads.access_discovery import AccessDiscoveryRequest, ReadAccessTurnPrompt, access_candidates, parse_read_access
    from fervis.lookup.relation_catalog.model import requires_caller_supplied_input
    sources=build_api_row_source_catalog(request.full_catalog)
    targets=tuple(source for source in sources.sources if source.read_id in read_ids
                  and any(requires_caller_supplied_input(param) for param in source.params))
    queue=deque((source,0) for source in targets)
    visited=set()
    dependencies=list(request.read_access.dependencies)
    while queue:
        source,offset=queue.popleft()
        marker=(source.id,offset)
        if marker in visited:
            continue
        visited.add(marker)
        current=ReadAccessCatalog(sources.sources,tuple(dependencies))
        required={param.param_ref for param in source.params if requires_caller_supplied_input(param)}
        if not required or required <= current.supplied_parameters(source):
            continue
        candidates=access_candidates(source,sources=sources,catalog=request.full_catalog)
        if offset>=len(candidates):
            continue
        discovery=AccessDiscoveryRequest(request.full_catalog,sources,(source,),candidate_offset=offset)
        result=_turn(ModelTurnPurpose.SOURCE_ACCESS,prompt=ReadAccessTurnPrompt(discovery),context=context,
            parse=lambda payload,current=discovery:parse_read_access(payload,request=current),
            parsed_payload_for_result=lambda result:{'read_dependencies':[asdict(item) for item in result.dependencies]},
            request=request,on_turn=on_turn).result
        dependencies.extend(result.dependencies)
        queue.append((source,offset+discovery.candidate_limit))
        for dependency in result.dependencies:
            parent=result.source(dependency.parent_source_ref)
            if any(requires_caller_supplied_input(param) for param in parent.params):
                queue.appendleft((parent,0))
    return ReadAccessCatalog(sources.sources,tuple(dependencies))


def compile_semantic_question(request, *, on_turn=None):
    request=replace(request, discovery_failures=[])
    timezone = request.runtime_values.timezone if request.runtime_values else "UTC"
    context=build_turn_prompt_context(current_question=request.question,
        conversation_context=request.conversation_context,host=request.host)
    def turn(purpose,prompt,parse):
        return _turn(purpose,prompt=prompt,context=context,parse=parse,
            request=request,on_turn=on_turn).result
    meaning=turn(ModelTurnPurpose.QUESTION_CONTRACT,SemanticQuestionFrameTurnPrompt(request.question_contract_request),
        lambda payload:parse_semantic_question_frame(payload,question_context_texts=(request.question,),
            conversation_text_by_resolved_input_ref=_conversation_input_text(request.question_contract_request)))
    if not isinstance(meaning,ParsedSemanticQuestionMeaning):
        return SemanticCompilationClarification(cause=meaning)
    contexts=intent_grounding_contexts(meaning,question=request.question)
    intent=IntentQuestionContract(meaning.inputs,tuple(context.requested_fact for context in contexts),meaning.input_denotations)
    inputs={item.id:item for item in meaning.inputs}
    references=reference_input_recall_tasks(contexts,inputs=inputs)
    facts=tuple(FactRecallBuckets(fact.requested_fact_id,(
        SemanticRecallBucket(f'{fact.requested_fact_id}:recall:population',SemanticRecallBucketKind.POPULATION,
            fact.requested_fact.origin,()),
        *(SemanticRecallBucket(f'{fact.requested_fact_id}:recall:{output.id}',SemanticRecallBucketKind.OUTPUT,
            output.origin,()) for output in fact.requested_fact.outputs))) for fact in contexts)
    recall_request=SemanticQueryEnrichmentRequest(tuple(bucket for fact in facts for bucket in fact.buckets),
        references,_resource_names(request.full_catalog))
    recall=turn(ModelTurnPurpose.QUERY_ENRICHMENT,SemanticQueryEnrichmentTurnPrompt(recall_request),
        lambda payload:parse_semantic_query_enrichment(payload,request=recall_request))
    selection=select_semantic_relation_catalog(SemanticCatalogSelectionRequest(request.full_catalog,(),
        recall.recall_bucket_matches,request.max_catalog_reads_per_fact,facts))
    search_terms={item.input_use_ref:item.catalog_search_terms for item in recall.input_resource_search_terms}
    resolver_reads={read.id:read for task in references for read in select_resolver_reads(
        request.full_catalog,catalog_search_terms=search_terms[task.input_use_ref],limit=3)}
    resolver_catalog=RelationCatalog(reads=tuple(resolver_reads.values()))
    observed=inspect_selected_representations(request.full_catalog,read_ids=tuple(resolver_reads),
        data_access_port=request.data_access_port,on_response=request.representation_observer,
        on_failure=request.discovery_failures.append)
    request=replace(request,full_catalog=observed)
    resolver_catalog=RelationCatalog(reads=tuple(read for read in observed.reads if read.id in resolver_reads))
    access=request.read_access
    def discover_access(read_ids):
        nonlocal request,access
        if read_ids:
            access=_discover_read_access(read_ids,request=request,context=context,on_turn=on_turn)
            request=replace(request,read_access=access)
        return access
    partitions=grounding_partitions(tuple(use for fact in contexts for use in fact.input_use_sites))
    temporal=time_grounding_tasks(partitions,inputs=inputs)
    grounding_request=SemanticGroundingRequest(question=request.question,inputs=meaning.inputs,tasks=(),
        read_access=access,set_origins={},resolver_catalog=resolver_catalog,time_tasks=temporal,
        runtime_date=request.runtime_values.runtime_date if request.runtime_values else '',
        timezone=request.runtime_values.timezone if request.runtime_values else 'timezone.utc')
    grounded=turn(ModelTurnPurpose.GROUNDING,SemanticGroundingTurnPrompt(grounding_request),
        lambda payload:parse_semantic_grounding(payload,request=grounding_request)) if temporal else SemanticGroundingResult((),())
    values=[*reference_input_values(partitions,inputs=inputs),
        *deterministic_scalar_values(partitions,inputs=inputs),*grounded.canonical_values]
    batches,assessments=[],[]
    batch=selection
    while batch is not None:
        observed=inspect_selected_representations(request.full_catalog,read_ids=batch.selected_read_ids,
            data_access_port=request.data_access_port,on_response=request.representation_observer,
            on_failure=request.discovery_failures.append)
        request=replace(request,full_catalog=observed)
        selected_ids=set(batch.selected_read_ids)
        batch=replace(batch,relation_catalog=RelationCatalog(reads=tuple(read for read in observed.reads if read.id in selected_ids)))
        eligibility_request=SemanticReadEligibilityRequest(indexes=(),fact_contexts=contexts,
            source_catalog=build_api_row_source_catalog(batch.relation_catalog),answer_catalog=batch.relation_catalog,
            identity_tasks=(),
            resolver_catalog=resolver_catalog,read_access=request.read_access)
        assessment=_read_eligibility_turn(eligibility_request,context=context,request=request,on_turn=on_turn)
        assessments.append(assessment)
        batches.append(batch)
        selection=combine_catalog_selection_batches(tuple(batches),full_catalog=request.full_catalog)
        batch=next_catalog_selection_batch(catalog_selection=selection,full_catalog=request.full_catalog,
            max_reads_per_fact=request.max_catalog_reads_per_fact)
    canonical=build_canonical_input_ledger(tuple(values),required_use_refs=tuple(use.use_ref for fact in contexts for use in fact.input_use_sites))
    eligibility=combine_semantic_read_eligibility_results(tuple(assessments))
    source_catalog=build_available_source_catalog(build_api_row_source_catalog(selection.relation_catalog),
        read_eligibility=eligibility,snapshot_namespace=request.run_id,read_access=access)
    view_catalog=build_query_view_catalog(selection.relation_catalog,access=access)
    fact_source_ids={fact.requested_fact_id:{source.id for source in build_available_source_catalog(
        build_api_row_source_catalog(selection.relation_catalog),read_eligibility=eligibility,
        snapshot_namespace=request.run_id,read_access=access,requested_fact_id=fact.requested_fact_id).sources}
        for fact in meaning.answer_requests}
    fact_views={fact_id:tuple(view for view in view_catalog.views if view.row_source_id in source_ids)
        for fact_id,source_ids in fact_source_ids.items()}
    blocked=tuple(fact_id for fact_id,views in fact_views.items() if not views)
    if blocked:
        return SemanticCompilationImpossible(question_contract=intent,canonical_values=canonical,
            blocked_fact_ids=blocked,source_contract_snapshot=source_catalog.contract_snapshot,
            reviewed_read_ids=selection.selected_read_ids)
    prepared=[]
    for fact in meaning.answer_requests:
        eligible_views=fact_views[fact.requested_fact_id]
        views=tuple(replace(view,name=fact.requested_fact_id+'__'+view.name) for view in eligible_views)
        tables={renamed.name:view_catalog.tables[original.name] for original,renamed in zip(eligible_views,views)}
        local_values=tuple(replace(value,use_refs=tuple(ref for ref in value.use_refs
            if ref.startswith(fact.requested_fact_id+':'))) for value in canonical if value.input_ref in fact.input_refs)
        menu=with_catalog_choices(query_parameter_menu(local_values),source_catalog=source_catalog,
            source_refs={view.row_source_id for view in views})
        planned_references=plan_fact_references(fact=fact,inputs=inputs,
            denotations={item.input_ref:item for item in meaning.input_denotations},values=local_values,
            catalog=request.full_catalog,reference_catalog=resolver_catalog,
            consumer_catalog=RelationCatalog(reads=tuple(read for read in request.full_catalog.reads
                if read.id in {view_catalog.tables[view.name]['read_id'] for view in eligible_views})),
            access=access,question=request.question,
            responses=request.clarification_responses,turn=turn,discover_access=discover_access,timezone=timezone)
        if isinstance(planned_references,QueryUnavailable):
            return SemanticCompilationImpossible(question_contract=intent,canonical_values=canonical,
                blocked_fact_ids=(fact.requested_fact_id,),source_contract_snapshot=source_catalog.contract_snapshot,
                reviewed_read_ids=tuple(resolver_reads))
        tables.update({reference.view.name:reference.table for reference in planned_references})
        reference_refs={reference.table['input_ref'] for reference in planned_references}
        # The final answer consumes established identities through relations. Raw
        # reference text is owned exclusively by those prerequisite queries.
        menu=without_input_parameters(menu,reference_refs)
        menu=with_reference_arguments(menu,planned_references)
        selection_boundary = None
        selection_limit = None
        if fact.selection_limit_input_ref is not None:
            value = next(value for value in local_values if value.input_ref == fact.selection_limit_input_ref)
            number = value.typed_value.payload.component_value(ValueComponent.VALUE)
            selection_limit = int(str(number))
            selection_boundary = next(menu.expressions[name] for name, description in menu.descriptions.items()
                if description['input_ref'] == fact.selection_limit_input_ref and description['projection'] == 'value')
            menu=without_input_parameters(menu,{fact.selection_limit_input_ref})
        authored=turn(ModelTurnPurpose.SOURCE_REALIZATION,
            QueryAnswerPrompt(question=request.question,meaning=fact,tables=tables,parameters=menu.descriptions,timezone=timezone),
            lambda payload:parse_query_answer(payload,table_names=set(tables),parameter_names=set(menu.expressions),
                meaning=fact,selection_limit=selection_limit,expected_input_refs=fact.input_refs,parameter_descriptions=menu.descriptions,tables=tables,
                request_parameters={name:{param['param_ref'] for param in table['request_parameters']} for name,table in tables.items()}))
        if isinstance(authored,QueryUnavailable):
            return SemanticCompilationImpossible(question_contract=intent,canonical_values=canonical,
                blocked_fact_ids=(fact.requested_fact_id,),source_contract_snapshot=source_catalog.contract_snapshot,
                reviewed_read_ids=selection.selected_read_ids)
        prepared.append((fact,authored,views,tables,menu,selection_boundary,planned_references))
    from fervis.lookup.relational_sql.binding import reads_requiring_access_discovery
    selected_reads=tuple(dict.fromkeys(read_id for _,authored,views,_,_,_,_ in prepared
        for read_id in reads_requiring_access_discovery(authored,views,catalog=request.full_catalog)))
    access=discover_access(selected_reads)
    source_catalog=build_available_source_catalog(build_api_row_source_catalog(selection.relation_catalog),
        read_eligibility=eligibility,snapshot_namespace=request.run_id,read_access=access)
    answers=[]
    for fact,authored,views,tables,menu,selection_boundary,planned_references in prepared:
        from fervis.lookup.relational_sql.binding import bind_query_answer
        bound=bind_query_answer(authored,menu,views,selection_boundary=selection_boundary)
        prerequisites,bindings=reference_prerequisites(planned_references,bound.bindings,argument_operations=bound.argument_operations)
        answers.append(compile_query_answer(question=request.question,query=authored.query,views=bound.views,timezone=timezone,
            output_types=authored.output_types,result_contract=authored.result,catalog=request.full_catalog,
            query_parameters=bound.query_parameters,parameters=bound.parameters,bindings=bindings,
            prerequisites=prerequisites,relation_views=tuple(item.view for item in planned_references),
            inputs=meaning.inputs,input_denotations=meaning.input_denotations,access=access,
            output_labels=authored.output_labels,fact_id=fact.requested_fact_id,namespace=fact.requested_fact_id+'.',
            selection_boundary=selection_boundary,meaning_inputs=bound.meaning_inputs,public_outputs=authored.outputs,output_origins=fact.output_origins,expected_input_refs=fact.input_refs))
    compiled=combine_query_answers(tuple(answers),catalog=request.full_catalog)
    referenced_inputs={ref for fact in compiled.question_contract.requested_facts for ref in fact.input_refs}
    if referenced_inputs != {item.id for item in meaning.inputs}:
        from fervis.lookup.relational_sql.execution import QueryValidationError
        raise QueryValidationError('Query omitted a grounded question operand')
    return SemanticCompilationSuccess(question_contract=compiled.question_contract,canonical_values=canonical,
        catalog_selection=selection,source_contract_snapshot=source_catalog.contract_snapshot,
        compilation=CompiledAnswer(compiled.program,compiled.bindings))
