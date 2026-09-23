"""Shared run contracts and model-turn helpers for typed compilation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace, field
from typing import Any

from fervis.lookup.available_sources import (
    SourceContractSnapshot,
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
from fervis.lookup.question_contract import (
    QuestionContract,
    QueryQuestionContract,
    QuestionContractRequest,
    RequestedFactSemanticIndex,
)
from fervis.lookup.read_eligibility import (
    IdentityRouteSelection,
    SemanticReadEligibilityRequest,
    parse_semantic_read_eligibility,
    SemanticReadEligibilityTurnPrompt,
    execute_identity_selection,
)
from fervis.lookup.relation_catalog import RelationCatalog, RelationDataAccessPort
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionResult,
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
    inspection_addresses: dict[str, dict[str, Any]] = field(default_factory=dict)


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
