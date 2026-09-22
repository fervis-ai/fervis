"""Author one independent logical specification before physical realization."""
from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.question_contract import (
    ParsedSemanticQuestionMeaning,
    QuestionContractRequest,
    SemanticQuestionFrameTurnPrompt,
    SemanticQuestionContractTurnPrompt,
    parse_semantic_question_frame,
    parse_semantic_question_contract,
)


def author_logical_plan(
    request: QuestionContractRequest,
    *,
    turn,
):
    """Preserve frame commitments in typed expressions, without catalog access.

    The injected turn is the runtime's existing accounted/correctable model
    boundary. Source discovery and physical computation cannot author or replace
    the specification returned here.
    """
    question_texts = (request.current_question,)
    resolution = request.conversation_resolution
    conversation_text = resolution.question_contract_input_text_by_ref() if resolution is not None else {}
    meaning = turn(
        ModelTurnPurpose.QUESTION_CONTRACT,
        SemanticQuestionFrameTurnPrompt(request),
        lambda payload: parse_semantic_question_frame(payload, question_context_texts=question_texts,
            conversation_text_by_resolved_input_ref=conversation_text),
    )
    if not isinstance(meaning, ParsedSemanticQuestionMeaning):
        return meaning
    return turn(
        ModelTurnPurpose.QUESTION_CONTRACT,
        SemanticQuestionContractTurnPrompt(request, meaning=meaning),
        lambda payload: parse_semantic_question_contract(payload, meaning=meaning, question_context_texts=question_texts,
            conversation_text_by_resolved_input_ref=conversation_text),
    )


def _logical_indexes(logical):
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    facts = logical.contract.requested_facts
    indexes = {index.requested_fact_id:index for index in logical.semantic_indexes}
    if (len(indexes) != len(logical.semantic_indexes) or len({fact.id for fact in facts}) != len(facts)
            or set(indexes) != {fact.id for fact in facts}):
        raise ValueError('Logical analysis must cover every requested fact exactly once')
    inputs = {item.id:item for item in logical.contract.inputs}
    denotations = {item.input_ref:item for item in logical.contract.input_denotations}
    for fact in facts:
        expected = analyze_requested_fact(fact, inputs=inputs, input_denotations=denotations)
        if indexes[fact.id] != expected:
            raise ValueError('Logical analysis differs from its independently authored requested fact')
    return indexes


def prepare_logical_realizations(logical, *, sources_by_fact, canonical_values, selected_reference_choices=()):
    """Carry independent fact indexes into physically feasible source scopes."""
    from dataclasses import replace
    from fervis.lookup.source_binding.candidates import candidate_source_strategy
    from fervis.lookup.source_binding.model import SemanticSourceBindingRequest

    requested_ids = tuple(fact.id for fact in logical.contract.requested_facts)
    indexes = _logical_indexes(logical)
    if (len(requested_ids) != len(set(requested_ids)) or set(sources_by_fact) != set(requested_ids)
            or len(indexes) != len(logical.semantic_indexes) or set(indexes) != set(requested_ids)):
        raise ValueError('Source realization must cover every requested fact exactly once')
    if any(choice.requested_fact_id not in indexes for choice in selected_reference_choices):
        raise ValueError('Reference choice targets an unknown requested fact')
    prepared = []
    for fact in logical.contract.requested_facts:
        index = indexes[fact.id]
        if index.requested_fact != fact:
            raise ValueError('Source realization index differs from its requested fact')
        use_refs = {use.use_ref for use in index.input_use_sites}
        values = tuple(replace(value, use_refs=tuple(ref for ref in value.use_refs if ref in use_refs))
                       for value in canonical_values if use_refs.intersection(value.use_refs))
        catalog = sources_by_fact[fact.id]
        prepared.append(SemanticSourceBindingRequest(index,
            candidate_source_strategy(index, catalog, values), catalog, values,
            selected_reference_choices=tuple(
                choice for choice in selected_reference_choices
                if choice.requested_fact_id == fact.id
            )))
    return tuple(prepared)


def compile_logical_bindings(logical, *, requests, bindings_by_fact):
    """Verify physical bindings against the original plan, then lower it."""
    from fervis.lookup.source_binding.verification import verify_source_strategy, VerifiedSourceStrategy
    from fervis.lookup.fact_compilation import compile_verified_source_strategies

    facts = {fact.id: fact for fact in logical.contract.requested_facts}
    by_id = {request.index.requested_fact_id: request for request in requests}
    if (len(facts) != len(logical.contract.requested_facts) or len(by_id) != len(requests)
            or set(by_id) != set(facts) or set(bindings_by_fact) != set(facts)):
        raise ValueError('Bindings must cover each logical fact exactly once')
    indexes = _logical_indexes(logical)
    verified = []
    for fact in logical.contract.requested_facts:
        request = by_id[fact.id]
        if request.index.requested_fact != fact:
            raise ValueError('Physical realization cannot replace its independently authored logical fact')
        if (dict(request.index.input_by_ref) != {item.id:item for item in logical.contract.inputs}
                or dict(request.index.input_denotation_by_ref) != {item.input_ref:item for item in logical.contract.input_denotations}):
            raise ValueError('Physical realization cannot replace logical inputs or denotations')
        if request.index != indexes[fact.id]:
            raise ValueError('Physical realization analysis differs from the independently authored logical fact')
        result = verify_source_strategy(bindings_by_fact[fact.id], request=request)
        if not isinstance(result, VerifiedSourceStrategy):
            raise ValueError(f'Logical fact realization failed verification: {result.reason.value}')
        verified.append(result)
    return compile_verified_source_strategies(tuple(verified))


def realize_logical_fact(request, *, turn):
    """Bind source properties and populations without reauthoring computation."""
    from fervis.lookup.source_binding import (
        SemanticSourceRealizationTurnPrompt, SemanticSourceBindingTurnPrompt,
        compile_source_realization, compile_source_binding_plan,
        SourceRealizationUnavailable, source_binding_clarification, verify_source_strategy,
    )
    from fervis.lookup.source_binding.set_population import SetPopulationTurnPrompt, apply_set_populations
    from fervis.lookup.source_binding.parser import empty_input_binding_payload

    realization = turn(ModelTurnPurpose.SOURCE_REALIZATION,
        SemanticSourceRealizationTurnPrompt(request),
        lambda payload: compile_source_realization(payload, request=request))
    if isinstance(realization, SourceRealizationUnavailable):
        return realization
    selected = realization
    realization = turn(ModelTurnPurpose.SOURCE_REALIZATION,
        SetPopulationTurnPrompt(selected),
        lambda payload: apply_set_populations(payload, realization=selected))
    if isinstance(realization, SourceRealizationUnavailable):
        return realization
    from fervis.lookup.source_binding.reference_prompt import reference_tasks, LiteralReferenceTurnPrompt, bind_literal_references
    if reference_tasks(realization):
        selected = realization
        realization = turn(ModelTurnPurpose.GROUNDING, LiteralReferenceTurnPrompt(selected),
            lambda payload: bind_literal_references(payload, realization=selected))
        if isinstance(realization, SourceRealizationUnavailable):
            return realization
    from fervis.lookup.source_binding.reference_prompt import (
        descriptor_tasks, DescriptorReferenceTurnPrompt, bind_descriptor_references,
    )
    if descriptor_tasks(realization):
        selected = realization
        realization = turn(ModelTurnPurpose.GROUNDING, DescriptorReferenceTurnPrompt(selected),
            lambda payload: bind_descriptor_references(payload, realization=selected))
        if isinstance(realization, SourceRealizationUnavailable):
            return realization
    clarification = source_binding_clarification(realization.request)
    if clarification is not None:
        return clarification
    empty = empty_input_binding_payload(realization.request)
    if empty is not None:
        plan = compile_source_binding_plan(empty, realization=realization)
    else:
        plan = turn(ModelTurnPurpose.SOURCE_BINDING,
            SemanticSourceBindingTurnPrompt(realization),
            lambda payload: compile_source_binding_plan(payload, realization=realization))
    return verify_source_strategy(plan, request=realization.request)


def realize_and_compile_logical_plan(logical, *, sources_by_fact, canonical_values, turn,
                                     selected_reference_choices=()):
    """The typed realization pipeline, with no executable-query authoring step."""
    from fervis.lookup.source_binding import SourceRealizationUnavailable, VerifiedSourceStrategy

    requests = prepare_logical_realizations(logical, sources_by_fact=sources_by_fact,
                                            canonical_values=canonical_values,
                                            selected_reference_choices=selected_reference_choices)
    for request in requests:
        if not request.strategy.branches:
            return SourceRealizationUnavailable(request.index.requested_fact_id,
                tuple(ref.token for ref in sorted(request.index.source_requirement_refs)),
                'Available sources cannot realize the logical sets and their required fields')
    realized = []
    for request in requests:
        outcome = realize_logical_fact(request, turn=turn)
        if not isinstance(outcome, VerifiedSourceStrategy):
            return outcome
        realized.append(outcome)
    return compile_logical_bindings(logical, requests=tuple(item.request for item in realized),
        bindings_by_fact={item.request.index.requested_fact_id:item.binding_plan for item in realized})
