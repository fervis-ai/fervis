"""Compile independent logical facts through discovered REST source contracts."""

from dataclasses import replace

from fervis.lookup.clarification.model import GroundingIdentityResponse
from fervis.lookup.clarification.context import clarification_response_ref
from fervis.lookup.source_binding.reference_bindings import SelectedReferenceChoice
from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.grounding import (
    SemanticGroundingRequest,
    SemanticGroundingResult,
    SemanticGroundingTurnPrompt,
    deterministic_scalar_values,
    grounding_partitions,
    reference_grounding_tasks,
    time_grounding_tasks,
    build_canonical_input_ledger,
    parse_semantic_grounding,
    reference_binding_options,
    IdentityExecutionClarification,
)
from fervis.lookup.query_enrichment import (
    SemanticQueryEnrichmentRequest,
    SemanticQueryEnrichmentTurnPrompt,
    reference_input_recall_tasks,
    semantic_recall_buckets,
    parse_semantic_query_enrichment,
)
from fervis.lookup.query_enrichment.semantic import semantic_recall_requirements
from fervis.lookup.question_contract import ParsedSemanticQuestionContract
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    build_row_source_catalog,
    build_api_row_source_catalog,
)
from fervis.lookup.relation_catalog.selection import (
    SemanticCatalogSelectionRequest,
    select_semantic_relation_catalog,
    select_resolver_reads,
)
from fervis.lookup.relation_catalog.selection.batches import (
    combine_catalog_selection_batches,
    next_catalog_selection_batch,
)
from fervis.lookup.read_eligibility import (
    SemanticReadEligibilityRequest,
    IdentityRouteSelection,
    execute_identity_selection,
    combine_semantic_read_eligibility_results,
)
from fervis.lookup.source_reads.representation import inspect_selected_representations
from fervis.lookup.source_binding import (
    SourceRealizationUnavailable,
    SourceBindingClarification,
    VerifiedSourceStrategy,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from .logical_planning import (
    author_logical_plan,
    prepare_logical_realizations,
    realize_logical_fact,
    compile_logical_bindings,
)
from . import semantic_compilation as shared


def _selected_reference_choices(responses):
    choices = []
    for response in responses:
        if not isinstance(response, GroundingIdentityResponse) or not response.reference_operand:
            continue
        if response.option.key is None:
            raise ValueError("Reference member clarification requires an entity key")
        choices.append(SelectedReferenceChoice(
            response.requested_fact_id, response.known_input_id,
            response.reference_operand, response.option.key,
            clarification_response_ref(response.response_id),
        ))
    return tuple(choices)


def compile_logical_question(request, *, on_turn=None):
    """Candidate runtime sequence; promotion requires native model and live gates."""
    request = replace(request, discovery_failures=[])
    context = build_turn_prompt_context(
        current_question=request.question,
        conversation_context=request.conversation_context,
        host=request.host,
    )

    def turn(purpose, prompt, parse):
        return shared._turn(
            purpose,
            prompt=prompt,
            context=context,
            parse=parse,
            request=request,
            on_turn=on_turn,
        ).result

    logical = author_logical_plan(request.question_contract_request, turn=turn)
    if not isinstance(logical, ParsedSemanticQuestionContract):
        return shared.SemanticCompilationClarification(cause=logical)
    contract, indexes = logical.contract, logical.semantic_indexes
    inputs = {item.id: item for item in contract.inputs}
    response_values = shared._grounding_response_values(
        indexes=indexes, responses=request.clarification_responses
    )
    certified = {ref for value in response_values for ref in value.use_refs}
    references = tuple(
        task
        for task in reference_input_recall_tasks(indexes, inputs=inputs)
        if task.input_use_ref not in certified
    )
    recall_request = SemanticQueryEnrichmentRequest(
        tuple(bucket for index in indexes for bucket in semantic_recall_buckets(index)),
        references,
        shared._resource_names(request.full_catalog),
        requirements=tuple(
            item for index in indexes for item in semantic_recall_requirements(index)
        ),
    )
    recall = turn(
        ModelTurnPurpose.QUERY_ENRICHMENT,
        SemanticQueryEnrichmentTurnPrompt(recall_request),
        lambda payload: parse_semantic_query_enrichment(
            payload, request=recall_request
        ),
    )
    selection = select_semantic_relation_catalog(
        SemanticCatalogSelectionRequest(
            request.full_catalog,
            indexes,
            recall.recall_bucket_matches,
            request.max_catalog_reads_per_fact,
        )
    )
    searches = {
        item.input_use_ref: item.catalog_search_terms
        for item in recall.input_resource_search_terms
    }
    resolver_reads = {
        task.input_use_ref: select_resolver_reads(
            request.full_catalog,
            catalog_search_terms=searches[task.input_use_ref],
            limit=3,
        )
        for task in references
    }
    resolver_ids = tuple(
        dict.fromkeys(read.id for reads in resolver_reads.values() for read in reads)
    )
    observed = inspect_selected_representations(
        request.full_catalog,
        read_ids=resolver_ids,
        data_access_port=request.data_access_port,
        on_response=request.representation_observer,
        on_failure=request.discovery_failures.append,
    )
    request = replace(request, full_catalog=observed)
    access_ids = tuple(
        dict.fromkeys(
            (
                *selection.selected_read_ids,
                *resolver_ids,
                *(
                    ref
                    for fact in selection.requested_fact_selections
                    for ref in fact.unselected_positive_read_ids
                ),
            )
        )
    )
    identity_access_required = bool(references) or any(
        denotation.denoted_instance_kind is not None
        for denotation in contract.input_denotations
    )
    preflight_read_ids = access_ids if identity_access_required else resolver_ids
    observed = inspect_selected_representations(
        request.full_catalog,
        read_ids=preflight_read_ids,
        data_access_port=request.data_access_port,
        on_response=request.representation_observer,
        on_failure=request.discovery_failures.append,
    )
    request = replace(request, full_catalog=observed)
    from fervis.lookup.source_reads.pagination_discovery import (
        PaginationDiscoveryRequest,
        PaginationDiscoveryPrompt,
        parse_pagination_discovery,
    )

    traversal_request = PaginationDiscoveryRequest(request.full_catalog, preflight_read_ids)
    if traversal_request.targets:
        observed = turn(
            ModelTurnPurpose.SOURCE_ACCESS,
            PaginationDiscoveryPrompt(traversal_request),
            lambda payload: parse_pagination_discovery(
                payload, request=traversal_request
            ),
        )
        request = replace(request, full_catalog=observed)
        selection = select_semantic_relation_catalog(SemanticCatalogSelectionRequest(
            request.full_catalog, indexes, recall.recall_bucket_matches,
            request.max_catalog_reads_per_fact))
    resolver_catalog = RelationCatalog(
        reads=tuple(read for read in observed.reads if read.id in resolver_ids)
    )
    options = {}
    for task in references:
        ids = {read.id for read in resolver_reads[task.input_use_ref]}
        catalog = RelationCatalog(
            reads=tuple(read for read in observed.reads if read.id in ids)
        )
        options[task.input_use_ref] = reference_binding_options(
            input_id=task.input_ref,
            resolver_catalog=catalog,
            resolver_row_sources=build_row_source_catalog(catalog),
            expected_identity=None,
        )
    initial_access_ids = preflight_read_ids
    access = shared._discover_read_access(
        initial_access_ids, request=request, context=context, on_turn=on_turn
    )
    request = replace(request, read_access=access)
    partitions = grounding_partitions(
        tuple(
            use
            for index in indexes
            for use in index.input_use_sites
            if use.use_ref not in certified
        )
    )
    denotations = {item.input_ref:item for item in contract.input_denotations}
    deferred = tuple(partition for partition in partitions
        if partition.requires_identity_resolution and isinstance(inputs[partition.input_ref].operand,str)
        and (bool(denotations[partition.input_ref].reference_descriptions)
             or not any(options.get(ref) for ref in partition.use_refs)))
    tasks = reference_grounding_tasks(
        tuple(partition for partition in partitions if partition not in deferred),
        resolver_options_by_use_ref=options,
        denoted_instance_kinds_by_input_ref=shared._denoted_instance_kinds(indexes),
    )
    temporal = time_grounding_tasks(partitions, inputs=inputs)
    grounding_request = SemanticGroundingRequest(
        question=request.question,
        inputs=contract.inputs,
        tasks=tasks,
        read_access=access,
        resolver_catalog=resolver_catalog,
        set_origins={
            ref: term.origin
            for index in indexes
            for ref, term in index.term_by_ref.items()
            if ref.kind.value == "set"
        },
        time_tasks=temporal,
        runtime_date=request.runtime_values.runtime_date
        if request.runtime_values
        else "",
        timezone=request.runtime_values.timezone if request.runtime_values else "UTC",
    )
    grounding = (
        turn(
            ModelTurnPurpose.GROUNDING,
            SemanticGroundingTurnPrompt(grounding_request),
            lambda payload: parse_semantic_grounding(
                payload, request=grounding_request
            ),
        )
        if tasks or temporal
        else SemanticGroundingResult((), ())
    )
    from fervis.lookup.grounding.reference_literals import reference_input_values
    values = [
        *reference_input_values(deferred, inputs=inputs),
        *response_values,
        *deterministic_scalar_values(partitions, inputs=inputs),
        *grounding.canonical_values,
    ]
    early_canonical = (
        build_canonical_input_ledger(
            tuple(values),
            required_use_refs=tuple(
                use.use_ref for index in indexes for use in index.input_use_sites
            ),
        )
        if not grounding.identity_tasks else None
    )

    def trial(current_request, current_selection, current_eligibility):
        assert early_canonical is not None
        return _realize_eligible_sources(
            current_request, logical=logical, selection=current_selection,
            eligibility=current_eligibility, canonical=early_canonical,
            indexes=indexes, access=current_request.read_access,
            contract=contract, turn=turn,
        )

    initial_selection = selection
    request, selection, eligibility, early_outcome = _assess_catalog_batches(
        request, selection=initial_selection, access=access, indexes=indexes,
        grounding=grounding, resolver_catalog=resolver_catalog,
        context=context, on_turn=on_turn, turn=turn,
        trial=trial if early_canonical is not None else None,
    )
    if isinstance(early_outcome, shared.SemanticCompilationSuccess):
        return early_outcome
    task_by_ref = {task.task_ref: task for task in grounding.identity_tasks}
    for outcome in eligibility.identity_outcomes:
        if not isinstance(outcome, IdentityRouteSelection):
            return shared.SemanticCompilationClarification(
                outcome, contract, tuple(values)
            )
        task = task_by_ref[outcome.task_ref]
        resolved = execute_identity_selection(
            task=task,
            selection=outcome,
            input_term=inputs[task.input_ref],
            full_catalog=request.full_catalog,
            data_access_port=request.data_access_port,
            source_read_key_prefix=f"semantic_identity_resolution:{task.task_ref}",
            source_read_lineage=request.identity_read_lineage,
            read_access=access,
        )
        if isinstance(resolved, IdentityExecutionClarification):
            return shared.SemanticCompilationClarification(
                resolved, contract, tuple(values)
            )
        values.append(resolved.canonical_value)
    canonical = build_canonical_input_ledger(
        tuple(values),
        required_use_refs=tuple(
            use.use_ref for index in indexes for use in index.input_use_sites
        ),
    )
    outcome = early_outcome or _realize_eligible_sources(
        request, logical=logical, selection=selection, eligibility=eligibility,
        canonical=canonical, indexes=indexes, access=access, contract=contract,
        turn=turn,
    )
    if isinstance(outcome, shared.SemanticCompilationSuccess) or grounding.identity_tasks:
        return outcome
    from fervis.lookup.relation_catalog.model import requires_caller_supplied_input

    deferred_access_ids = tuple(
        read_id for read_id in access_ids
        if read_id not in initial_access_ids
        and any(requires_caller_supplied_input(param)
                for param in request.full_catalog.read(read_id).params)
    )
    if not deferred_access_ids:
        return outcome
    access = shared._discover_read_access(
        deferred_access_ids, request=replace(request, read_access=access),
        context=context, on_turn=on_turn,
    )
    request = replace(request, read_access=access)
    request, selection, eligibility, deferred_outcome = _assess_catalog_batches(
        request, selection=initial_selection, access=access, indexes=indexes,
        grounding=grounding, resolver_catalog=resolver_catalog,
        context=context, on_turn=on_turn, turn=turn, trial=trial,
    )
    return deferred_outcome or _realize_eligible_sources(
        request, logical=logical, selection=selection, eligibility=eligibility,
        canonical=canonical, indexes=indexes, access=access, contract=contract,
        turn=turn,
    )


def _realize_eligible_sources(
    request, *, logical, selection, eligibility, canonical, indexes, access,
    contract, turn,
):
    sources = build_row_source_catalog(
        selection.relation_catalog, memory_relations=request.memory_relations
    )
    available = build_available_source_catalog(
        sources,
        read_eligibility=eligibility,
        snapshot_namespace=request.run_id,
        read_access=access,
    )
    per_fact = {
        index.requested_fact_id: build_available_source_catalog(
            sources,
            read_eligibility=eligibility,
            snapshot_namespace=request.run_id,
            read_access=access,
            requested_fact_id=index.requested_fact_id,
        )
        for index in indexes
    }
    binding_requests = prepare_logical_realizations(
        logical, sources_by_fact=per_fact, canonical_values=canonical,
        selected_reference_choices=_selected_reference_choices(request.clarification_responses),
    )
    missing = next(
        (item for item in binding_requests if not item.strategy.branches), None
    )
    if missing is not None:
        return shared.SemanticCompilationImpossible(
            contract, canonical, (missing.index.requested_fact_id,),
            available.contract_snapshot, selection.selected_read_ids,
        )
    verified = []
    for binding_request in binding_requests:
        from fervis.lookup.source_binding.catalog_responses import (
            catalog_response_values,
        )

        binding_request = replace(
            binding_request,
            catalog_values=catalog_response_values(
                requested_fact_id=binding_request.index.requested_fact_id,
                source_catalog=binding_request.source_catalog,
                responses=request.clarification_responses,
            ),
        )
        binding_request = _interpret_populations(
            binding_request,
            turn=turn,
            catalog=selection.relation_catalog,
            on_failure=request.discovery_failures.append,
        )
        result = realize_logical_fact(binding_request, turn=turn)
        if isinstance(result, SourceRealizationUnavailable):
            return shared.SemanticCompilationImpossible(
                contract,
                canonical,
                (result.requested_fact_id,),
                available.contract_snapshot,
                selection.selected_read_ids,
                result.unmet_requirement_refs,
            )
        if isinstance(result, SourceBindingClarification):
            return shared.SemanticCompilationClarification(result, contract, canonical)
        if not isinstance(result, VerifiedSourceStrategy):
            raise ValueError(f"Logical source binding failed verification: {result}")
        verified.append(result)
    compiled = compile_logical_bindings(
        logical,
        requests=tuple(item.request for item in verified),
        bindings_by_fact={
            item.request.index.requested_fact_id: item.binding_plan for item in verified
        },
    )
    return shared.SemanticCompilationSuccess(
        contract,
        canonical,
        selection,
        available.contract_snapshot,
        shared.CompiledAnswer(compiled.answer_program, compiled.initial_bindings),
    )


def _assess_catalog_batches(
    request, *, selection, access, indexes, grounding, resolver_catalog,
    context, on_turn, turn, trial=None,
):
    assessments, batches = [], []
    attempted = None
    batch = selection
    while batch is not None:
        observed = inspect_selected_representations(
            request.full_catalog,
            read_ids=batch.selected_read_ids,
            data_access_port=request.data_access_port,
            on_response=request.representation_observer,
            on_failure=request.discovery_failures.append,
        )
        from fervis.lookup.source_reads.pagination_discovery import (
            PaginationDiscoveryRequest, PaginationDiscoveryPrompt,
            parse_pagination_discovery,
        )
        traversal = PaginationDiscoveryRequest(observed, batch.selected_read_ids)
        if traversal.targets:
            observed = turn(
                ModelTurnPurpose.SOURCE_ACCESS,
                PaginationDiscoveryPrompt(traversal),
                lambda payload: parse_pagination_discovery(payload, request=traversal),
            )
        request = replace(request, full_catalog=observed)
        batch = replace(
            batch,
            relation_catalog=RelationCatalog(
                reads=tuple(
                    read for read in observed.reads
                    if read.id in batch.selected_read_ids
                )
            ),
        )
        assessment_request = SemanticReadEligibilityRequest(
            indexes,
            (
                build_row_source_catalog(
                    batch.relation_catalog, memory_relations=request.memory_relations
                )
                if not assessments
                else build_api_row_source_catalog(batch.relation_catalog)
            ),
            batch.relation_catalog,
            grounding.identity_tasks if not assessments else (),
            resolver_catalog,
            access,
        )
        assessments.append(shared._read_eligibility_turn(
            assessment_request, context=context, request=request, on_turn=on_turn
        ))
        batches.append(batch)
        selection = combine_catalog_selection_batches(
            tuple(batches), full_catalog=request.full_catalog
        )
        eligibility = combine_semantic_read_eligibility_results(tuple(assessments))
        if trial is not None:
            attempted = trial(request, selection, eligibility)
            if isinstance(attempted, shared.SemanticCompilationSuccess):
                return request, selection, eligibility, attempted
        batch = next_catalog_selection_batch(
            catalog_selection=selection,
            full_catalog=request.full_catalog,
            max_reads_per_fact=request.max_catalog_reads_per_fact,
        )
    return (
        request, selection,
        combine_semantic_read_eligibility_results(tuple(assessments)), attempted,
    )


def _interpret_populations(request, *, turn, catalog, on_failure):
    from fervis.lookup.source_binding.population_interpretation import (
        SourcePopulationTurnPrompt,
        population_interpretation_targets,
        parse_population_interpretations,
    )

    for finite_cover in (False, True):
        if not population_interpretation_targets(request, finite_cover=finite_cover):
            continue
        interpretations = turn(
            ModelTurnPurpose.SOURCE_POPULATION,
            SourcePopulationTurnPrompt(
                request, read_catalog=catalog, finite_cover=finite_cover
            ),
            lambda payload, current=request, cover=finite_cover: (
                parse_population_interpretations(
                    payload, request=current, finite_cover=cover, on_failure=on_failure
                )
            ),
        )
        replaced = {(item.source_ref, item.parameter_ref) for item in interpretations}
        request = replace(
            request,
            population_interpretations=(
                *(
                    item
                    for item in request.population_interpretations
                    if (item.source_ref, item.parameter_ref) not in replaced
                ),
                *interpretations,
            ),
        )
    return request
