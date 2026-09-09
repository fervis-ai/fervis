"""Native question-to-SQL orchestration with canonical grounding and execution."""
from dataclasses import replace

from fervis.lookup.question_contract import (
    ParsedSemanticQuestionMeaning, SemanticQuestionFrameTurnPrompt, parse_semantic_question_frame,
)
from fervis.lookup.question_contract.grounding_context import intent_grounding_contexts
from fervis.lookup.question_contract.model import IntentQuestionContract
from fervis.lookup.query_enrichment import (
    SemanticQueryEnrichmentRequest, SemanticQueryEnrichmentTurnPrompt,
    parse_semantic_query_enrichment, reference_input_recall_tasks,
)
from fervis.lookup.query_enrichment.semantic import SemanticRecallBucket, SemanticRecallBucketKind
from fervis.lookup.relation_catalog.selection.selector.semantic_selection import (
    FactRecallBuckets, SemanticCatalogSelectionRequest, select_semantic_relation_catalog,
)
from fervis.lookup.relation_catalog.selection.batches import combine_catalog_selection_batches, next_catalog_selection_batch
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.representation import inspect_selected_representations
from fervis.lookup.grounding import (
    grounding_partitions, time_grounding_tasks,
    SemanticGroundingRequest, SemanticGroundingTurnPrompt, SemanticGroundingResult,
    parse_semantic_grounding, deterministic_scalar_values, build_canonical_input_ledger,
)
from fervis.lookup.read_eligibility import (
    SemanticReadEligibilityRequest,
    combine_semantic_read_eligibility_results,
)
from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt, parse_query_answer, QueryUnavailable
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.parameters import query_parameter_menu, with_catalog_choices, with_reference_arguments, without_input_parameters
from fervis.lookup.relational_sql.compiler import compile_query_answer, combine_query_answers
from fervis.lookup.answer_program.values import ValueComponent
from fervis.lookup.fact_compilation import FactCompilationResult
from .reference_queries import reference_input_values, plan_fact_references, reference_prerequisites
from fervis.lookup.relation_catalog.selection import select_resolver_reads


def compile_query_question(request, *, on_turn=None):
    # Import the shared boundary here to avoid a module initialization cycle.
    from . import semantic_compilation as shared
    request=replace(request, discovery_failures=[])
    timezone = request.runtime_values.timezone if request.runtime_values else "UTC"
    context=build_turn_prompt_context(current_question=request.question,
        conversation_context=request.conversation_context,host=request.host)
    def turn(purpose,prompt,parse):
        return shared._turn(purpose,prompt=prompt,context=context,parse=parse,
            request=request,on_turn=on_turn).result
    meaning=turn(ModelTurnPurpose.QUESTION_CONTRACT,SemanticQuestionFrameTurnPrompt(request.question_contract_request),
        lambda payload:parse_semantic_question_frame(payload,question_context_texts=(request.question,),
            conversation_text_by_resolved_input_ref=shared._conversation_input_text(request.question_contract_request)))
    if not isinstance(meaning,ParsedSemanticQuestionMeaning):
        return shared.SemanticCompilationClarification(cause=meaning)
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
        references,shared._resource_names(request.full_catalog))
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
            access=shared._discover_read_access(read_ids,request=request,context=context,on_turn=on_turn)
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
        assessment=shared._read_eligibility_turn(eligibility_request,context=context,request=request,on_turn=on_turn)
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
        if request.discovery_failures:
            raise request.discovery_failures[0]
        return shared.SemanticCompilationImpossible(question_contract=intent,canonical_values=canonical,
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
            catalog=request.full_catalog,reference_catalog=resolver_catalog,access=access,question=request.question,
            responses=request.clarification_responses,turn=turn,discover_access=discover_access,timezone=timezone)
        if isinstance(planned_references,QueryUnavailable):
            return shared.SemanticCompilationImpossible(question_contract=intent,canonical_values=canonical,
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
            if request.discovery_failures:
                raise request.discovery_failures[0]
            return shared.SemanticCompilationImpossible(question_contract=intent,canonical_values=canonical,
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
    return shared.SemanticCompilationSuccess(question_contract=compiled.question_contract,canonical_values=canonical,
        catalog_selection=selection,source_contract_snapshot=source_catalog.contract_snapshot,
        compilation=FactCompilationResult(compiled.program,compiled.bindings))
