from fervis.lookup.question_contract import QueryRequestedFact, QueryRequestedOutput, QueryQuestionContract, InputTerm, InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.question_contract.grounding_context import grounding_fact_context
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType
from fervis.lookup.answer_program.result_projection import RelationResultOutput
from fervis.lookup.grounding.semantic import (grounding_partitions, reference_grounding_tasks, deterministic_scalar_values,
    IdentityExecutionClarification, IdentityExecutionFailureReason)
from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.read_eligibility.semantic import SemanticReadEligibilityRequest
from fervis.lookup.read_eligibility.semantic_prompt import SemanticReadEligibilityTurnPrompt
from fervis.lookup.read_eligibility.semantic_schema import build_semantic_read_eligibility_schema
from fervis.lookup.turn_prompts import TurnPromptContext
from fervis.lookup.orchestration.terminal_results import semantic_clarification_fact_result


def _context(identity):
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Precision of the selected instrument')
    output=QueryRequestedOutput.from_projection('r1',origin,RelationResultOutput('r1','answer',field_id='precision',role='answer_value'),'number')
    fact=QueryRequestedFact('fact_1',origin,(),(output,),input_refs=('i1',))
    term=InputTerm('i1',origin,'Cirrus',TextType())
    denotation=InputDenotation('d1','i1','instrument name','Copied replacement operand.',
        'instrument' if identity else None,
        InputDenotationKind.IDENTITY_REFERENCE if identity else InputDenotationKind.NON_IDENTITY_SCALAR)
    return grounding_fact_context(fact,inputs={'i1':term},input_denotations={'i1':denotation})


def test_declared_identity_grounding_needs_no_artificial_graph_refs():
    context=_context(True)
    (use,)=context.input_use_sites
    assert use.identity_set_ref is None and use.reference_fact_ref is None
    partitions=grounding_partitions(context.input_use_sites)
    assert partitions[0].requires_identity_resolution
    tasks=reference_grounding_tasks(partitions,resolver_options_by_use_ref={},denoted_instance_kinds_by_input_ref={'i1':'instrument'})
    assert len(tasks)==1 and tasks[0].input_ref=='i1'
    assert tasks[0].expected_set_ref is None
    assert deterministic_scalar_values(partitions,inputs=context.input_by_ref)==()
    scalar=_context(False)
    values=deterministic_scalar_values(grounding_partitions(scalar.input_use_sites),inputs=scalar.input_by_ref)
    assert len(values)==1 and values[0].use_refs==('fact_1:sql_input:i1',)


def test_eligibility_keeps_sql_request_inputs_and_requirements():
    context=_context(True)
    catalog=RelationCatalog(reads=())
    request=SemanticReadEligibilityRequest(indexes=(),source_catalog=build_api_row_source_catalog(catalog),
        answer_catalog=catalog,identity_tasks=(),resolver_catalog=catalog,fact_contexts=(context,))
    assert request.input_term('i1').operand=='Cirrus'
    schema=build_semantic_read_eligibility_schema(request)
    assert 'fact_1' in schema['properties']['read_assessments_by_requested_fact']['properties']
    prompt=SemanticReadEligibilityTurnPrompt(request).to_model_payload(TurnPromptContext(current_question='And for Cirrus?')).prompt_text
    assert 'Cirrus' in prompt and 'Precision of the selected instrument' in prompt
    assert 'pinned_computation_contract' in prompt


def test_sql_identity_failure_becomes_owned_clarification_for_replacement_input():
    context=_context(True)
    contract=QueryQuestionContract(tuple(context.input_by_ref.values()),(context.requested_fact,),tuple(context.input_denotation_by_ref.values()))
    failure=IdentityExecutionClarification('fact_1:sql_input:i1:reference_grounding','i1',('fact_1:sql_input:i1',),
        IdentityExecutionFailureReason.NOT_FOUND,('source_read:current',))
    result=semantic_clarification_fact_result(failure,contract=contract)
    assert result.outcome.clarifications
    assert 'Cirrus' in str(result.outcome.clarifications[0])


def test_uncompiled_identity_failure_retains_intent_and_lineage_without_a_fake_program():
    from fervis.lookup.question_contract.model import IntentFact, IntentOutput, IntentQuestionContract
    from fervis.lookup.contract_codec import canonical_contract_payload, decode_canonical_contract
    context=_context(True)
    fact=IntentFact('fact_1',context.requested_fact.origin,
        (IntentOutput('r1',context.requested_fact.origin),),('i1',),'qualifying_instances')
    intent=IntentQuestionContract(tuple(context.input_by_ref.values()),(fact,),tuple(context.input_denotation_by_ref.values()))
    restored=decode_canonical_contract(canonical_contract_payload(intent),IntentQuestionContract)
    failure=IdentityExecutionClarification('fact_1:sql_input:i1:reference_grounding','i1',('fact_1:sql_input:i1',),
        IdentityExecutionFailureReason.NOT_FOUND,('source_read:current',))
    result=semantic_clarification_fact_result(failure,contract=restored)
    assert 'Cirrus' in str(result.outcome.clarifications[0])
    assert not hasattr(restored.requested_facts[0],'operations')
