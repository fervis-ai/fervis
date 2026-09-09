import json

from fervis.lookup.relational_sql.reference_planning import ReferenceMeaning, ReferenceQueryPrompt
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.turn_prompts import TurnPromptContext
from fervis.model_io.telemetry import enforce_model_turn_prompt_budget


def test_large_reference_catalog_keeps_all_fields_inside_the_prompt_budget():
    tables = {f'view_{i}': {
        'read_id': f'read_{i}', 'path': f'/records/{i}/', 'description': 'A complete record collection.', 'row_path':'',
        'columns': {name:{'type':'string','description':'Declared record attribute available in the response.',
            'label':name,'nullable':False,'choices':[],'request_parameter_ref':''}
            for name in ('id', *(f'field_{j}' for j in range(39)))},
        'request_parameters': [], 'automatic_request_parameters': [],
        'candidate_keys':[{'entity_kind':'record','key_id':'primary','components':{'id':'id'},'context_columns':[]}],
        'entity_references':[],
    } for i in range(45)}
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, 'record Alpha')
    meaning = ReferenceMeaning('fact_1','i1','record','Identify the named record.',(origin,),('i1',),reference_text='Alpha')
    prompt = ReferenceQueryPrompt(meaning=meaning, tables=tables,
        parameters={'p1':{'kind':'input','input_ref':'i1','value_type':'string','value':'Alpha'}})
    invocation = prompt.to_model_payload(TurnPromptContext(current_question='Find record Alpha.'))
    enforce_model_turn_prompt_budget(prompt=invocation.prompt_text, tool_specs=prompt.tool_contract().tool_specs)
    encoded = invocation.prompt_text.split('Declared API views:\n',1)[1].split('\n\n',1)[0]
    decoded = json.loads(encoded)
    for table in decoded.values():
        for column in table['columns'].values():
            column.pop('identity_roles', None)
    assert decoded == {name:{key:value for key,value in table.items() if key != 'read_id'} for name,table in tables.items()}


def test_prompt_budget_failure_retains_the_actual_preflight_diagnostic():
    from types import SimpleNamespace
    import pytest
    from fervis.lookup.model_turn import run_one_of_tool_model_turn, ModelTurnGenerationFailure
    from fervis.model_io.structured_output.specs import required_tool_spec
    tool = required_tool_spec(tool_name='submit', tool_description='Submit a result', input_schema={'type':'object','properties':{},'required':[],'additionalProperties':False})
    invocation = SimpleNamespace(prompt_text='x'*400001, system_prompt='', provider_schema={}, tool_specs=(tool,))
    with pytest.raises(ModelTurnGenerationFailure) as caught:
        run_one_of_tool_model_turn(invocation=invocation, model_port=object(), provider='openai', max_thinking_tokens=1,
            prompt_budget_error_message='Too large', model_error_message='Provider failed')
    assert caught.value.duration_ms == 0
    assert caught.value.error_context['exception_class'] == 'ModelTurnPromptBudgetError'
    assert '>400000' in caught.value.error_context['message']


def test_reference_member_prompt_excludes_other_members_but_preserves_runtime_evidence():
    question = 'Count observations for the primary record or the record named Cedar.'
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, question)
    meaning = ReferenceMeaning('fact_1','i1','record',question,(origin,),('i1',),
        reference_text='the primary record',reference_is_collection_member=True,reference_kind='description')
    prompt = ReferenceQueryPrompt(meaning=meaning,tables={},parameters={},
        consumer_view_refs=())
    invocation = prompt.to_model_payload(TurnPromptContext(current_question=question))
    assert 'Cedar' not in invocation.prompt_text
    assert 'the primary record' in invocation.prompt_text
    assert prompt.meaning.output_origins == (origin,)
    assert prompt.meaning.return_request_basis == question


def test_reference_definition_is_not_presented_as_a_runtime_parameter():
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, 'the primary record')
    meaning = ReferenceMeaning('fact_1','i1','record','the primary record',(origin,),('i1',),
        reference_text='the primary record',reference_kind='description')
    prompt = ReferenceQueryPrompt(meaning=meaning,tables={},
        parameters={'p1_1':{'input_ref':'i1','value_type':'string','label':'the primary record'}})
    invocation = prompt.to_model_payload(TurnPromptContext(current_question='the primary record'))
    assert 'p1_1' not in invocation.prompt_text
    assert 'the primary record' in invocation.prompt_text
    assert prompt.parameters['p1_1']['kind'] == 'definition'
