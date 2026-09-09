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
    prompt = ReferenceQueryPrompt(question='Find record Alpha.', meaning=meaning, tables=tables,
        parameters={'p1':{'kind':'input','input_ref':'i1','value_type':'string','value':'Alpha'}})
    invocation = prompt.to_model_payload(TurnPromptContext(current_question='Find record Alpha.'))
    enforce_model_turn_prompt_budget(prompt=invocation.prompt_text, tool_specs=prompt.tool_contract().tool_specs)
    encoded = invocation.prompt_text.split('Declared API views:\n',1)[1].split('\n\n',1)[0]
    assert json.loads(encoded) == {name:{key:value for key,value in table.items() if key != 'read_id'} for name,table in tables.items()}


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
