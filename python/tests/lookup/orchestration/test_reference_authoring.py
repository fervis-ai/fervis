from dataclasses import replace
from types import SimpleNamespace

import pytest
from jsonschema import validate
from fervis.lookup.orchestration.reference_authoring import ReferenceContractPrompt, parse_reference_contracts
from fervis.lookup.orchestration.reference_slots import reference_input_menu
from fervis.lookup.relational_sql.parameters import query_parameter_menu, with_reference_arguments, without_input_parameters
from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt, parse_query_answer
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.execution import QueryValidationError
from fervis.lookup.relation_catalog import RelationCatalog, CatalogParam, ParamSource, EntityKeyComponentTarget, CandidateKey, CandidateKeyComponent
from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.question_contract import InputTerm, InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType
from tests.lookup.relational_engine.test_dependent_reads import _read
from tests.lookup.relational_sql.test_authoring import payload


def reference_contract_payload(authorities):
    return {'reference_contracts':{ref:({'kind':'address'} if authority is None else
        {'kind':'identity','authority':authority}) for ref,authority in authorities.items()}}


def context(*, extra_authorities=0, composite=False, opaque=False):
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, '42')
    term = InputTerm('i1', origin, '42', TextType())
    denotation = InputDenotation('d1', 'i1', 'specified record', 'The input identifies the record.', 'record', InputDenotationKind.IDENTITY_REFERENCE)
    value = FactValue.literal(id='input', known_input_id='i1', literal_type=LiteralType.STRING, value='42', proof_refs=('question_input:i1',))
    menu = reference_input_menu(query_parameter_menu((CanonicalInputValue(value.id, 'i1', ('fact_1:sql_input:i1',), value, value.proof_refs),)), {'i1':denotation})
    records = _read('records', paired=composite)
    if composite:
        records = replace(records, candidate_keys=(CandidateKey('primary', 'records', (
            CandidateKeyComponent('country', 'records.zone'), CandidateKeyComponent('id', 'records.id')), primary=True),))
    params = (CatalogParam('record_id', 'record_id', ParamSource.PATH, 'integer', required=True,
        entity_target=None if opaque else EntityKeyComponentTarget('records', 'primary', 'id')),)
    if composite:
        params += (CatalogParam('country', 'country', ParamSource.PATH, 'string', required=True,
            entity_target=EntityKeyComponentTarget('records', 'primary', 'country')),)
    child = _read('observations', params=params)
    views = build_query_view_catalog(RelationCatalog(reads=(child,)))
    reference_views = build_query_view_catalog(RelationCatalog(reads=(records, *( _read('unrelated_'+str(i)) for i in range(extra_authorities)))))
    fact = SimpleNamespace(requested_fact_id='fact_1', input_refs=('i1',), output_kinds=('value',),
        output_origins=(origin,), ordering_origins=(), result_kind='scalar', selection_kind='all_results', selection_limit_input_ref=None)
    prompt = ReferenceContractPrompt(question='How many observations belong to the specified record?', meaning=fact,
        tables=views.tables, parameters=menu.descriptions, inputs={'i1':term}, denotations={'i1':denotation}, reference_tables=reference_views.tables)
    return prompt, menu, next(iter(views.tables))


def selected_context(*, composite=False, opaque=False, extra_authorities=0):
    prompt,menu,view=context(composite=composite,opaque=opaque,extra_authorities=extra_authorities)
    authority='records/primary(country,id)' if composite else 'records/primary(id)'
    selection=parse_reference_contracts(reference_contract_payload({'i1':authority}),prompt=prompt)
    references=tuple(SimpleNamespace(view=slot.view,table={**slot.table,'kind':'resolved_reference'},input_refs=slot.input_refs) for slot in selection.slots)
    lowered=with_reference_arguments(without_input_parameters(menu,{'i1'}),references)
    tables={**prompt.tables,**{reference.view.name:reference.table for reference in references}}
    return QueryAnswerPrompt(question=prompt.question,meaning=prompt.meaning,tables=tables,parameters=lowered.descriptions),lowered,view,selection


def test_unrelated_authorities_do_not_multiply_selected_reference_symbols():
    selections=[]
    for count in (0,30):
        prompt,menu,_,selection=selected_context(extra_authorities=count)
        assert len(selection.slots)==1
        assert {description['kind'] for description in menu.descriptions.values()}=={'reference_argument'}
        assert len(menu.expressions)==1
        assert prompt.tables['i1']['kind']=='resolved_reference'
        selections.append(selection.slots[0])
    assert selections[0]==selections[1]


def test_composite_rest_binding_projects_each_declared_parameter_component():
    prompt,menu,view,_=selected_context(composite=True)
    symbols={description['projection']:name for name,description in menu.descriptions.items()}
    body=payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',api_bindings=[
        {'view':view,'parameter_ref':parameter,'binding':symbols['key_component:'+component]}
        for parameter,component in [('record_id','id'),('country','country')]])
    validate(body,prompt._schema())
    answer=parse_query_answer(body,table_names=set(prompt.tables),tables=prompt.tables,parameter_names=set(menu.expressions),
        parameter_descriptions=menu.descriptions,meaning=prompt.meaning,expected_input_refs=('i1',))
    assert {argument.parameter_ref:menu.descriptions[argument.binding]['projection'] for argument in answer.request_arguments}=={
        'record_id':'key_component:id','country':'key_component:country'}


@pytest.mark.parametrize('selection',[{}, {'i1':'invented'}, {'i1':None}, {'i1':'records/primary(id)','i2':'records/primary(id)'}])
def test_invalid_reference_selections_are_rejected(selection):
    prompt,_,_=context()
    with pytest.raises(QueryValidationError):parse_reference_contracts({'reference_contracts':selection},prompt=prompt)


def test_resource_address_selection_does_not_create_identity_slots():
    prompt,_,_=context(opaque=True)
    body=reference_contract_payload({'i1':None})
    validate(body,prompt._schema())
    assert parse_reference_contracts(body,prompt=prompt).slots==()


@pytest.mark.parametrize('parameter',['name','country'])
def test_consumer_schema_offers_only_compatible_bound_reference_fields(parameter):
    from jsonschema import ValidationError
    prompt,menu,view,_=selected_context(composite=True,opaque=True)
    prompt.tables[view]['request_parameters'].append({'param_ref':'name','name':'name','type':'string','source':'query'})
    wrong=next(name for name,description in menu.descriptions.items() if description['projection']=='key_component:country')
    if parameter=='name':
        wrong=next(name for name,description in menu.descriptions.items() if description['projection']=='key_component:id')
    body=payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',api_bindings=[
        {'view':view,'parameter_ref':'name' if parameter=='name' else 'record_id','binding':wrong}])
    with pytest.raises(ValidationError):validate(body,prompt._schema())


@pytest.mark.parametrize(('value','source','kind','allowed'),[
    ('42','path','integer',True),('Alpha','path','integer',False),('Alpha','path','string',True),
    ('Alpha','query','string',False),('Alpha','query','uuid',False),
    ('00000000-0000-0000-0000-000000000001','query','uuid',True),
])
def test_literal_address_requires_a_compatible_api_argument(value,source,kind,allowed):
    original,menu,view=context(opaque=True)
    parameter={**original.tables[view]['request_parameters'][0],'type':kind,'source':source}
    prompt=ReferenceContractPrompt(question=original.question,meaning=original.meaning,
        tables={view:{**original.tables[view],'request_parameters':[parameter]}},
        parameters={name:{**item,'value':value,'label':value} for name,item in menu.descriptions.items()},
        inputs={'i1':replace(original.inputs['i1'],operand=value)},denotations=original.denotations,reference_tables=original.reference_tables)
    assert prompt.reference_inputs['i1']['allows_literal_address'] is allowed
    schema=prompt._schema()
    variants=schema['properties']['reference_contracts']['properties']['i1']['anyOf']
    assert any(schema['$defs'][branch['$ref'].rsplit('/',1)[-1]]['properties']['kind']['enum']==['address'] for branch in variants) is allowed


def test_reference_sql_names_are_local_while_program_identifiers_remain_disjoint():
    prompt,_,_=context()
    first=parse_reference_contracts(reference_contract_payload({'i1':'records/primary(id)'}),prompt=prompt).slots[0]
    prompt.meaning=SimpleNamespace(**{**vars(prompt.meaning),'requested_fact_id':'fact_2'})
    second=parse_reference_contracts(reference_contract_payload({'i1':'records/primary(id)'}),prompt=prompt).slots[0]
    assert first.view.name==second.view.name=='i1'
    assert first.reference_id!=second.reference_id
    assert first.view.relation_id!=second.view.relation_id


def test_factual_query_rejects_the_provisional_reference_protocol():
    prompt,menu,view,_=selected_context()
    with pytest.raises(QueryValidationError,match='established before'):
        parse_query_answer(payload(reference_demands=[],query=f'SELECT COUNT(*) AS total FROM "{view}"'),
            table_names=set(prompt.tables),tables=prompt.tables,parameter_names=set(menu.expressions))
    with pytest.raises(QueryValidationError,match='menu symbols'):
        parse_query_answer(payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',api_bindings=[
            {'view':view,'parameter_ref':'record_id','binding':{'reference_input':'i1'}}]),
            table_names=set(prompt.tables),tables=prompt.tables,parameter_names=set(menu.expressions))


def test_factual_prompt_separates_guarded_sql_inputs_from_api_definitions():
    import json
    from fervis.lookup.relational_sql.reference_planning import ReferenceMeaning
    from fervis.lookup.turn_prompts import TurnPromptContext
    prompt, menu, view, _ = selected_context()
    prompt.meaning = ReferenceMeaning('fact_1', 'i1', 'records', 'Count observations.', (), ('i1',),
        reference_text='Alpha', result_kind='scalar', output_kinds=('value',))
    rendered = prompt.to_model_payload(TurnPromptContext(current_question=prompt.question)).prompt_text
    references = json.loads(rendered.split('Resolved SQL inputs:\n',1)[1].split('\n\n',1)[0])
    api = json.loads(rendered.split('Declared API views:\n',1)[1].split('\n\n',1)[0])
    assert set(references) == {'i1'}
    assert set(api) == {view}
    assert references['i1']['columns']['id']['type'] == 'integer'
    for description in menu.descriptions.values():
        assert description['relation_id'] not in rendered
    assert 'columns' not in prompt._schema()['properties']
