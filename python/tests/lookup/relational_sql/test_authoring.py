import pytest

from fervis.lookup.relational_sql.authoring import parse_query_answer
from fervis.lookup.relational_sql.execution import QueryValidationError


def api_invocations(query, bindings=(), relation_names=()):
    """Build explicit API declarations for these SQL fixtures."""
    from sqlglot import parse_one, exp
    from sqlglot.optimizer.scope import traverse_scope
    from sqlglot.errors import SqlglotError
    grouped = {}
    for item in bindings:
        name = item.get('name') or item['view']
        group = grouped.setdefault((item['view'],name), {'view':item['view'],'name':name,'arguments':[]})
        group['arguments'].append({'parameter_ref':item['parameter_ref'],'binding':item['binding']})
    try:
        sources = [source for scope in traverse_scope(parse_one(query,read='duckdb'))
                   for _,source in scope.selected_sources.values() if isinstance(source,exp.Table)]
    except SqlglotError:
        sources = []
    for source in sources:
        if source.name in relation_names or any(group['name']==source.name for group in grouped.values()):continue
        if any(group['view']==source.name and group['name']==source.alias for group in grouped.values()):continue
        grouped[(source.name,source.name)]={'view':source.name,'name':source.name,'arguments':[]}
    return list(grouped.values())


def payload(**changes):
    bindings = changes.pop('api_bindings', ())
    relation_names = changes.pop('relation_names', ())
    value = {'query':'SELECT COUNT(*) AS total FROM items','mode':'scalar',
        'columns':[{'name':'total','value_type':'integer'}],
        'outputs':[{'kind':'value','column':'total','label':'Item count'}],
        'ordering':[],'interpretations':[]}
    result = {**value, **changes}
    if 'api_invocations' not in result:
        result['api_invocations'] = api_invocations(result['query'],bindings,relation_names)
    for invocation in result['api_invocations']:
        invocation.setdefault('population_bindings', [])
    return result


def test_scalar_answer_declares_a_single_public_value():
    answer = parse_query_answer(payload(), table_names={'items'}, parameter_names=set())
    assert answer.result.mode == 'scalar'
    assert answer.output_types == {'total': 'integer'}
    assert answer.output_labels == {'total': 'Item count'}


@pytest.mark.parametrize('changes', [
    {'query': 'SELECT COUNT(*) AS total FROM invented'},
    {'query': 'SELECT COUNT(*) AS total FROM items WHERE id = $made_up'},
    {'columns': [*payload()['columns'], *payload()['columns']]},
    {'ordering': [{'column': 'missing', 'descending': True}]},
    {'api_bindings': [{'view': 'items', 'parameter_ref': 'limit', 'binding': 'invented'}]},
])
def test_authoring_rejects_unknown_or_inconsistent_declarations(changes):
    with pytest.raises(QueryValidationError):
        parse_query_answer(payload(**changes), table_names={'items'}, parameter_names=set())


def test_rank_column_is_hidden_but_has_declared_result_owner():
    from types import SimpleNamespace
    meaning=SimpleNamespace(result_kind='grouped_results',selection_kind='first_rank_with_ties',output_origins=('item',),ordering_origins=('score',),output_kinds=('value',))
    answer = parse_query_answer(payload(
        query='SELECT id, score FROM items', mode='rows',
        columns=[{'name':'id','value_type':'integer'},{'name':'score','value_type':'number'}],
        outputs=[{'kind':'value','column':'id','label':'Item'}],
        ordering=[{'column':'score','descending':True}],
    ), table_names={'items'}, parameter_names=set(),meaning=meaning)
    assert answer.result.output_columns == ('id',)
    assert answer.result.ordering[0].column == 'score'
    assert answer.result.selection == 'first_with_ties'


def test_single_property_can_be_scalar_without_inventing_a_population_aggregate():
    from types import SimpleNamespace
    meaning=SimpleNamespace(result_kind='qualifying_instances',selection_kind='all_results',output_origins=('precision',),ordering_origins=(),output_kinds=('value',))
    answer=parse_query_answer(payload(query='SELECT precision AS total FROM items WHERE id=$identity'),
        table_names={'items'},parameter_names={'identity'},meaning=meaning)
    assert answer.result.mode=='scalar'
    assert 'COUNT' not in answer.query


def test_query_cannot_omit_requested_outputs_even_when_it_declares_scalar():
    from types import SimpleNamespace
    meaning=SimpleNamespace(result_kind='grouped_results',selection_kind='all_results',output_origins=('instrument','precision'),ordering_origins=(),output_kinds=('identity','value'))
    with pytest.raises(QueryValidationError,match='output inventory'):
        parse_query_answer(payload(),table_names={'items'},parameter_names=set(),meaning=meaning)


def test_provider_schema_excludes_automatically_supplied_arguments():
    from jsonschema import validate, ValidationError
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt
    prompt=QueryAnswerPrompt(question='Property?',meaning=None,
        tables={'items':{'request_parameters':[],'automatic_request_parameters':['parent_id']}},
        parameters={'p1':{}})
    validate(payload(),prompt._schema())
    with pytest.raises(ValidationError):
        validate(payload(api_bindings=[{'view':'items','parameter_ref':'parent_id','binding':'p1'}]),prompt._schema())


def test_provider_schema_binds_only_declared_view_parameter_pairs():
    from jsonschema import validate, ValidationError
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt
    prompt=QueryAnswerPrompt(question='Property?',meaning=None,
        tables={'items':{'request_parameters':[{'param_ref':'item_id'}]},
                'groups':{'request_parameters':[{'param_ref':'group_id'}]}},parameters={'p1':{}})
    with pytest.raises(ValidationError):
        validate(payload(api_bindings=[{'view':'items','parameter_ref':'group_id','binding':'p1'}]),prompt._schema())
    validate(payload(api_bindings=[{'view':'items','parameter_ref':'item_id','binding':'p1'}]),prompt._schema())


def test_explicit_catalog_interpretation_interns_equivalent_typed_sql_literal():
    descriptions={'p1':{'may_interpret':True},'c1':{'kind':'catalog_choice','type':'boolean','value':'true'}}
    answer=parse_query_answer(payload(query='SELECT COUNT(*) AS total FROM items WHERE active=TRUE',
        interpretations=[{'input':'p1','choice':'c1','basis':'The declared field maps this category to true.'}]),
        table_names={'items'},parameter_names=set(descriptions),parameter_descriptions=descriptions)
    assert answer.parameter_names==('c1',)
    assert '$c1' in answer.query


@pytest.mark.parametrize('predicate', ['active=FALSE', "name='true'"])
def test_catalog_interpretation_does_not_replace_a_different_typed_value(predicate):
    descriptions={'p1':{'may_interpret':True},'c1':{'kind':'catalog_choice','type':'boolean','value':'true'}}
    with pytest.raises(QueryValidationError):
        parse_query_answer(payload(query=f'SELECT COUNT(*) AS total FROM items WHERE {predicate}',
            interpretations=[{'input':'p1','choice':'c1','basis':'The declared field maps this category to true.'}]),
            table_names={'items'},parameter_names=set(descriptions),parameter_descriptions=descriptions)


def test_explicit_interpretation_resolves_the_input_symbol_to_its_catalog_value():
    descriptions={'p1':{'may_interpret':True},'c1':{'kind':'catalog_choice','type':'boolean','value':'true'}}
    answer=parse_query_answer(payload(query='SELECT COUNT(*) AS total FROM items WHERE active=$p1',
        interpretations=[{'input':'p1','choice':'c1','basis':'Declared acceptance state.'}]),
        table_names={'items'},parameter_names=set(descriptions),parameter_descriptions=descriptions)
    assert answer.parameter_names==('c1',)
    assert '$p1' not in answer.query


def test_interpretation_resolves_a_request_argument_without_changing_its_owner():
    descriptions={'p1':{'may_interpret':True},'c1':{'kind':'catalog_choice','type':'boolean','value':'false'}}
    answer=parse_query_answer(payload(api_bindings=[{'view':'items','parameter_ref':'validated','binding':'p1'}],
        interpretations=[{'input':'p1','choice':'c1','basis':'Pending validation is false.'}]),
        table_names={'items'},parameter_names=set(descriptions),parameter_descriptions=descriptions,
        request_parameters={'items':{'validated'}})
    assert answer.request_arguments[0].binding=='c1'


@pytest.mark.parametrize(('result_kind','roles'),[
    ('grouped_results',('identity','value')),
    ('qualifying_instances',('identity',)),
    ('grouped_results',('value',)),
])
def test_existence_cannot_replace_the_requested_public_contract(result_kind,roles):
    from types import SimpleNamespace
    meaning=SimpleNamespace(result_kind=result_kind,selection_kind='all_results',
        output_origins=tuple('requested' for _ in roles),output_kinds=roles,ordering_origins=())
    with pytest.raises(QueryValidationError,match='existence|output'):
        parse_query_answer(payload(mode='existence',outputs=[],query='SELECT id AS total FROM items'),
            table_names={'items'},parameter_names=set(),meaning=meaning)


def test_existence_retains_a_single_population_value_obligation():
    from types import SimpleNamespace
    meaning=SimpleNamespace(result_kind='scalar',selection_kind='all_results',
        output_origins=('whether any matches exist',),output_kinds=('value',),ordering_origins=())
    assert parse_query_answer(payload(mode='existence',outputs=[],query='SELECT id AS total FROM items'),
        table_names={'items'},parameter_names=set(),meaning=meaning).result.mode=='existence'


def test_argument_schema_shares_the_binding_symbol_domain():
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt
    params={f'p{index}':{} for index in range(175)}
    tables={f't{index}':{'request_parameters':[{'param_ref':f't{index}.filter'}]} for index in range(60)}
    schema=QueryAnswerPrompt(question='Count matching rows.',meaning=None,tables=tables,parameters=params)._schema()
    def enums(value):
        if isinstance(value,dict):
            return len(value.get('enum',()))+sum(enums(item) for key,item in value.items() if key!='enum')
        if isinstance(value,list):return sum(enums(item) for item in value)
        return 0
    assert enums(schema)<1000


def test_provider_schema_preserves_requested_output_count_and_roles():
    from types import SimpleNamespace
    from jsonschema import validate, ValidationError
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt
    meaning=SimpleNamespace(result_kind='grouped_results',output_kinds=('identity',),
        output_origins=('item',),ordering_origins=('score',))
    tables={'items':{'candidate_keys':[{'entity_kind':'item','key_id':'pk','components':{'id':'id'}}]}}
    schema=QueryAnswerPrompt(question='Which item scored highest?',meaning=meaning,tables=tables,parameters={})._schema()
    answer=payload(mode='rows',columns=[{'name':'id','value_type':'integer'},{'name':'score','value_type':'number'}],
        outputs=[{'kind':'identity','authority':'item/pk(id)','components':{'id':'id'},'label':'item','display_column':None}],
        ordering=[{'column':'score','descending':True}])
    validate(answer,schema)
    with pytest.raises(ValidationError):
        validate({**answer,'outputs':[{'kind':'value','column':'id','label':'item'}]},schema)
    with pytest.raises(ValidationError):
        validate({**answer,'outputs':answer['outputs']*2},schema)
    with pytest.raises(ValidationError):
        validate({**answer,'mode':'existence','outputs':[]},schema)


@pytest.mark.parametrize(('selection','limit','expected'),[
    ('first_rank_with_ties',None,'first_with_ties'),
    ('take_with_boundary_ties',3,'take_with_ties'),
    ('position_with_ties',2,'position_with_ties'),
])
def test_authoring_does_not_redeclare_selection_owned_by_the_question(selection,limit,expected):
    from types import SimpleNamespace
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt
    meaning=SimpleNamespace(result_kind='grouped_results',selection_kind=selection,
        output_origins=('item',),ordering_origins=('score',),output_kinds=('value',))
    schema=QueryAnswerPrompt(question='Rank the items.',meaning=meaning,tables={},parameters={})._schema()
    assert 'selection' not in schema['properties'] and 'limit' not in schema['properties']
    value=payload(query='SELECT id,score FROM items',mode='rows',
        columns=[{'name':'id','value_type':'integer'},{'name':'score','value_type':'number'}],
        outputs=[{'kind':'value','column':'id','label':'item'}],ordering=[{'column':'score','descending':True}])
    value.pop('selection',None);value.pop('limit',None)
    answer=parse_query_answer(value,table_names={'items'},parameter_names=set(),meaning=meaning,selection_limit=limit)
    assert answer.result.selection==expected and answer.result.limit==limit


def _identity_parameter_fixture():
    tables={
        'sites':{'columns':{'id':{'type':'integer'},'name':{'type':'string'}},
            'candidate_keys':[{'entity_kind':'site','key_id':'pk','components':{'id':'id'}}],
            'request_parameters':[{'param_ref':'site_name','type':'string'}]},
        'observations':{'columns':{'id':{'type':'integer'},'site_id':{'type':'integer'}},
            'candidate_keys':[{'entity_kind':'observation','key_id':'pk','components':{'id':'id'}}],
            'entity_references':[{'target_entity_kind':'site','target_key_id':'pk','components':{'id':'site_id'}}],
            'request_parameters':[{'param_ref':'site_id','type':'integer'}]},
    }
    menu={'key':{'kind':'identity','value_type':'integer','projection':'key_component:id',
        'identity':{'entity_kind':'site','key_id':'pk','components':['id']}}}
    return tables,menu


@pytest.mark.parametrize(('query','valid'),[
    ('SELECT COUNT(*) AS total FROM observations WHERE site_id=$key',True),
    ('SELECT COUNT(*) AS total FROM sites WHERE id=$key',True),
    ('SELECT COUNT(*) AS total FROM observations WHERE id=$key',False),
    ('SELECT COUNT(*) AS total FROM sites WHERE name=$key',False),
    ('WITH s AS (SELECT id AS site FROM sites) SELECT COUNT(*) AS total FROM s WHERE site=$key',True),
    ('WITH s AS (SELECT name AS site FROM sites) SELECT COUNT(*) AS total FROM s WHERE site=$key',False),
])
def test_identity_parameters_keep_their_key_authority_in_sql_comparisons(query,valid):
    tables,menu=_identity_parameter_fixture()
    def parse():return parse_query_answer(payload(query=query,relation_names=('reference_i1',)),table_names=set(tables),parameter_names=set(menu),
        tables=tables,parameter_descriptions=menu)
    if valid:parse()
    else:
        with pytest.raises(QueryValidationError,match='identity'):parse()


def test_request_binding_cannot_reinterpret_a_key_as_a_name():
    from jsonschema import validate,ValidationError
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt
    tables,menu=_identity_parameter_fixture()
    bad=payload(query='SELECT COUNT(*) AS total FROM observations o JOIN sites s ON s.id=o.site_id WHERE o.site_id=$key',
        api_bindings=[{'view':'sites','parameter_ref':'site_name','binding':'key'}])
    with pytest.raises(QueryValidationError,match='type'):
        parse_query_answer(bad,table_names=set(tables),parameter_names=set(menu),tables=tables,parameter_descriptions=menu)
    with pytest.raises(ValidationError):
        validate(bad,QueryAnswerPrompt(question='Count observations for this site.',meaning=None,tables=tables,parameters=menu)._schema())
    tables['observations']['request_parameters'][0]['entity_target'] = {'entity_kind':'site','key_id':'pk','component_id':'id'}
    good=payload(query='SELECT COUNT(*) AS total FROM observations WHERE site_id=$key',
        api_bindings=[{'view':'observations','parameter_ref':'site_id','binding':'key'}])
    validate(good,QueryAnswerPrompt(question='Count observations for this site.',meaning=None,tables=tables,parameters=menu)._schema())


@pytest.mark.parametrize('use_reference',[False,True])
def test_input_consumption_follows_guarded_reference_relations(use_reference):
    from types import SimpleNamespace
    meaning=SimpleNamespace(result_kind='scalar',selection_kind='all_results',selection_limit_input_ref=None,
        output_origins=('count',),ordering_origins=(),output_kinds=('value',))
    tables={'records':{'columns':{'site_id':{'type':'integer'}}},
        'reference_i1':{'kind':'resolved_reference','input_refs':['i1'],'columns':{'id':{'type':'integer'}}}}
    query='SELECT COUNT(*) AS total FROM records'+(' JOIN reference_i1 ON records.site_id=reference_i1.id' if use_reference else '')
    def parse():return parse_query_answer(payload(query=query,relation_names=('reference_i1',)),table_names=set(tables),parameter_names=set(),
        meaning=meaning,expected_input_refs=('i1',),tables=tables,parameter_descriptions={})
    if use_reference:parse()
    else:
        with pytest.raises(QueryValidationError,match='input operands'):parse()


def test_unknown_key_column_is_diagnosed_before_identity_lineage():
    tables = {'records': {'columns': {'id': {'type':'integer'}, 'name': {'type':'string'}},
        'candidate_keys':[{'entity_kind':'record','key_id':'primary','components':{'id':'id'}}]}}
    query = payload(query='SELECT data_id AS id FROM records', mode='rows',
        columns=[{'name':'id','value_type':'integer'}],
        outputs=[{'kind':'identity','authority':'record/primary(id)','components':{'id':'id'},'label':'record','display_column':None}])
    with pytest.raises(QueryValidationError) as caught:
        parse_query_answer(query, table_names=set(tables), parameter_names=set(), tables=tables)
    assert 'undeclared column' in str(caught.value)
    assert 'data_id' in str(caught.value)
    assert 'records' in str(caught.value) and 'id' in str(caught.value)


@pytest.mark.parametrize("declared", [["total", "inner_id"], ["wrong"], ["total"]])
def test_authoring_output_inventory_belongs_to_outer_query_not_ctes(declared):
    body = payload(query="WITH members AS (SELECT id AS inner_id FROM items) SELECT COUNT(*) AS total FROM members",
        columns=[{"name": name, "value_type": "integer"} for name in declared],
        outputs=[{"kind": "value", "column": declared[0], "label": "count"}])
    arguments = dict(table_names={"items"}, parameter_names=set(), tables={"items": {"columns": {"id": {"type": "integer"}}}})
    if declared == ["total"]:
        assert parse_query_answer(body, **arguments).output_types == {"total": "integer"}
    else:
        with pytest.raises(QueryValidationError, match="result columns"):
            parse_query_answer(body, **arguments)


@pytest.mark.parametrize('allowed', [('APPROVED',), ('category', 'day')])
def test_interpreted_api_argument_must_belong_to_destination_choices(allowed):
    descriptions = {'p1':{'input_ref':'i1','value_type':'string','value':'approved','may_interpret':True},
        'c1':{'kind':'catalog_choice','type':'choice','value':'APPROVED'}}
    tables = {'items':{'columns':{'id':{'type':'integer'}},'request_parameters':[
        {'param_ref':'selection','source':'query','type':'choice','choices':list(allowed)}]}}
    body = payload(api_bindings=[{'view':'items','parameter_ref':'selection','binding':'p1'}],
        interpretations=[{'input':'p1','choice':'c1','basis':'The approved category maps to the declared APPROVED value.'}])
    args = dict(table_names={'items'},parameter_names=set(descriptions),parameter_descriptions=descriptions,tables=tables)
    if allowed == ('APPROVED',):
        assert parse_query_answer(body,**args).request_arguments[0].binding == 'c1'
    else:
        with pytest.raises(QueryValidationError, match='argument'):
            parse_query_answer(body,**args)
