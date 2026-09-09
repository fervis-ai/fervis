"""API invocation declarations own a SQL name and all its argument bindings."""
import pytest
from jsonschema import validate, ValidationError
from fervis.lookup.relational_sql.authoring import parse_query_answer, QueryAnswerPrompt
from fervis.lookup.relational_sql.execution import QueryValidationError


def body(invocations, query='SELECT COUNT(*) AS total FROM selected'):
    return {'query':query,'mode':'scalar','columns':[{'name':'total','value_type':'integer'}],
        'outputs':[{'kind':'value','column':'total','label':'count'}],'ordering':[],
        'api_invocations':invocations,'interpretations':[]}


def definitions():
    return {'long_api_definition_name':{'columns':{'id':{'type':'integer'}},
        'request_parameters':[{'param_ref':'record_id','type':'integer','source':'query'}]}}


def test_invocation_name_is_independent_of_catalog_name_and_sql_range_alias():
    tables=definitions()
    request=body([{'view':'long_api_definition_name','name':'selected',
        'arguments':[{'parameter_ref':'record_id','binding':'p1'}]}],
        'SELECT COUNT(*) AS total FROM selected AS s')
    result=parse_query_answer(request,table_names=set(tables),parameter_names={'p1'},tables=tables,
        parameter_descriptions={'p1':{'value_type':'integer','value':42}})
    assert result.referenced_views == ('selected',)
    assert result.request_arguments[0].sql_view == 'selected'
    assert result.request_arguments[0].view == 'long_api_definition_name'


def test_argument_free_invocation_is_still_an_explicit_sql_source():
    tables=definitions()
    result=parse_query_answer(body([{'view':'long_api_definition_name','name':'selected','arguments':[]}]),
        table_names=set(tables),parameter_names=set(),tables=tables)
    assert result.referenced_views == ('selected',)
    assert result.api_invocations[0].name == 'selected'


def test_catalog_definition_is_not_an_implicit_default_sql_invocation():
    tables=definitions()
    with pytest.raises(QueryValidationError,match='unregistered'):
        parse_query_answer(body([], 'SELECT COUNT(*) AS total FROM long_api_definition_name'),
            table_names=set(tables),parameter_names=set(),tables=tables)


def test_invocations_with_duplicate_sql_names_are_rejected():
    invocation={'view':'long_api_definition_name','name':'selected','arguments':[]}
    with pytest.raises(QueryValidationError,match='unique'):
        parse_query_answer(body([invocation,invocation]),table_names=set(definitions()),parameter_names=set(),tables=definitions())


def test_provider_schema_groups_bindings_under_their_api_definition():
    tables=definitions()
    prompt=QueryAnswerPrompt(question='Count records.',meaning=None,tables=tables,
        parameters={'p1':{'value_type':'integer','value':42}})
    request=body([{'view':'long_api_definition_name','name':'selected',
        'arguments':[{'parameter_ref':'record_id','binding':'p1'}]}])
    validate(request,prompt._schema())
    request['api_invocations'][0]['arguments'][0]['parameter_ref']='invented'
    with pytest.raises(ValidationError):validate(request,prompt._schema())


def test_sql_source_identifiers_follow_duckdb_case_insensitive_rules():
    tables=definitions()
    result=parse_query_answer(body([{'view':'long_api_definition_name','name':'selected','arguments':[]}],
        'SELECT COUNT(*) AS total FROM SELECTED AS S'),table_names=set(tables),parameter_names=set(),tables=tables)
    assert result.referenced_views == ('selected',)

@pytest.mark.parametrize('bound_view', ['b', 'a'])
def test_sql_names_cannot_change_another_api_definitions_argument_contract(bound_view):
    tables={view:{'columns':{'id':{'type':'integer'}},'request_parameters':[
        {'param_ref':'key','source':'query','type':kind}]} for view,kind in [('a','integer'),('b','string')]}
    invocations=[{'view':'a','name':'b','arguments':[]}, {'view':'b','name':'second','arguments':[]}]
    next(item for item in invocations if item['view']==bound_view)['arguments']=[{'parameter_ref':'key','binding':'p1'}]
    request=body(invocations,'SELECT (SELECT COUNT(*) FROM b) + (SELECT COUNT(*) FROM second) AS total')
    args=dict(table_names=set(tables),parameter_names={'p1'},tables=tables,
        parameter_descriptions={'p1':{'value_type':'string','value':'abc'}})
    if bound_view=='b':
        result=parse_query_answer(request,**args)
        assert result.request_arguments[0].view == 'b'
        assert result.request_arguments[0].sql_view == 'second'
    else:
        with pytest.raises(QueryValidationError,match='argument'):
            parse_query_answer(request,**args)
