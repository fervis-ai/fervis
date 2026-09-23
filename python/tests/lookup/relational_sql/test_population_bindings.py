"""Lexical restrictions can be implemented by a documented API population."""
from types import SimpleNamespace

import pytest

from fervis.lookup.relational_sql.authoring import parse_query_answer
from fervis.lookup.relational_sql.execution import QueryValidationError, SqlTable, execute_query
from tests.lookup.relational_sql.test_authoring import payload


def parse_population(*, binding=None, description='Returns all accepted entries.', parameter=None):
    tables={'entries':{'description':description,'columns':{'id':{'type':'integer'}},'request_parameters':[]}}
    parameters={'p1':parameter or {'input_ref':'i1','kind':'literal','value_type':'string','value':'accepted','may_interpret':True}}
    meaning=SimpleNamespace(result_kind='scalar',selection_kind='all_results',selection_limit_input_ref=None,
        output_origins=('count',),ordering_origins=(),output_kinds=('value',))
    body=payload(query='SELECT COUNT(*) AS total FROM accepted',api_invocations=[{
        'view':'entries','name':'accepted','arguments':[],
        'population_bindings':[binding or {'input':'p1','basis':'The declared API returns only the requested accepted category.'}]}])
    return parse_query_answer(body,table_names=set(tables),tables=tables,parameter_names=set(parameters),
        parameter_descriptions=parameters,meaning=meaning,expected_input_refs=('i1',))


def test_source_population_consumes_lexical_input_without_inventing_a_field_filter():
    authored=parse_population()
    result=execute_query(authored.query,tables={'accepted':SqlTable({'id':'INTEGER'},({'id':1},{'id':3}))},parameters={})
    assert result.rows==((2,),)
    assert authored.parameter_names==()
    assert authored.api_invocations[0].population_bindings[0].input=='p1'


@pytest.mark.parametrize('changes',[
    {'binding':{'input':'unknown','basis':'A population assertion.'}},
    {'binding':{'input':'p1','basis':' '}},
    {'description':''},
    {'parameter':{'input_ref':'i1','kind':'time','value_type':'date','may_interpret':False}},
    {'parameter':{'input_ref':'i1','kind':'literal','value_type':'number','may_interpret':False}},
    {'parameter':{'input_ref':'i1','kind':'reference_literal','value_type':'string','may_interpret':True}},
])
def test_source_population_requires_owned_lexical_input_and_declared_contract(changes):
    with pytest.raises(QueryValidationError,match='population'):
        parse_population(**changes)
