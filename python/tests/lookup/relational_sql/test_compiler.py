import pytest
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.relational_sql.acquisition import ApiView
from fervis.lookup.relational_sql.results import ResultContract, ResultOrder
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize(('query','types','contract','expected'),[
    ('SELECT COUNT(*) AS total FROM items',{'total':'number'},ResultContract('scalar'),[3]),
    ('SELECT COUNT(*) AS total FROM items WHERE id>10',{'total':'number'},ResultContract('scalar'),[0]),
    ('SELECT 0 AS witness FROM items WHERE id=1',{'witness':'integer'},ResultContract('existence'),[True]),
    ('SELECT FALSE AS witness FROM items WHERE id=1',{'witness':'boolean'},ResultContract('existence'),[True]),
    ('SELECT id, CASE WHEN id=1 THEN 10 ELSE 5 END AS score FROM items',
     {'id':'integer','score':'integer'},ResultContract('rows',('id',),(ResultOrder('score',True),),'position_with_ties',2),[2,3]),
    ('SELECT id, CASE WHEN id=1 THEN 10 ELSE 5 END AS score FROM items',
     {'id':'integer','score':'integer'},ResultContract('rows',('id',),(ResultOrder('score',True),),'take_with_ties',2),[1,2,3]),
    ('SELECT id FROM items WHERE id>5',{'id':'integer'},ResultContract('existence'),[False]),
    ('SELECT id FROM items WHERE id>2',{'id':'integer'},ResultContract('existence'),[True]),
    ('SELECT id, CASE WHEN id<3 THEN 10 ELSE 5 END AS score FROM items',
     {'id':'integer','score':'integer'},ResultContract('rows',('id',),(ResultOrder('score',True),),'first_with_ties'),[1,2]),
])
def test_compiler_uses_canonical_result_operations(query,types,contract,expected):
    catalog=RelationCatalog(reads=(_read('items'),))
    source=build_api_row_source_catalog(catalog).sources[0]
    views=(ApiView('items',source.id,{'id':source.fields[0].id},{}),)
    compiled=compile_query_answer(question='Evaluate the declared fixture query.',query=query,
        views=views,output_types=types,result_contract=contract,catalog=catalog)
    assert decode_answer_program(canonical_answer_program_json(compiled.program))==compiled.program
    class Port:
        def read(self,**kwargs):
            return {'responseStatus':200,'responseBody':[{'id':1},{'id':2},{'id':3}]}
    executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    rows=executed.fact_result.outcome.projected_rows
    assert [next(iter(row.values.values())) for row in rows]==expected
    assert executed.proof_node_refs_by_result_output_id['result_1']


def test_multiple_answers_share_one_canonical_invocation():
    from fervis.lookup.relational_sql.compiler import combine_query_answers
    catalog=RelationCatalog(reads=(_read('items'),))
    source=build_api_row_source_catalog(catalog).sources[0]
    views=(ApiView('items',source.id,{'id':source.fields[0].id},{}),)
    answers=tuple(compile_query_answer(question=question,query=query,views=views,
        output_types={'value':'integer'},result_contract=ResultContract('scalar'),catalog=catalog,
        fact_id=f'fact_{i}',namespace=f'fact_{i}.')
        for i,(question,query) in enumerate((('How many items?', 'SELECT COUNT(*) AS value FROM items'),
            ('What is the largest identifier?', 'SELECT MAX(id) AS value FROM items')), start=1))
    combined=combine_query_answers(answers,catalog=catalog)
    assert len(combined.program.fact_template)==2
    assert decode_answer_program(canonical_answer_program_json(combined.program))==combined.program
    class Port:
        def __init__(self):
            self.calls=0
        def read(self,**kwargs):
            self.calls+=1
            return {'responseStatus':200,'responseBody':[{'id':2},{'id':7}]}
    port=Port()
    executed=invoke_answer_program(program=combined.program,bindings=combined.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert executed.issue is None
    assert port.calls==1
    assert set(executed.proof_node_refs_by_result_output_id)=={'fact_1.result_1','fact_2.result_1'}


@pytest.mark.parametrize(('source_type','value','output_type'),[
    ('string','001','integer'),('string','true','boolean'),('integer',7,'string'),
    ('string','2026-09-08','date'),('string','2026-09-08T12:00:00','datetime'),
])
def test_typed_sql_output_cannot_be_reinterpreted_as_another_scalar_kind(source_type,value,output_type):
    from fervis.lookup.plan_execution.errors import VerificationError
    catalog=RelationCatalog(reads=(_read('items',value_type=source_type),))
    source=build_api_row_source_catalog(catalog).sources[0]
    compiled=compile_query_answer(question='Return the observed property.',query='SELECT id AS value FROM items',
        views=(ApiView('items',source.id,{'id':source.fields[0].id},{}),),
        output_types={'value':output_type},result_contract=ResultContract('scalar'),catalog=catalog)
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':value}]}
    with pytest.raises(VerificationError,match='SQL.*type'):
        invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))


@pytest.mark.parametrize(('sql_type','value','output_type','expected'),[
    ('INTEGER','001','integer',1),('BOOLEAN','true','boolean',True),
])
def test_explicit_sql_conversion_has_a_typed_result(sql_type,value,output_type,expected):
    catalog=RelationCatalog(reads=(_read('items',value_type='string'),))
    source=build_api_row_source_catalog(catalog).sources[0]
    compiled=compile_query_answer(question='Evaluate the declared conversion.',
        query=f'SELECT CAST(id AS {sql_type}) AS value FROM items',
        views=(ApiView('items',source.id,{'id':source.fields[0].id},{}),),
        output_types={'value':output_type},result_contract=ResultContract('scalar'),catalog=catalog)
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':value}]}
    executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    actual=next(iter(executed.fact_result.outcome.projected_rows[0].values.values()))
    assert actual==expected and type(actual) is type(expected)


def test_result_projection_cannot_name_a_missing_query_column():
    from fervis.lookup.relational_sql.execution import QueryValidationError
    catalog=RelationCatalog(reads=(_read('items'),));source=build_api_row_source_catalog(catalog).sources[0]
    with pytest.raises(QueryValidationError,match='column'):
        compile_query_answer(question='Return the requested value.',query='SELECT id FROM items',
            views=(ApiView('items',source.id,{'id':source.fields[0].id},{}),),output_types={'id':'integer'},
            result_contract=ResultContract('rows',('missing',)),catalog=catalog)
