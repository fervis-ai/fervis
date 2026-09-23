import pytest
from dataclasses import replace
from fervis.lookup.relational_sql.column_usage import required_columns,ROW_PRESENCE_COLUMN
from fervis.lookup.relational_sql.execution import execute_query,SqlTable,QueryValidationError
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.acquisition import ApiView
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.model import CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from tests.lookup.relational_engine.test_dependent_reads import _read


def test_dependency_projection_keeps_filters_joins_and_correlated_columns():
    used=required_columns('SELECT p.id FROM parents p WHERE EXISTS (SELECT 1 FROM children c WHERE c.parent=p.id AND c.score>2)',
        {'parents':{'id','name'},'children':{'id','parent','score','note'}})
    assert used=={'parents':frozenset({'id'}),'children':frozenset({'parent','score'})}


@pytest.mark.parametrize('filtered',[False,True])
def test_count_does_not_materialize_unused_nullable_nested_fields(filtered):
    read=_read('deals')
    read=replace(read,fields=(*read.fields,CatalogField('active','boolean',path='active',row_path_id='root'),
        CatalogField('promo.id','integer',path='promo.id',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0]
    view=ApiView('deals',source.id,{field.path:field.id for field in source.fields if field.path},{})
    query='SELECT COUNT(*) AS total FROM deals'+(' WHERE active=TRUE' if filtered else '')
    compiled=compile_query_answer(question='Count the requested deals.',query=query,views=(view,),
        output_types={'total':'integer'},result_contract=ResultContract('scalar'),catalog=catalog)
    assert {field.field_id for relation in compiled.program.relations for field in relation.fields}==({'active'} if filtered else set())
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':1,'active':True,'promo':None},{'id':2,'active':False,'promo':None}]}
    executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[1 if filtered else 2]


def test_zero_column_rows_are_countable_but_have_no_invented_public_fields():
    tables={'items':SqlTable({},({},{},{}))}
    assert execute_query('SELECT COUNT(*) FROM items',tables=tables).rows==((3,),)
    with pytest.raises(QueryValidationError):
        execute_query(f'SELECT {ROW_PRESENCE_COLUMN} FROM items',tables=tables)


def test_wildcard_exclusion_survives_source_projection():
    read=_read('items')
    read=replace(read,fields=(*read.fields,CatalogField('optional','string',path='optional',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0]
    compiled=compile_query_answer(question='List item identifiers.',query='SELECT * EXCLUDE (optional) FROM items',
        views=(ApiView('items',source.id,{f.path:f.id for f in source.fields if f.path},{}),),
        output_types={'id':'integer'},result_contract=ResultContract('rows',('id',)),catalog=catalog)
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':5}]}
    executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[5]


@pytest.mark.parametrize('parent',[None,{},7])
def test_nullable_parent_propagates_null_without_hiding_missing_or_invalid_structure(parent):
    from fervis.lookup.source_reads.response import EndpointResponseError
    read=_read('items')
    read=replace(read,fields=(*read.fields,
        CatalogField('optional','object',path='optional',row_path_id='root',nullable=True),
        CatalogField('optional.name','string',path='optional.name',row_path_id='root',nullable=True)))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0]
    compiled=compile_query_answer(question='Count records without an optional name.',
        query='SELECT COUNT(*) AS total FROM items WHERE "optional.name" IS NULL',
        views=(ApiView('items',source.id,{f.path:f.id for f in source.fields if f.path in {'id','optional.name'}},{}),),
        output_types={'total':'integer'},result_contract=ResultContract('scalar'),catalog=catalog)
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':1,'optional':parent},{'id':2,'optional':{'name':'Present'}}]}
    def invoke():
        return invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    if parent is not None:
        with pytest.raises(EndpointResponseError,match='unavailable'):invoke()
        return
    answer=invoke()
    assert answer.issue is None
    assert next(iter(answer.fact_result.outcome.projected_rows[0].values.values()))==1
