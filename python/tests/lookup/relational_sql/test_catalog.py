from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.source_reads.access_model import ReadAccessCatalog, ReadDependency, AccessArgument
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from tests.lookup.relational_engine.test_dependent_reads import _program


def test_projected_catalog_names_execute_through_correlated_request_context():
    _, _, catalog = _program()
    parent, child = build_api_row_source_catalog(catalog).sources
    projected = build_query_view_catalog(catalog)
    view = next(view for view in projected.views if view.row_source_id == child.id)
    path_column = next(name for name,definition in projected.tables[view.name]['columns'].items()
                       if definition['request_parameter_ref'] == 'facility_id')
    assert projected.tables[view.name]['candidate_keys'][0]['components'] == {'id': 'id'}
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name,args))
            return {'responseStatus': 200, 'responseBody': [{'id': 1}, {'id': 2}] if endpoint_name=='facilities' else [{'id': args['facility_id'] * 10}]}
    access = ReadAccessCatalog((parent,child), (ReadDependency(child.id,parent.id,
        (AccessArgument('facility_id','facilities.id'),),'Every instrument has a facility.'),))
    compiled=compile_query_answer(question='List the child identifiers for parent 2.',
        query=f'SELECT id FROM "{view.name}" WHERE "{path_column}"=2',
        views=projected.views,catalog=catalog,output_types={'id':'integer'},
        result_contract=ResultContract('rows',('id',)),access=access)
    answer=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert answer.issue is None
    assert [next(iter(row.values.values())) for row in answer.fact_result.outcome.projected_rows]==[20]
    assert calls == [('facilities',{}),('instruments',{'facility_id':1}),('instruments',{'facility_id':2})]


def test_sql_author_does_not_bind_arguments_owned_by_complete_traversal():
    _,_,catalog=_program()
    parent,child=build_api_row_source_catalog(catalog).sources
    access=ReadAccessCatalog((parent,child),(ReadDependency(child.id,parent.id,
        (AccessArgument('facility_id','facilities.id'),),'Every instrument has a facility.'),))
    projected=build_query_view_catalog(catalog,access=access)
    table=projected.tables[child.id]
    assert table['request_parameters']==[]
    assert table['automatic_request_parameters']==['facility_id']


def test_record_population_remains_countable_without_projectable_scalar_fields():
    from dataclasses import replace
    from fervis.lookup.relation_catalog import RelationCatalog,CatalogField
    from tests.lookup.relational_engine.test_dependent_reads import _read
    read=replace(_read('records'),fields=(CatalogField('raw','json',path='raw',row_path_id='root'),),candidate_keys=())
    catalog=RelationCatalog(reads=(read,))
    projected=build_query_view_catalog(catalog)
    assert len(projected.views)==1
    view=projected.views[0]
    assert view.columns=={}
    compiled=compile_query_answer(question='How many records?',query=f'SELECT COUNT(*) AS total FROM "{view.name}"',
        views=projected.views,catalog=catalog,output_types={'total':'integer'},result_contract=ResultContract('scalar'))
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'raw':{'a':1}},{'raw':{'b':[2,3]}}]}
    answer=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert answer.issue is None
    assert next(iter(answer.fact_result.outcome.projected_rows[0].values.values()))==2


def test_sql_catalog_exposes_actual_uuid_and_decimal_representations():
    from dataclasses import replace
    from fervis.lookup.relation_catalog import RelationCatalog,CatalogField
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from fervis.lookup.relational_sql.execution import QueryValidationError
    from tests.lookup.relational_engine.test_dependent_reads import _read
    import pytest
    read=_read('records')
    read=replace(read,fields=(replace(read.fields[0],type='uuid'),
        CatalogField('amount','decimal',path='amount',row_path_id='root')))
    views=build_query_view_catalog(RelationCatalog(reads=(read,)))
    view=views.views[0]
    assert views.tables[view.name]['columns']['id']['type']=='uuid'
    assert views.tables[view.name]['columns']['amount']['type']=='number'
    payload={'query':f'SELECT id FROM "{view.name}"','mode':'rows',
        'columns':[{'name':'id','value_type':'uuid'}],
        'outputs':[{'kind':'identity','authority':view.name+':key:0','components':{'id':'id'},'label':'record','display_column':None}],
        'ordering':[],'request_arguments':[],'interpretations':[]}
    parse_query_answer(payload,table_names=set(views.tables),parameter_names=set(),tables=views.tables)
    erased = {**payload, 'columns': [{'name': 'id', 'value_type': 'string'}]}
    with pytest.raises(QueryValidationError, match='source key type'):
        parse_query_answer(erased, table_names=set(views.tables), parameter_names=set(), tables=views.tables)
    payload['query']=f'SELECT CAST(id AS VARCHAR) AS id FROM "{view.name}"'
    with pytest.raises(QueryValidationError,match='preserve key values'):
        parse_query_answer(payload,table_names=set(views.tables),parameter_names=set(),tables=views.tables)
