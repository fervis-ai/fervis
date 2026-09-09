from tests.lookup.relational_sql.test_authoring import payload as query_payload
from dataclasses import replace
import pytest

from fervis.lookup.answer_program.result_projection import EntityKeyProjection, EntityKeyProjectionComponent
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.model import CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.relational_sql.acquisition import ApiView
from fervis.lookup.relational_sql.outputs import QueryOutput
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.results import ResultContract, ResultOrder
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize('expression',['id','name','id + 1','777'])
def test_identity_projection_requires_unchanged_declared_key_values(expression):
    read=_read('items')
    read=replace(read,fields=(*read.fields,CatalogField('name','string',path='name',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,))
    source=build_api_row_source_catalog(catalog).sources[0]
    key=source.candidate_keys[0]
    view=ApiView('items',source.id,{field.path:field.id for field in source.fields if field.path}, {})
    output=QueryOutput('item',identity=EntityKeyProjection(key.entity_kind,key.id,
        (EntityKeyProjectionComponent(key.components[0].id,'key_id'),)))
    def compile():
        return compile_query_answer(question='Which items share the highest score?',
            query=f'SELECT {expression} AS key_id, 1 AS score FROM items',views=(view,),
            output_types={'key_id':'integer','score':'integer'},
            result_contract=ResultContract('rows',('key_id',),(ResultOrder('score',True),),'first_with_ties'),
            catalog=catalog,public_outputs=(output,))
    if expression!='id':
        with pytest.raises(VerificationError,match='identity|key'):
            compile()
        return
    compiled=compile()
    persisted=decode_answer_program(canonical_answer_program_json(compiled.program))
    assert persisted==compiled.program
    class Port:
        def read(self,**kwargs):
            return {'responseStatus':200,'responseBody':[{'id':1,'name':'A'},{'id':2,'name':'B'}]}
    executed=invoke_answer_program(program=persisted,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    keys=[next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]
    assert [key.components[0].value for key in keys]==[1,2]
    assert all(item.entity_kind==key.entity_kind for item in keys)


@pytest.mark.parametrize('second_alias',['a','b'])
def test_composite_identity_components_must_come_from_the_same_record(second_alias):
    from fervis.lookup.answer_program.operations import SqlQuerySpec, SqlRelationInput, SqlColumnBinding, SqlOutputField
    from fervis.lookup.plan_execution.verification.contract_types import RelationContract, RelationEntityKey, RelationEntityKeyComponent
    from fervis.lookup.relational_sql.identity_lineage import sql_identity_keys
    fields={'country':frozenset(),'id':frozenset()}
    contract=RelationContract(fields,(),{},field_types={'country':'string','id':'integer'},entity_keys=(RelationEntityKey('district','compound',
        (RelationEntityKeyComponent('country','country'),RelationEntityKeyComponent('id','id'))),))
    spec=SqlQuerySpec(f'SELECT a.country AS country, {second_alias}.id AS district_id FROM records a CROSS JOIN records b',
        (SqlRelationInput('records','rows',(SqlColumnBinding('country','country'),SqlColumnBinding('id','id'))),),
        (SqlOutputField('country','string'),SqlOutputField('district_id','integer')),
        entity_keys=(EntityKeyProjection('district','compound',
            (EntityKeyProjectionComponent('country','country'),EntityKeyProjectionComponent('id','district_id'))),))
    if second_alias=='b':
        with pytest.raises(VerificationError,match='different record'):
            sql_identity_keys(spec,{'rows':contract})
    else:
        assert len(sql_identity_keys(spec,{'rows':contract}))==1


def test_identity_lineage_uses_sql_identifier_normalization():
    from fervis.lookup.answer_program.operations import SqlQuerySpec, SqlRelationInput, SqlColumnBinding, SqlOutputField
    from fervis.lookup.plan_execution.verification.contract_types import RelationContract, RelationEntityKey, RelationEntityKeyComponent
    from fervis.lookup.relational_sql.identity_lineage import sql_identity_keys
    source=RelationContract({'field.id':frozenset()},(),{},field_types={'field.id':'integer'},entity_keys=(RelationEntityKey('item','pk',
        (RelationEntityKeyComponent('id','field.id'),)),))
    spec=SqlQuerySpec('SELECT "RecordID" AS "OutputID" FROM "Records"',
        (SqlRelationInput('Records','rows',(SqlColumnBinding('RecordID','field.id'),)),),
        (SqlOutputField('OutputID','integer'),),entity_keys=(EntityKeyProjection('item','pk',
            (EntityKeyProjectionComponent('id','OutputID'),)),))
    assert len(sql_identity_keys(spec,{'rows':source}))==1


def test_entity_label_is_current_data_and_does_not_change_identity():
    from fervis.lookup.answer_rendering import render_fact_result,rendered_fact_text
    from fervis.lookup.lineage.results import _execution_lineage_value
    read=replace(_read('items'),fields=(*_read('items').fields,CatalogField('name','string',path='name',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0];key=source.candidate_keys[0]
    view=ApiView('items',source.id,{field.path:field.id for field in source.fields if field.path},{})
    output=QueryOutput('item',identity=EntityKeyProjection(key.entity_kind,key.id,
        (EntityKeyProjectionComponent(key.components[0].id,'id'),)),display_column='name')
    compiled=compile_query_answer(question='Which item?',query='SELECT id,name FROM items',views=(view,),
        output_types={'id':'integer','name':'string'},result_contract=ResultContract('rows',('id','name')),
        catalog=catalog,public_outputs=(output,))
    class Port:
        name='Original name'
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':1,'name':self.name}]}
    port=Port()
    def run():
        executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
        assert executed.issue is None
        return executed.fact_result
    first=run()
    assert 'Original name' in rendered_fact_text(render_fact_result(first))
    kind,value=_execution_lineage_value(first,'result_1')
    assert value['label']=='Original name' and value['components']=={'id':1}
    port.name='Updated name'
    second=run()
    assert first.outcome.projected_rows[0].values==second.outcome.projected_rows[0].values
    assert 'Updated name' in rendered_fact_text(render_fact_result(second))


@pytest.mark.parametrize('output_type',['integer','string'])
def test_identity_output_cannot_coerce_distinct_string_keys(output_type):
    read=_read('items',value_type='string')
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0];key=source.candidate_keys[0]
    output=QueryOutput('item',identity=EntityKeyProjection(key.entity_kind,key.id,
        (EntityKeyProjectionComponent(key.components[0].id,'id'),)))
    def compile():
        return compile_query_answer(question='Which items?',query='SELECT id FROM items',
            views=(ApiView('items',source.id,{'id':source.fields[0].id},{}),),
            output_types={'id':output_type},result_contract=ResultContract('rows',('id',)),
            catalog=catalog,public_outputs=(output,))
    if output_type=='integer':
        with pytest.raises(VerificationError,match='identity.*type'):
            compile()
        return
    compiled=compile()
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':'001'},{'id':'1'}]}
    executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())).components[0].value for row in executed.fact_result.outcome.projected_rows]==['001','1']


@pytest.mark.parametrize('use',['key_column','name_column','name_argument'])
def test_canonical_compilation_checks_identity_parameters_without_the_model_parser(use):
    from fervis.lookup.answer_program.values import FactValue
    from fervis.lookup.canonical_data import EntityKeyValue,EntityKeyComponentValue
    from fervis.lookup.grounding.semantic import CanonicalInputValue
    from fervis.lookup.relational_sql.parameters import query_parameter_menu
    from fervis.lookup.answer_program.operations import SqlNamedInput
    from fervis.lookup.relation_catalog import CatalogParam,ParamSource
    from fervis.lookup.question_contract import InputTerm,InputDenotation
    from fervis.lookup.question_contract.model import InputDenotationKind
    from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,TextType
    from fervis.lookup.relational_sql.execution import QueryValidationError
    from fervis.lookup.answer_program.values import BindingSet
    read=_read('sites',params=(CatalogParam('sites.query.name','name',ParamSource.QUERY,'string'),))
    read=replace(read,fields=(*read.fields,CatalogField('sites.name','string',path='name',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0];key=source.candidate_keys[0]
    def identity(kind):
        return FactValue.identity(id='site',known_input_id='i1',key=EntityKeyValue(kind,key.id,(EntityKeyComponentValue('id',1),)),
            display_value='Default',matched_field_ref='sites.name',matched_field_path='name',matched_value='Default',
            proof_refs=('current_lookup',),source_refs=(read.id,))
    value=identity(key.entity_kind)
    menu=query_parameter_menu((CanonicalInputValue('site','i1',('fact_1:sql_input:i1',),value,value.proof_refs),))
    expression=menu.expressions['p1_1']
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'The site named Default.')
    def compile():
        return compile_query_answer(question=origin.meaning,
            query='SELECT COUNT(*) AS total FROM sites'+('' if use=='name_argument' else f' WHERE {"id" if use=="key_column" else "name"}=$key'),
            views=(ApiView('sites',source.id,{f.path:f.id for f in source.fields if f.path},
                {'sites.query.name':expression} if use=='name_argument' else {}),),
            output_types={'total':'integer'},result_contract=ResultContract('scalar'),catalog=catalog,
            query_parameters=() if use=='name_argument' else (SqlNamedInput('key',expression),),
            parameters=menu.program_inputs.parameters,bindings=menu.program_inputs.bindings,
            inputs=(InputTerm('i1',origin,'Default',TextType()),),
            input_denotations=(InputDenotation('d1','i1','named site','The operand identifies one site.',key.entity_kind,InputDenotationKind.IDENTITY_REFERENCE),),
            expected_input_refs=('i1',))
    if use!='key_column':
        with pytest.raises((QueryValidationError, VerificationError),match='identity|type'):compile()
        return
    compiled=compile()
    program=decode_answer_program(canonical_answer_program_json(compiled.program))
    class Port:
        calls=0
        def read(self,**kwargs):
            self.calls+=1
            return {'responseStatus':200,'responseBody':[{'id':1,'name':'Default'}]}
    port=Port()
    bad_bindings=BindingSet.from_bindings(tuple(replace(binding,value=identity('another_entity')) for binding in compiled.bindings.bindings))
    with pytest.raises(QueryValidationError,match='identity'):
        invoke_answer_program(program=program,bindings=bad_bindings,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert port.calls==0
    answer=invoke_answer_program(program=program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert answer.issue is None and port.calls==1
    assert next(iter(answer.fact_result.outcome.projected_rows[0].values.values()))==1


@pytest.mark.parametrize(('country_field','id_field','valid'),[('from_country','from_id',True),('to_country','to_id',True),('from_country','to_id',False)])
def test_composite_output_preserves_one_reference_tuple_within_a_source_row(country_field,id_field,valid):
    from fervis.lookup.answer_program.operations import SqlQuerySpec,SqlRelationInput,SqlColumnBinding,SqlOutputField
    from fervis.lookup.plan_execution.verification.contract_types import RelationContract,RelationEntityKey,RelationEntityKeyComponent
    from fervis.lookup.relational_sql.identity_lineage import sql_identity_keys
    fields={'from_country':'string','from_id':'integer','to_country':'string','to_id':'integer'}
    source=RelationContract({f:frozenset() for f in fields},(),{},field_types=fields,
        entity_keys=tuple(RelationEntityKey('district','pk',(
            RelationEntityKeyComponent('country',prefix+'_country'),RelationEntityKeyComponent('id',prefix+'_id')))
            for prefix in ('from','to')))
    spec=SqlQuerySpec(f'SELECT {country_field} AS country,{id_field} AS district_id FROM transfers',
        (SqlRelationInput('transfers','rows',tuple(SqlColumnBinding(f,f) for f in fields)),),
        (SqlOutputField('country','string'),SqlOutputField('district_id','integer')),
        entity_keys=(EntityKeyProjection('district','pk',(
            EntityKeyProjectionComponent('country','country'),EntityKeyProjectionComponent('id','district_id'))),))
    if valid:
        assert len(sql_identity_keys(spec,{'rows':source}))==1
    else:
        with pytest.raises(VerificationError,match='identity|key|tuple'):
            sql_identity_keys(spec,{'rows':source})


@pytest.mark.parametrize('role',['identity','related_entity'])
def test_question_identity_roles_produce_a_required_sql_identity_output(role):
    from types import SimpleNamespace
    from jsonschema import validate
    from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt,parse_query_answer
    from fervis.lookup.relational_sql.catalog import build_query_view_catalog
    catalog=RelationCatalog(reads=(_read('locations'),))
    views=build_query_view_catalog(catalog)
    view=views.views[0]
    meaning=SimpleNamespace(output_kinds=(role,),output_origins=('work location',),ordering_origins=(),
        result_kind='qualifying_instances',selection_kind='all_results',selection_limit_input_ref=None)
    prompt=QueryAnswerPrompt(question='Where were the shifts worked?',meaning=meaning,tables=views.tables,parameters={})
    payload={'query':f'SELECT id FROM "{view.name}"','mode':'rows',
        'columns':[{'name':'id','value_type':'integer'}],
        'outputs':[{'kind':'identity','authority':'locations/primary(id)','components':{'id':'id'},'label':'work location','display_column':None}],
        'ordering':[],'api_bindings':[],'interpretations':[]}
    payload = query_payload(**payload)
    validate(payload,prompt._schema())
    authored=parse_query_answer(payload,table_names=set(views.tables),parameter_names=set(),meaning=meaning,tables=views.tables)
    assert authored.outputs[0].identity.entity_kind=='locations'


@pytest.mark.parametrize('operator',['UNION','UNION ALL'])
@pytest.mark.parametrize('wrapped',[False,True])
@pytest.mark.parametrize('valid',[False,True])
def test_each_union_branch_preserves_a_complete_observed_identity(operator,wrapped,valid):
    from fervis.lookup.relation_catalog import EntityReference,EntityReferenceComponent
    from fervis.lookup.relational_sql.catalog import build_query_view_catalog
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from fervis.lookup.relational_sql.execution import QueryValidationError
    fields={'from_country':'string','from_id':'integer','to_country':'string','to_id':'integer'}
    read=replace(_read('transfers'),fields=tuple(CatalogField(name,kind,path=name,row_path_id='root') for name,kind in fields.items()),
        candidate_keys=(),entity_references=tuple(EntityReference(prefix,'district','pk',(
            EntityReferenceComponent('country',prefix+'_country'),EntityReferenceComponent('id',prefix+'_id')))
            for prefix in ('from','to')))
    catalog=RelationCatalog(reads=(read,))
    views=build_query_view_catalog(catalog);view=views.views[0]
    query=f'''SELECT from_country AS country, {"from_id" if valid else "to_id"} AS id FROM "{view.name}"
        {operator} SELECT to_country, {"to_id" if valid else "from_id"} FROM "{view.name}"'''
    if wrapped:query=f'WITH districts AS ({query}) SELECT country,id FROM districts'
    projection=EntityKeyProjection('district','pk',(EntityKeyProjectionComponent('country','country'),EntityKeyProjectionComponent('id','id')))
    payload={'query':query,'mode':'rows','columns':[{'name':'country','value_type':'string'},{'name':'id','value_type':'integer'}],
        'outputs':[{'kind':'identity','authority':'district/pk(country,id)','components':{'country':'country','id':'id'},'label':'district','display_column':None}],
        'ordering':[],'api_bindings':[],'interpretations':[]}
    payload = query_payload(**payload)
    def author():return parse_query_answer(payload,table_names=set(views.tables),parameter_names=set(),tables=views.tables)
    def compile():return compile_query_answer(question='Which districts?',query=query,views=views.views,catalog=catalog,
        output_types={'country':'string','id':'integer'},result_contract=ResultContract('rows',('country','id')),
        public_outputs=(QueryOutput('district',identity=projection),))
    if not valid:
        with pytest.raises(QueryValidationError,match='identity|key|tuple'):author()
        with pytest.raises(VerificationError,match='identity|key|tuple'):compile()
        return
    author()
    compiled=compile()
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'from_country':'KE','from_id':1,'to_country':'UG','to_id':2}]}
    result=invoke_answer_program(program=decode_answer_program(canonical_answer_program_json(compiled.program)),bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert result.issue is None
    keys=[next(iter(row.values.values())) for row in result.fact_result.outcome.projected_rows]
    assert {tuple(key.component_values().values()) for key in keys}=={('KE',1),('UG',2)}


def test_unused_empty_response_view_does_not_invalidate_identity_lineage():
    from fervis.lookup.relational_sql.outputs import _verify_authored_identity_outputs
    tables = {
        'items': {'columns': {'id': {'type': 'integer'}},
                  'candidate_keys': [{'entity_kind': 'item', 'key_id': 'pk',
                                      'components': {'id': 'id'}, 'context_columns': []}]},
        'envelope': {'columns': {}, 'candidate_keys': []},
    }
    output = QueryOutput('item', identity=EntityKeyProjection('item', 'pk',
        (EntityKeyProjectionComponent('id', 'key_id'),)))
    _verify_authored_identity_outputs('SELECT "id" AS key_id FROM items',
        {'key_id': 'integer'}, tables, (output,))


@pytest.mark.parametrize('display', [None, '', 'Current label'])
def test_identity_text_fallback_is_readable_and_preserves_typed_rows(display):
    from uuid import UUID
    from fervis.lookup.answer_rendering import render_fact_result, rendered_fact_text
    read = _read('items', value_type='uuid')
    read = replace(read, fields=(*read.fields, CatalogField('name', 'string', path='name', row_path_id='root')))
    catalog = RelationCatalog(reads=(read,))
    source = build_api_row_source_catalog(catalog).sources[0]
    key = source.candidate_keys[0]
    view = ApiView('items', source.id, {field.path: field.id for field in source.fields if field.path}, {})
    output = QueryOutput('item', identity=EntityKeyProjection(key.entity_kind, key.id,
        (EntityKeyProjectionComponent(key.components[0].id, 'id'),)), display_column='name')
    compiled = compile_query_answer(question='Which items?', query='SELECT id,name FROM items', views=(view,),
        output_types={'id': 'uuid', 'name': 'string'}, result_contract=ResultContract('rows', ('id', 'name')),
        catalog=catalog, public_outputs=(output,))
    ids = ['00000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002']
    class Port:
        def read(self, **kwargs):
            return {'responseStatus': 200, 'responseBody': [{'id': value, 'name': display} for value in ids]}
    result = invoke_answer_program(program=compiled.program, bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog), ports=RuntimePorts(Port(), LookupMemory()))
    assert result.issue is None
    outcome = result.fact_result.outcome
    assert [row.values['result_1'].components[0].value for row in outcome.projected_rows] == list(map(UUID, ids))
    text = rendered_fact_text(render_fact_result(result.fact_result))
    assert text.splitlines() == ['item: '+(display or value) for value in ids]


@pytest.mark.parametrize("display_type", ["datetime", "date", "integer", "number", "uuid", "boolean", "string"])
def test_identity_display_type_is_checked_inside_authoring_correction(display_type):
    from fervis.lookup.relational_sql.outputs import parse_query_outputs
    from fervis.lookup.relational_sql.execution import QueryValidationError
    tables = {"records": {"columns": {"id": {"type": "integer"}, "label": {"type": display_type}},
        "candidate_keys": [{"entity_kind": "record", "key_id": "pk", "components": {"id": "id"}}]}}
    arguments = dict(payload=[{"kind": "identity", "authority": "record/pk(id)",
        "components": {"id": "id"}, "label": "record", "display_column": "label"}],
        columns={"id": "integer", "label": display_type}, tables=tables, query="SELECT id, label FROM records")
    if display_type == "string":
        assert parse_query_outputs(**arguments)[0].display_column == "label"
    else:
        with pytest.raises(QueryValidationError, match="display field must be textual"):
            parse_query_outputs(**arguments)
