from dataclasses import replace
import pytest
from jsonschema import validate
from fervis.lookup.relational_sql.reference_planning import ReferenceMeaning,ReferenceQueryPrompt,parse_reference_query,compile_reference_plan
from fervis.lookup.relational_sql.parameters import query_parameter_menu,with_catalog_choices
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.read_eligibility import SemanticReadEligibilityResult,ReadRequirementAssessment,SemanticReadDecision
from fervis.lookup.relation_catalog import RelationCatalog,CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_model import ReadAccessCatalog
from fervis.lookup.answer_program.values import FactValue,LiteralType
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.question_contract import InputTerm,InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,TextType
from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize(('case','text','expected'),[
    ('name','Default',1),('role','the primary record',2),('split','Ada Lovelace',2),
    ('unfiltered_name','Default',1),('unfiltered_missing','Default',None),
])
def test_reference_authoring_supports_observed_expressions_without_backend_name_rules(case,text,expected):
    from fervis.lookup.relation_catalog import CatalogParam, ParamSource
    read=replace(_read('records', params=(CatalogParam('name', 'name', ParamSource.QUERY, 'string'),)),fields=(*_read('records').fields,
        CatalogField('name','string',path='name',row_path_id='root'),
        CatalogField('given','string',path='given',row_path_id='root'),
        CatalogField('family','string',path='family',row_path_id='root'),
        CatalogField('primary','boolean',path='is_primary',row_path_id='root',metadata={'description':'True marks the configured primary record.'})))
    catalog=RelationCatalog(reads=(read,));sources=build_api_row_source_catalog(catalog);source=sources.sources[0]
    eligibility=SemanticReadEligibilityResult((ReadRequirementAssessment('fact_1',read.id,(source.id,),read.id,
        tuple(f.field_ref for f in source.fields),'These records expose the reference attributes.',SemanticReadDecision.RETAIN),),())
    available=build_available_source_catalog(sources,read_eligibility=eligibility)
    view_catalog=build_query_view_catalog(catalog);view=view_catalog.views[0]
    value=FactValue.literal(id='reference_text',known_input_id='i1',literal_type=LiteralType.STRING,value=text,proof_refs=('question_input:i1',))
    menu=with_catalog_choices(query_parameter_menu((CanonicalInputValue(value.id,'i1',('fact_1:sql_input:i1',),value,value.proof_refs),)),
        source_catalog=available,source_refs={source.id})
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,text)
    meaning=ReferenceMeaning('fact_1','i1','record',f'Identify {text}.',(origin,),('i1',),reference_text=text,reference_kind="description" if case == "role" else "literal")
    prompt=ReferenceQueryPrompt(question=f'Return the ID for {text}.',meaning=meaning,tables=view_catalog.tables,parameters=menu.descriptions)
    interpretations=[]
    if case=='role':
        choice=next(name for name,desc in menu.descriptions.items() if desc.get('kind')=='catalog_choice' and desc['value']=='true')
        predicate=f'is_primary=${choice}'
        interpretations=[{'input':'p1_1','choice':choice,'basis':'The API identifies primary records using this flag.'}]
    else:predicate=("CONCAT(given, ' ', family)" if case=='split' else 'name')+'=$p1_1'
    if case == 'unfiltered_name': predicate = 'TRUE'
    if case == 'unfiltered_missing': predicate = '$p1_1 = $p1_1'
    matching = "CONCAT(given, ' ', family)" if case == 'split' else 'name'
    extra = f', {matching} AS matched_name' if case != 'role' else ''
    payload={'query':f'SELECT id AS record_id{extra} FROM "{view.name}" WHERE {predicate}',
        'mode':'rows','columns':[{'name':'record_id','value_type':'integer'}, *([{'name':'matched_name','value_type':'string'}] if case != 'role' else [])],
        'outputs':[{'kind':'identity','authority':view.name+':key:0','components':{'id':'record_id'},'label':'record','display_column':None}],
        'ordering':[],'request_arguments':[],'interpretations':[],
        'reference_binding':{'kind':'description','basis':'The primary property defines the role.'} if case=='role' else {'kind':'literal','match_column':'matched_name'}}
    if case != "role": payload["query"] = payload["query"].split(" WHERE ")[0]
    validate(payload,prompt._schema())
    authored=parse_reference_query(payload,prompt=prompt,menu=menu)
    if case == 'role':
        from copy import deepcopy
        from jsonschema import ValidationError
        from fervis.lookup.relational_sql.execution import QueryValidationError
        assert prompt.parameters['p1_1']['kind'] == 'definition'
        bad = deepcopy(payload)
        bad['request_arguments'] = [{'view': view.name, 'parameter_ref': 'name', 'binding': 'p1_1'}]
        with pytest.raises(ValidationError):
            validate(bad, prompt._schema())
        with pytest.raises(QueryValidationError):
            parse_reference_query(bad, prompt=prompt, menu=menu)
    inputs=(InputTerm('i1',origin,text,TextType()),)
    denotations=(InputDenotation('d1','i1','record reference','The input denotes a record.','records',InputDenotationKind.IDENTITY_REFERENCE,reference_descriptions=(text,) if case == 'role' else ()),)
    reference=compile_reference_plan(authored,meaning=meaning,menu=menu,views=view_catalog.views,catalog=catalog,
        inputs=inputs,input_denotations=denotations,access=ReadAccessCatalog())
    final=compile_query_answer(question='Return the referenced ID.',query=f'SELECT id AS value FROM "{reference.view.name}"',
        views=(),relation_views=(reference.view,),prerequisites=reference.program,catalog=catalog,
        bindings=reference.bindings,inputs=inputs,input_denotations=denotations,expected_input_refs=('i1',),
        output_types={'value':'integer'},result_contract=ResultContract('scalar'))
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':([
            {'id':1,'name':'Default','given':'Ada','family':'Byron','is_primary':False},
            {'id':2,'name':'Cedar','given':'Ada','family':'Lovelace','is_primary':True}][1:] if case == 'unfiltered_missing' else [
            {'id':1,'name':'Default','given':'Ada','family':'Byron','is_primary':False},
            {'id':2,'name':'Cedar','given':'Ada','family':'Lovelace','is_primary':True}])}
    executed=invoke_answer_program(program=final.program,bindings=final.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    if expected is None:
        from fervis.lookup.identity_types import IdentityExecutionFailureReason
        assert executed.fact_result is None
        assert executed.issue.reference.reason is IdentityExecutionFailureReason.NOT_FOUND
        return
    assert executed.issue is None
    assert next(iter(executed.fact_result.outcome.projected_rows[0].values.values()))==expected


@pytest.mark.parametrize('match_expression', ['$p1_1', "'Alpha'", "CASE WHEN $p1_1 = $p1_1 THEN 'Alpha' END", "COALESCE(name, 'Alpha')", "CASE WHEN name IS NULL THEN 'Alpha' ELSE name END", "CONCAT(name, 'Alpha')"])
def test_reference_output_contract_does_not_request_unused_presentation_fields(match_expression):
    from jsonschema import ValidationError
    from fervis.lookup.relational_sql.execution import QueryValidationError
    catalog=RelationCatalog(reads=(replace(_read('records'),fields=(*_read('records').fields,CatalogField('name','string',path='name',row_path_id='root'))),))
    views=build_query_view_catalog(catalog)
    view=views.views[0]
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Identify the record.')
    meaning=ReferenceMeaning('fact_1','i1','records','Identify the record.',(origin,),('i1',),reference_text='Alpha')
    value=FactValue.literal(id='name',known_input_id='i1',literal_type=LiteralType.STRING,value='Alpha',proof_refs=('question_input:i1',))
    menu=query_parameter_menu((CanonicalInputValue(value.id,'i1',('fact_1:sql_input:i1',),value,value.proof_refs),))
    prompt=ReferenceQueryPrompt(question='Identify the record.',meaning=meaning,tables=views.tables,parameters=menu.descriptions)
    payload={'query':f'SELECT id, {match_expression} AS display_name FROM "{view.name}"','mode':'rows',
        'columns':[{'name':'id','value_type':'integer'},{'name':'display_name','value_type':'string'}],
        'outputs':[{'kind':'identity','authority':view.name+':key:0','components':{'id':'id'},'label':'record','display_column':'display_name'}],
        'ordering':[],'request_arguments':[],'interpretations':[],'reference_binding':{'kind':'literal','match_column':'display_name'}}
    with pytest.raises(ValidationError):validate(payload,prompt._schema())
    with pytest.raises(QueryValidationError,match='display projection'):
        parse_reference_query(payload,prompt=prompt,menu=menu)
    payload['outputs'][0]['display_column']=None
    validate(payload,prompt._schema())
    with pytest.raises(QueryValidationError, match='literal|observed'):
        parse_reference_query(payload,prompt=prompt,menu=menu)


@pytest.mark.parametrize('query', [
    'SELECT id FROM records LIMIT 1',
    'SELECT id FROM (SELECT id FROM records LIMIT 1) AS subset',
    'SELECT id FROM records OFFSET 1',
    'SELECT id FROM records QUALIFY ROW_NUMBER() OVER (ORDER BY id)=1',
    'SELECT DISTINCT ON (name) id FROM records',
])
def test_reference_candidate_truncation_is_rejected(query):
    from fervis.lookup.relational_sql.reference_matching import reject_reference_truncation
    from fervis.lookup.relational_sql.execution import QueryValidationError
    with pytest.raises(QueryValidationError, match='truncated'):
        reject_reference_truncation(query)


def test_literal_formatting_cannot_turn_missing_components_into_a_name():
    from fervis.lookup.relational_sql.reference_matching import literal_match_query
    from fervis.lookup.relational_sql.execution import execute_query, SqlTable
    query = literal_match_query("SELECT id, CONCAT(name, '-') AS matched_name FROM records",
        column='matched_name', parameter='name', tables={'records': {'columns': {'id': {}, 'name': {}}}})
    result = execute_query(query, tables={'records': SqlTable({'id':'INTEGER','name':'TEXT'},
        ({'id':1,'name':None},))}, parameters={'name':'-'})
    assert result.rows == ()


@pytest.mark.parametrize('predicate', ['name = $p1', '$p1 = name', 'r.name = $p1'])
def test_only_equivalent_literal_predicates_are_canonicalized(predicate):
    from fervis.lookup.relational_sql.reference_matching import literal_match_query
    from fervis.lookup.relational_sql.execution import execute_query, SqlTable, QueryValidationError
    tables = {'records': {'columns': {'id': {}, 'name': {}}}}
    query = literal_match_query(f'SELECT id, name AS matched_name FROM records r WHERE {predicate}',
        column='matched_name', parameter='p1', tables=tables)
    result = execute_query(query, tables={'records': SqlTable({'id':'INTEGER','name':'TEXT'},
        ({'id':1,'name':'Default'}, {'id':2,'name':'Default'}))}, parameters={'p1':'Default'})
    assert len(result.rows) == 2
    with pytest.raises(QueryValidationError, match='predicates|observed'):
        literal_match_query('SELECT id, name AS matched_name FROM records WHERE id=1',
            column='matched_name', parameter='p1', tables=tables)
