from dataclasses import replace
import pytest
from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.relations import Relation,RelationSource,SourceKind,RelationField,FieldBindingRole,EndpointParamBinding
from fervis.lookup.answer_program.operations import Operation,SqlQuerySpec,SqlRelationInput,SqlColumnBinding,SqlOutputField
from fervis.lookup.answer_program.result_projection import EntityKeyProjection,EntityKeyProjectionComponent
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.source_reads.access_execution import execute_access_program
from fervis.lookup.relation_catalog import RelationCatalog,CatalogField,CatalogParam,ParamSource,EntityKeyComponentTarget
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.identity_types import IdentityExecutionFailureReason
from fervis.lookup.outcomes.errors import ExecutionIssueKind
from tests.lookup.relational_engine.test_dependent_reads import _read


def reference_program():
    sites=replace(_read('sites'),fields=(*_read('sites').fields,
        CatalogField('sites.default','boolean',path='is_default',row_path_id='root')))
    observations=_read('observations',params=(CatalogParam('site_id','site_id',ParamSource.PATH,'integer',
        required=True,entity_target=EntityKeyComponentTarget('sites','primary','id')),))
    catalog=RelationCatalog(reads=(sites,observations))
    sources={s.read_id:s for s in build_api_row_source_catalog(catalog).sources}
    fields={f.path:f.id for f in sources['sites'].fields if f.path}
    root=Relation('sites',RelationSource(SourceKind.API_READ,read_id='sites',row_source_id=sources['sites'].id),
        tuple(RelationField(field,(FieldBindingRole.OUTPUT,)) for field in fields.values()))
    child=Relation('observations',RelationSource(SourceKind.API_READ,read_id='observations',row_source_id=sources['observations'].id,
        argument_relation_id='selected_site',param_bindings=(EndpointParamBinding('site_id',FieldRef('id')),)),())
    reference=Operation('reference',SqlQuerySpec('SELECT id FROM sites WHERE is_default',
        (SqlRelationInput('sites','sites',tuple(SqlColumnBinding(path,field) for path,field in fields.items())),),
        (SqlOutputField('id','integer'),),scalar=True,
        entity_keys=(EntityKeyProjection('sites','primary',(EntityKeyProjectionComponent('id','id'),)),),
        reference_input_ref='default_site'),output_relation='selected_site')
    answer=Operation('answer',SqlQuerySpec('SELECT COUNT(*) AS total FROM observations',
        (SqlRelationInput('observations','observations',()),),(SqlOutputField('total','integer'),),scalar=True),output_relation='answer_rows')
    return RelationProgram(relations=(root,child),operations=(reference,answer)),catalog


class Port:
    def __init__(self,rows):self.rows=rows;self.calls=[]
    def read(self,*,endpoint_name,args):
        self.calls.append((endpoint_name,args))
        rows=self.rows if endpoint_name=='sites' else [{'id':i} for i in range(args['site_id'])]
        return {'responseStatus':200,'responseBody':rows}


def test_reference_is_recomputed_before_dependent_requests_on_every_execution():
    program,catalog=reference_program()
    for selected in (1,2):
        port=Port([{'id':i,'is_default':i==selected} for i in (1,2,3)])
        result=execute_access_program(program,catalog=catalog,read_session=ApiReadSession(port))
        assert result.engine_output.issue is None
        assert result.engine_output.relation('answer_rows').rows==({'total':selected},)
        assert port.calls==[('sites',{}),('observations',{'site_id':selected})]


@pytest.mark.parametrize(('rows','reason'),[
    ([{'id':1,'is_default':False}],IdentityExecutionFailureReason.NOT_FOUND),
    ([{'id':1,'is_default':True},{'id':2,'is_default':True}],IdentityExecutionFailureReason.AMBIGUOUS_RESULT),
    ([{'id':1,'is_default':True},{'id':None,'is_default':True}],IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT),
])
def test_unresolved_reference_prevents_all_dependent_requests(rows,reason):
    program,catalog=reference_program();port=Port(rows)
    result=execute_access_program(program,catalog=catalog,read_session=ApiReadSession(port))
    issue=result.engine_output.issue
    assert issue.kind is ExecutionIssueKind.REFERENCE_RESOLUTION
    assert issue.reference.input_ref=='default_site'
    assert issue.reference.reason is reason
    assert port.calls==[('sites',{})]
    if reason is IdentityExecutionFailureReason.AMBIGUOUS_RESULT:
        assert [key.component_values() for key in issue.reference.candidates]==[{'id':1},{'id':2}]


@pytest.mark.parametrize('dependent_api',[False,True])
def test_reference_subplan_is_part_of_the_persisted_answer_program(dependent_api):
    from fervis.lookup.relational_sql.compiler import compile_query_answer
    from fervis.lookup.relational_sql.acquisition import ApiView,RelationView
    from fervis.lookup.relational_sql.results import ResultContract
    from fervis.lookup.contract_codec import canonical_answer_program_json,decode_answer_program
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    base,catalog=reference_program()
    prelude=RelationProgram(relations=(base.relations[0],),operations=(base.operations[0],))
    source=next(s for s in build_api_row_source_catalog(catalog).sources if s.read_id=='observations')
    views=(ApiView('observations',source.id,{}, {'site_id':FieldRef('id')},argument_relation_id='selected_site'),) if dependent_api else ()
    compiled=compile_query_answer(question='Evaluate the current selected site.',
        query='SELECT COUNT(*) AS value FROM observations' if dependent_api else 'SELECT id AS value FROM selected',
        views=views,relation_views=() if dependent_api else (RelationView('selected','selected_site',{'id':'id'}),),
        prerequisites=prelude,catalog=catalog,output_types={'value':'integer'},result_contract=ResultContract('scalar'))
    program=decode_answer_program(canonical_answer_program_json(compiled.program))
    assert any(op.spec.reference_input_ref=='default_site' for op in program.operations if isinstance(op.spec,SqlQuerySpec))
    for selected in (1,2):
        port=Port([{'id':i,'is_default':i==selected} for i in (1,2,3)])
        execution=invoke_answer_program(program=program,bindings=compiled.bindings,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
        assert execution.issue is None
        assert next(iter(execution.fact_result.outcome.projected_rows[0].values.values()))==selected
        assert port.calls==[('sites',{}),*((('observations',{'site_id':selected}),) if dependent_api else ())]


def test_compiled_reference_retains_original_input_provenance_in_the_final_query():
    from fervis.lookup.relational_sql.compiler import compile_query_answer
    from fervis.lookup.relational_sql.reference_compilation import compile_reference_result
    from fervis.lookup.relational_sql.acquisition import ApiView
    from fervis.lookup.relational_sql.outputs import QueryOutput
    from fervis.lookup.relational_sql.results import ResultContract
    from fervis.lookup.relational_sql.parameters import query_parameter_menu
    from fervis.lookup.grounding.semantic import CanonicalInputValue
    from fervis.lookup.answer_program.values import FactValue,LiteralType
    from fervis.lookup.answer_program.operations import SqlNamedInput
    from fervis.lookup.question_contract import InputTerm,InputDenotation
    from fervis.lookup.question_contract.model import InputDenotationKind
    from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,TextType
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    read=replace(_read('sites'),fields=(*_read('sites').fields,CatalogField('sites.name','string',path='name',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0]
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'The site named Cedar.')
    value=FactValue.literal(id='name',known_input_id='i1',literal_type=LiteralType.STRING,value='Cedar',proof_refs=('question_input:i1',))
    menu=query_parameter_menu((CanonicalInputValue('name','i1',('fact_1:sql_input:i1',),value,value.proof_refs),))
    inputs=(InputTerm('i1',origin,'Cedar',TextType()),)
    denotations=(InputDenotation('d1','i1','site name','The supplied name identifies the site.','sites',InputDenotationKind.IDENTITY_REFERENCE),)
    selected=compile_query_answer(question=origin.meaning,query='SELECT id AS site_key FROM sites WHERE name=$name',
        views=(ApiView('sites',source.id,{f.path:f.id for f in source.fields if f.path},{}),),catalog=catalog,
        output_types={'site_key':'integer'},result_contract=ResultContract('rows',('site_key',)),
        query_parameters=(SqlNamedInput('name',menu.expressions['p1_1']),),parameters=menu.program_inputs.parameters,
        bindings=menu.program_inputs.bindings,inputs=inputs,input_denotations=denotations,
        public_outputs=(QueryOutput('site',identity=EntityKeyProjection('sites','primary',(EntityKeyProjectionComponent('id','site_key'),))),),
        namespace='lookup.', lookup_input_ref='i1')
    reference=compile_reference_result(selected,input_ref='i1',output_types={'site_key':'integer'})
    assert reference.input_refs==('i1',)
    final=compile_query_answer(question='Return the identified site ID.',query='SELECT id AS value FROM reference_i1',
        views=(),relation_views=(reference.view,),prerequisites=reference.program,catalog=catalog,
        output_types={'value':'integer'},result_contract=ResultContract('scalar'),bindings=reference.bindings,
        inputs=inputs,input_denotations=denotations,expected_input_refs=('i1',))
    assert final.question_contract.requested_facts[0].input_refs==('i1',)
    class NamedPort:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':2,'name':'Cedar'},{'id':3,'name':'Lake'}]}
    execution=invoke_answer_program(program=final.program,bindings=final.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(NamedPort(),LookupMemory()))
    assert execution.issue is None
    assert next(iter(execution.fact_result.outcome.projected_rows[0].values.values()))==2


def test_reference_runtime_failure_retains_its_collection_member():
    program,catalog=reference_program()
    reference=program.operations[0]
    program=replace(program,operations=(replace(reference,spec=replace(reference.spec,reference_operand='Beta')),
        *program.operations[1:]))
    port=Port([{'id':1,'is_default':False}])
    result=execute_access_program(program,catalog=catalog,read_session=ApiReadSession(port))
    assert result.engine_output.issue.reference.operand=='Beta'
    assert result.engine_output.issue.reference.reason is IdentityExecutionFailureReason.NOT_FOUND
    assert port.calls==[('sites',{})]


def test_reference_extrema_preserve_tied_candidates_before_uniqueness_guard():
    from fervis.lookup.relational_sql.reference_matching import reject_reference_truncation
    program, catalog = reference_program()
    query = 'SELECT id FROM sites WHERE is_default = (SELECT MAX(is_default) FROM sites)'
    reject_reference_truncation(query)
    reference = replace(program.operations[0], spec=replace(program.operations[0].spec, query=query))
    program = replace(program, operations=(reference, *program.operations[1:]))
    port = Port([{'id':1,'is_default':True}, {'id':2,'is_default':True}, {'id':3,'is_default':False}])
    result = execute_access_program(program, catalog=catalog, read_session=ApiReadSession(port))
    assert result.engine_output.issue.reference.reason is IdentityExecutionFailureReason.AMBIGUOUS_RESULT
    assert len(result.engine_output.issue.reference.candidates) == 2
    assert port.calls == [('sites', {})]


@pytest.mark.parametrize('nested', [False, True])
def test_persisted_reference_truncation_is_rejected_before_reads(nested):
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
    from fervis.lookup.answer_program.model import AnswerProgram
    from fervis.lookup.plan_execution.errors import VerificationError
    program, catalog = reference_program()
    reference = program.operations[0]
    if nested:
        candidate = replace(reference, id='candidate', output_relation='candidate_rows',
            spec=replace(reference.spec, query='SELECT id FROM sites WHERE is_default LIMIT 1',
                         scalar=False, reference_input_ref=''))
        reference = replace(reference, spec=replace(reference.spec, query='SELECT id FROM candidates',
            inputs=(SqlRelationInput('candidates','candidate_rows',(SqlColumnBinding('id','id'),)),)))
        program = replace(program, operations=(candidate, reference, *program.operations[1:]))
    else:
        reference = replace(reference, spec=replace(reference.spec, query=reference.spec.query+' LIMIT 1'))
        program = replace(program, operations=(reference, *program.operations[1:]))
    program = decode_answer_program(canonical_answer_program_json(AnswerProgram(
        parameters=program.parameters, relations=program.relations, operations=program.operations)))
    port = Port([{'id':1,'is_default':True}, {'id':2,'is_default':True}])
    with pytest.raises(VerificationError, match='truncated'):
        execute_access_program(program, catalog=catalog, read_session=ApiReadSession(port))
    assert port.calls == []
