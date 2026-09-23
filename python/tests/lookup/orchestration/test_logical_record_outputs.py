"""Typed logical outputs can expose observed records without nominal keys."""
from dataclasses import replace
import pytest
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.question_contract.model import RequestedFact,RequestedOutput,SetTerm,FactTerm,Subject,InstanceInterpretation,AllResults,Aggregate,AggregateFunction
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,IdentifierType
from fervis.lookup.orchestration.logical_planning import realize_and_compile_logical_plan
from fervis.lookup.available_sources import snapshot_source_catalog,SourceFieldBinding
from fervis.lookup.relation_catalog import RelationCatalog,CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.fact_compilation.model import FactCompilationResult
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('computed_property', [False, True])
def test_unannotated_records_keep_occurrences_and_selected_properties_on_replay(computed_property,grouped):
    result, catalog = _record_candidates(grouped=grouped)
    from fervis.lookup.contract_codec import canonical_answer_program_json,decode_answer_program
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    program=decode_answer_program(canonical_answer_program_json(result.answer_program))
    output=program.result_projection.relation_outputs[0]
    assert output.entity_key is None
    assert set(output.record_fields)=={'id','label'}
    if computed_property:
        from fervis.lookup.answer_program.operations import ProjectSpec
        from fervis.lookup.answer_program.expressions import FieldRef
        from fervis.lookup.relational_sql.compiler import _constant
        operation=next(op for op in program.operations if isinstance(op.spec,ProjectSpec) and
            any(item.output_field==output.record_fields['id'] and isinstance(item.expression,FieldRef) for item in op.spec.outputs))
        changed=replace(operation,spec=replace(operation.spec,outputs=tuple(
            replace(item,expression=_constant('fabricated_property',999)) if item.output_field==output.record_fields['id'] else item
            for item in operation.spec.outputs)))
        program=replace(program,operations=tuple(changed if op.id==changed.id else op for op in program.operations))
    reads=[]
    class Port:
        def read(self,**kwargs):
            reads.append(kwargs)
            return {'responseStatus':200,'responseBody':[{'id':1,'label':'A','flag':'invalid'},
                {'id':1,'label':'A','flag':'invalid'}]}
    if computed_property:
        from fervis.lookup.plan_execution.errors import VerificationError
        with pytest.raises(VerificationError,match='observed'):
            invoke_answer_program(program=program,bindings=result.initial_bindings,
                environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
        assert reads==[]
        return
    executed=invoke_answer_program(program=program,bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[
        {'id':1,'label':'A'},{'id':1,'label':'A'}]
    if grouped:
        assert [list(row.values.values())[1] for row in executed.fact_result.outcome.projected_rows]==[1,1]


def _record_candidates(*, grouped=False):
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'entry records')
    fact=RequestedFact('fact_1',origin,(SetTerm('s1',origin),),(),(FactTerm('self','s1',IdentifierType('s1'),origin),),(),
        Subject('s1',InstanceInterpretation.RESOURCE_POPULATION),None,(),
        (RequestedOutput('record','self',origin),),(),AllResults(),())
    if grouped:
        fact=replace(fact,grouping_refs=('self',),expressions=(Aggregate('count',AggregateFunction.COUNT,'s1',None,False,origin),),
            outputs=(*fact.outputs,RequestedOutput('count','count',origin)))
    index=analyze_requested_fact(fact,inputs={},input_denotations={})
    logical=ParsedSemanticQuestionContract('Return observed entries.',QuestionContract((),(fact,)),(index,))
    read=replace(_read('entries'),candidate_keys=(),fields=(*_read('entries').fields,
        CatalogField('entries.label','string',path='label',row_path_id='root'),
        CatalogField('entries.flag','boolean',path='flag',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,))
    source,=build_api_row_source_catalog(catalog).sources
    available=snapshot_source_catalog((source,))
    fields={field.path:SourceFieldBinding(source.id,field).ref for field in source.fields}
    calls=[]
    def turn(purpose,prompt,parse):
        calls.append(type(prompt).__name__)
        from jsonschema import validate
        original_parse= parse
        def parse(payload):
            validate(payload,prompt.tool_contract().tool_specs[0].input_schema)
            return original_parse(payload)
        branch=prompt.request.strategy.branches[0].branch_id
        if len(calls)==1:
            return parse({'set_bindings':{'fact_1:set:s1':[{'branch_id':branch,'mapping_basis':'Each entry is an observed record.',
                'rows_ref':source.id,'record_fields':[{'name':name,'field_ref':fields[name]} for name in ('id','label')]}]},
                'fact_bindings':{},'association_bindings':{}})
        return parse({'populations':{'fact_1:set:s1':[{'branch_id':branch,'logical_set_meaning':'entry records',
            'mapping_basis':'All rows are entry records.','population':{'kind':'exact_population'}}]}})
    result=realize_and_compile_logical_plan(logical,sources_by_fact={'fact_1':available},canonical_values=(),turn=turn)
    assert isinstance(result,FactCompilationResult)
    return result, catalog


@pytest.mark.parametrize('distinct', [False, True])
def test_observed_row_identity_can_be_counted_without_becoming_a_nominal_key(distinct):
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'observed entry records')
    fact=RequestedFact('fact_1',origin,(SetTerm('s1',origin),),(),
        (FactTerm('self','s1',IdentifierType('s1'),origin),),
        (Aggregate('n',AggregateFunction.COUNT,'self',None,distinct,origin),),
        Subject('s1',InstanceInterpretation.RAW_DATA_RECORD),None,(),
        (RequestedOutput('count','n',origin),),(),AllResults(),())
    index=analyze_requested_fact(fact,inputs={},input_denotations={})
    logical=ParsedSemanticQuestionContract('Count record occurrences.',QuestionContract((),(fact,)),(index,))
    catalog=RelationCatalog(reads=(replace(_read('entries'),candidate_keys=()),))
    source,=build_api_row_source_catalog(catalog).sources
    available=snapshot_source_catalog((source,))
    turns=[]
    def turn(purpose,prompt,parse):
        turns.append(prompt)
        branch=prompt.request.strategy.branches[0].branch_id
        if len(turns)==1:
            return parse({'set_bindings':{'fact_1:set:s1':[{'branch_id':branch,'mapping_basis':'Each API row is an observed record.',
                'rows_ref':source.id,'record_fields':[]}]},'fact_bindings':{},'association_bindings':{}})
        return parse({'populations':{'fact_1:set:s1':[{'branch_id':branch,'logical_set_meaning':'observed entry records',
            'mapping_basis':'All returned records.','population':{'kind':'exact_population'}}]}})
    result=realize_and_compile_logical_plan(logical,sources_by_fact={'fact_1':available},canonical_values=(),turn=turn)
    assert isinstance(result,FactCompilationResult)
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':1},{'id':1}]}
    executed=invoke_answer_program(program=result.answer_program,bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert next(iter(executed.fact_result.outcome.projected_rows[0].values.values()))==2


@pytest.mark.parametrize('identity_order', [False, True])
def test_record_ordering_uses_observed_properties_not_internal_occurrence_numbers(identity_order):
    from fervis.lookup.question_contract.model import Ordering,OrderingDirection
    from fervis.lookup.semantic_types import TextType
    from fervis.lookup.source_binding import SourceRealizationUnavailable
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'entry records')
    fact=RequestedFact('fact_1',origin,(SetTerm('s1',origin),),(),
        (FactTerm('self','s1',IdentifierType('s1'),origin),FactTerm('label','s1',TextType(),origin)),(),
        Subject('s1',InstanceInterpretation.RESOURCE_POPULATION),None,(),(RequestedOutput('record','self',origin),),
        (Ordering('self' if identity_order else 'label',OrderingDirection.ASCENDING,origin),),AllResults(),())
    index=analyze_requested_fact(fact,inputs={},input_denotations={})
    logical=ParsedSemanticQuestionContract('Order entries.',QuestionContract((),(fact,)),(index,))
    read=replace(_read('entries'),candidate_keys=(),fields=(*_read('entries').fields,
        CatalogField('entries.label','string',path='label',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,))
    source,=build_api_row_source_catalog(catalog).sources
    available=snapshot_source_catalog((source,))
    fields={field.path:SourceFieldBinding(source.id,field).ref for field in source.fields}
    calls=[]
    def turn(purpose,prompt,parse):
        calls.append(prompt)
        branch=prompt.request.strategy.branches[0].branch_id
        if len(calls)==1:
            return parse({'set_bindings':{'fact_1:set:s1':[{'branch_id':branch,'mapping_basis':'Observed entries.',
                'rows_ref':source.id,'record_fields':[{'name':name,'field_ref':fields[name]} for name in ('id','label')]}]},
                'fact_bindings':{'fact_1:fact:label':[{'branch_id':branch,'mapping_basis':'The observed entry label.',
                    'field_ref':fields['label']}]},'association_bindings':{}})
        return parse({'populations':{'fact_1:set:s1':[{'branch_id':branch,'logical_set_meaning':'entry records',
            'mapping_basis':'All returned entries.','population':{'kind':'exact_population'}}]}})
    result=realize_and_compile_logical_plan(logical,sources_by_fact={'fact_1':available},canonical_values=(),turn=turn)
    if identity_order:
        assert isinstance(result,SourceRealizationUnavailable)
        assert calls==[]
        return
    assert isinstance(result,FactCompilationResult)
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':1,'label':'B'},{'id':1,'label':'A'}]}
    executed=invoke_answer_program(program=result.answer_program,bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values()))['label'] for row in executed.fact_result.outcome.projected_rows]==['A','B']


def test_record_properties_cannot_be_borrowed_from_another_eligible_carrier():
    from jsonschema import validate
    from fervis.lookup.orchestration.logical_planning import prepare_logical_realizations
    from fervis.lookup.source_binding.parser import compile_source_realization
    from fervis.lookup.source_binding.schema import build_semantic_source_realization_schema
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'entry records')
    fact=RequestedFact('fact_1',origin,(SetTerm('s1',origin),),(),
        (FactTerm('self','s1',IdentifierType('s1'),origin),),(),
        Subject('s1',InstanceInterpretation.RESOURCE_POPULATION),None,(),
        (RequestedOutput('record','self',origin),),(),AllResults(),())
    index=analyze_requested_fact(fact,inputs={},input_denotations={})
    logical=ParsedSemanticQuestionContract('Return records.',QuestionContract((),(fact,)),(index,))
    catalog=RelationCatalog(reads=tuple(replace(_read(name),candidate_keys=()) for name in ('entries','other')))
    sources=build_api_row_source_catalog(catalog).sources
    source=next(item for item in sources if item.read_id=='entries')
    other=next(item for item in sources if item.read_id=='other')
    available=snapshot_source_catalog(sources)
    request,=prepare_logical_realizations(logical,sources_by_fact={'fact_1':available},canonical_values=())
    payload={'set_bindings':{'fact_1:set:s1':[{'branch_id':request.strategy.branches[0].branch_id,
        'mapping_basis':'Select entries.','rows_ref':source.id,'record_fields':[
            {'name':'id','field_ref':SourceFieldBinding(other.id,other.fields[0]).ref}]}]},
        'fact_bindings':{},'association_bindings':{}}
    validate(payload,build_semantic_source_realization_schema(request))
    with pytest.raises(ValueError,match='selected carrier'):
        compile_source_realization(payload,request=request)


def test_joined_record_output_preserves_properties_and_duplicate_occurrences():
    from fervis.lookup.question_contract.model import AssociationTerm,FactLocalRef
    from fervis.lookup.semantic_types import TextType
    from fervis.lookup.relation_catalog import EntityReference,EntityReferenceComponent
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'entries with parent labels')
    fact=RequestedFact('fact_1',origin,(SetTerm('entry',origin),SetTerm('parent',origin)),
        (AssociationTerm('parent_of','entry','parent',origin),),
        (FactTerm('self','entry',IdentifierType('entry'),origin),FactTerm('name','entry',TextType(),origin),
         FactTerm('label','parent',TextType(),origin)),(),
        Subject('entry',InstanceInterpretation.RESOURCE_POPULATION),None,(),
        tuple(RequestedOutput(name,name,origin) for name in ('self','name','label')),(),AllResults(),())
    index=analyze_requested_fact(fact,inputs={},input_denotations={})
    logical=ParsedSemanticQuestionContract('Return entry records and parent labels.',QuestionContract((),(fact,)),(index,))
    entries=replace(_read('entries'),candidate_keys=(),fields=(*_read('entries').fields,
        CatalogField('entries.name','string',path='name',row_path_id='root'),
        CatalogField('entries.parent_id','integer',path='parent_id',row_path_id='root')),
        entity_references=(EntityReference('parent','parents','primary',(EntityReferenceComponent(target_component_id='id',local_field_ref='entries.parent_id'),)),))
    parents=replace(_read('parents'),fields=(*_read('parents').fields,CatalogField('parents.label','string',path='label',row_path_id='root')))
    catalog=RelationCatalog(reads=(entries,parents))
    sources=build_api_row_source_catalog(catalog).sources
    available=snapshot_source_catalog(sources)
    by_read={source.read_id:source for source in sources}
    fields={(source.read_id,field.path):SourceFieldBinding(source.id,field).ref for source in sources for field in source.fields}
    parent_identity=next(value.identity_ref for value in available.identity_evidence if value.source_ref==by_read['parents'].id)
    turns=[]
    def turn(purpose,prompt,parse):
        turns.append(prompt)
        branch=prompt.request.strategy.branches[0].branch_id
        if len(turns)==1:
            return parse({'set_bindings':{
                'fact_1:set:entry':[{'branch_id':branch,'mapping_basis':'Entry records.','rows_ref':by_read['entries'].id,
                    'record_fields':[{'name':name,'field_ref':fields['entries',name]} for name in ('id','name')]}],
                'fact_1:set:parent':[{'branch_id':branch,'mapping_basis':'Parent records.','rows_ref':parent_identity,'record_fields':[]}]},
                'fact_bindings':{f'fact_1:fact:{name}':[{'branch_id':branch,'mapping_basis':'Observed property.','field_ref':fields[read,name]}]
                    for read,name in [('entries','name'),('parents','label')]},
                'association_bindings':{'fact_1:association:parent_of':[{'branch_id':branch,'mapping_basis':'Declared parent reference.',
                    "field_pairs": [], 'realization_ref':available.relation_evidence[0].evidence_ref,'reference_from_set_ref':None}]}})
        return parse({'populations':{ref:[{'branch_id':branch,'logical_set_meaning':index.term_by_ref[FactLocalRef.from_token(ref)].origin.meaning,
            'mapping_basis':'All source rows.','population':{'kind':'exact_population'}}] for ref in ('fact_1:set:entry','fact_1:set:parent')}})
    result=realize_and_compile_logical_plan(logical,sources_by_fact={'fact_1':available},canonical_values=(),turn=turn)
    assert isinstance(result,FactCompilationResult)
    from fervis.lookup.contract_codec import canonical_answer_program_json,decode_answer_program
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    program=decode_answer_program(canonical_answer_program_json(result.answer_program))
    class Port:
        def read(self,*,endpoint_name,args):
            rows=[{'id':1,'name':'A','parent_id':10}]*2 if endpoint_name=='entries' else [{'id':10,'label':'P'}]
            return {'responseStatus':200,'responseBody':rows}
    executed=invoke_answer_program(program=program,bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [list(row.values.values()) for row in executed.fact_result.outcome.projected_rows]==[
        [{'id':1,'name':'A'},'A','P'],[{'id':1,'name':'A'},'A','P']]
