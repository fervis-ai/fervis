from decimal import Decimal
from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.answer_program.model import AnswerProgram, RelationProgram
from fervis.lookup.answer_program.operations import Operation, SqlQuerySpec, SqlRelationInput, SqlColumnBinding, SqlNamedInput, SqlOutputField
from fervis.lookup.answer_program.relations import Relation, RelationSource, RelationField, SourceKind, FieldBindingRole
from fervis.lookup.answer_program.values import FactValue, ConstantRef, LiteralType
from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_execution import execute_access_program
from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
from tests.lookup.relational_engine.test_dependent_reads import _read


def test_sql_is_a_serializable_operation_in_the_verified_program_kernel():
    catalog = RelationCatalog(reads=(_read('items'),))
    source = build_api_row_source_catalog(catalog).sources[0]
    relation = Relation('rows', RelationSource(SourceKind.API_READ, read_id='items', row_source_id=source.id),
                        (RelationField(source.fields[0].id, (FieldBindingRole.IDENTITY, FieldBindingRole.OUTPUT)),))
    cutoff = FactValue.literal(id='cutoff',literal_type=LiteralType.NUMBER,value='1',proof_refs=('question:cutoff',))
    operation = Operation('query', SqlQuerySpec(
        'SELECT COUNT(*) AS total FROM items WHERE id > :cutoff',
        (SqlRelationInput('items','rows',(SqlColumnBinding('id',source.fields[0].id),)),),
        (SqlOutputField('total','number'),),
        (SqlNamedInput('cutoff',ConstantRef('cutoff','question:v1',cutoff)),), scalar=True),
        output_relation='answer_rows')
    program = RelationProgram(relations=(relation,),operations=(operation,))
    saved = AnswerProgram(relations=program.relations, operations=program.operations)
    assert decode_answer_program(canonical_answer_program_json(saved)) == saved
    class Port:
        def read(self, **kwargs):
            return {'responseStatus':200,'responseBody':[{'id':1},{'id':2},{'id':3}]}
    result = execute_access_program(program, catalog=catalog, read_session=ApiReadSession(Port()))
    output = result.engine_output.relation('answer_rows')
    assert output.rows == ({'total':Decimal('2')},)
    assert {'read:items','question:cutoff'} <= set(output.evidence.proof_refs)


def test_sql_answer_uses_canonical_invocation_projection_and_output_proof():
    from fervis.lookup.answer_program.model import FactFulfillment
    from fervis.lookup.answer_program.operations import ComputeSpec
    from fervis.lookup.answer_program.values import BindingSet, NodeOutputRef
    from fervis.lookup.answer_program.result_projection import ResultProjection, ScalarResultOutput
    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.question_contract import QueryQuestionContract, QueryRequestedFact, QueryRequestedOutput, QueryOperationDeclaration, QuerySourceDeclaration
    from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
    from fervis.lookup.contract_codec import canonical_contract_fingerprint
    from fervis.lookup.answer_rendering import render_fact_result, rendered_fact_text
    catalog = RelationCatalog(reads=(_read('items'),))
    source = build_api_row_source_catalog(catalog).sources[0]
    relation = Relation('rows',RelationSource(SourceKind.API_READ,read_id='items',row_source_id=source.id),
                        (RelationField(source.fields[0].id,(FieldBindingRole.IDENTITY,FieldBindingRole.OUTPUT)),))
    sql = SqlQuerySpec('SELECT COUNT(*) AS total FROM items',
                      (SqlRelationInput('items','rows',(SqlColumnBinding('id',source.fields[0].id),)),),
                      (SqlOutputField('total','number'),),scalar=True)
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, 'How many items are there?')
    operations = (
        Operation('query',sql,output_relation='query_rows'),
        Operation('output',ComputeSpec(NodeOutputRef('query','total'),output_scalar='count')))
    projection = ScalarResultOutput('result_count','count',label='count',role='answer_value')
    fact = QueryRequestedFact('fact_1',origin,
        tuple(QueryOperationDeclaration(item.id,canonical_contract_fingerprint(item)) for item in operations),
        (QueryRequestedOutput.from_projection('r1',origin,projection,'number'),),
        sources=(QuerySourceDeclaration(relation.id,canonical_contract_fingerprint(relation)),))
    question = QueryQuestionContract((),(fact,))
    program = AnswerProgram(relations=(relation,), operations=operations,
        fulfillment=(FactFulfillment('fact_1','r1','result_count'),),
        result_projection=ResultProjection(scalar_outputs=(projection,)))
    program,bindings = compile_answer_program(program,question_contract=question,catalog=catalog,bindings=BindingSet())
    class Port:
        def read(self, **kwargs):
            return {'responseStatus':200,'responseBody':[{'id':1},{'id':2},{'id':3}]}
    executed = invoke_answer_program(program=program,bindings=bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert rendered_fact_text(render_fact_result(executed.fact_result)) == 'result_count: 3'
    assert executed.proof_node_refs_by_result_output_id['result_count']
    assert any(node.operator == 'sql_query' for node in executed.proof_graph.nodes)


def test_named_property_is_a_scalar_lookup_and_rebinding_reads_fresh_data():
    from dataclasses import replace
    from fervis.lookup.relation_catalog.model import CatalogField
    from fervis.lookup.answer_program.model import FactFulfillment
    from fervis.lookup.answer_program.values import (BindingSet, ParameterBinding, ParameterRef,
        ParameterDeclaration, ParameterRole, ParameterValueType, BindingProvenance, BindingProvenanceKind)
    from fervis.lookup.answer_program.result_projection import ResultProjection, RelationResultOutput
    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.question_contract import (QueryQuestionContract, QueryRequestedFact,
        QueryRequestedOutput, QueryOperationDeclaration, QuerySourceDeclaration, QueryParameterDeclaration, InputTerm, InputDenotation)
    from fervis.lookup.question_contract.model import InputDenotationKind
    from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType
    from fervis.lookup.contract_codec import canonical_contract_fingerprint
    from fervis.lookup.answer_rendering import render_fact_result, rendered_fact_text
    read = _read('instruments')
    read = replace(read, fields=(*read.fields,
        CatalogField('name','string',path='name',row_path_id='root'),
        CatalogField('precision','number',path='precision',row_path_id='root')))
    catalog = RelationCatalog(reads=(read,))
    source = build_api_row_source_catalog(catalog).sources[0]
    fields = tuple(field for field in source.fields if not field.declared_entity_kind)
    relation = Relation('rows',RelationSource(SourceKind.API_READ,read_id=read.id,row_source_id=source.id),
        tuple(RelationField(field.id,(FieldBindingRole.OUTPUT,)) for field in fields))
    operation = Operation('query',SqlQuerySpec('SELECT precision FROM instruments WHERE name=:name',
        (SqlRelationInput('instruments','rows',tuple(SqlColumnBinding(field.path,field.id) for field in fields)),),
        (SqlOutputField('precision','number'),),(SqlNamedInput('name',ParameterRef('question.name')),),scalar=True),
        output_relation='query_rows')
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'What is the precision of the Borealis instrument?')
    projection = RelationResultOutput('result_precision','query_rows',field_id='precision',label='precision',role='answer_value')
    fact = QueryRequestedFact('fact_1',origin,
        (QueryOperationDeclaration(operation.id,canonical_contract_fingerprint(operation)),),
        (QueryRequestedOutput.from_projection('r1',origin,projection,'number'),),input_refs=('i1',),
        sources=(QuerySourceDeclaration(relation.id,canonical_contract_fingerprint(relation)),))
    question = QueryQuestionContract((InputTerm('i1',origin,'Borealis',TextType()),),(fact,),
        (InputDenotation('d1','i1','instrument name','The supplied name identifies the requested instrument.',
                         None,InputDenotationKind.NON_IDENTITY_SCALAR),))
    def bindings(name):
        value = FactValue.literal(id='name',known_input_id='i1',literal_type=LiteralType.STRING,
                                  value=name,proof_refs=('question_input:i1',))
        return BindingSet((ParameterBinding('question.name',value,
            BindingProvenance(BindingProvenanceKind.QUESTION_INPUT,('question_input:i1',))),))
    program = AnswerProgram(parameters=(ParameterDeclaration('question.name',ParameterRole.QUESTION_INPUT,
        ParameterValueType.STRING,input_ref='i1',input_use_refs=(fact.input_use_ref('i1'),)),),
        relations=(relation,),operations=(operation,),fulfillment=(FactFulfillment('fact_1','r1',projection.id),),
        result_projection=ResultProjection(relation_outputs=(projection,)))
    fact = replace(fact,parameters=tuple(QueryParameterDeclaration(item.id,canonical_contract_fingerprint(item)) for item in program.parameters))
    question = replace(question,requested_facts=(fact,))
    program,initial = compile_answer_program(program,question_contract=question,catalog=catalog,bindings=bindings('Borealis'))
    class Port:
        calls = 0
        precision = '3.5'
        def read(self, **kwargs):
            self.calls += 1
            return {'responseStatus':200,'responseBody':[{'id':1,'name':'Borealis','precision':'2.5'},
                {'id':2,'name':'Cirrus','precision':self.precision}]}
    port = Port()
    def run(selected):
        executed = invoke_answer_program(program=program,bindings=selected,environment=ExecutionEnvironment(catalog=catalog),
                                          ports=RuntimePorts(port,LookupMemory()))
        assert executed.issue is None
        return rendered_fact_text(render_fact_result(executed.fact_result))
    assert run(initial) == 'precision: 2.5'
    assert run(bindings('Cirrus')) == 'precision: 3.5'
    port.precision = '4.5'
    assert run(bindings('Cirrus')) == 'precision: 4.5'
    assert port.calls == 3
    # Continue the persisted callable signature through the real grounding seam.
    from fervis.lookup.answer_program.persistence import ProgramInvocation, StoredProgramInvocation
    from fervis.lookup.answer_program.rerun import RerunnableProgramInvocation
    from fervis.lookup.conversation_resolution.callable_frames import CallableFrameProgram, CallableFrameArgument, callable_frame_bindings
    from fervis.lookup.contract_codec import answer_program_id
    from fervis.lineage.enums import ProgramInvocationKind
    from fervis.memory.conversation_context.semantic_frames import _semantic_frame_projection
    from fervis.lookup.orchestration.semantic_compilation import SemanticCompilationRequest, resolve_semantic_continuation_arguments
    from fervis.lookup.question_contract import QuestionContractRequest
    from fervis.lookup.turn_prompts import HostPromptContext
    stored=StoredProgramInvocation(ProgramInvocation('inv1','run1',answer_program_id(program),initial,
        ProgramInvocationKind.COMPILED_QUESTION),program)
    signature=_semantic_frame_projection(program.fact_template[0],stored=stored).callable
    assert signature is not None
    frame=CallableFrameProgram(RerunnableProgramInvocation.parse(stored),signature,
        (CallableFrameArgument('question.name','i1',(fact.input_use_ref('i1'),),'And for Cirrus?','Cirrus',None),))
    class NoModel:
        def generate(self,**kwargs):
            raise AssertionError('A literal scalar continuation needs no grounding model call')
    continuation=SemanticCompilationRequest('run2','And for Cirrus?',
        QuestionContractRequest(current_question='And for Cirrus?',conversation_context={}),
        catalog,(),port,NoModel(),'openai',64,10,None,{},HostPromptContext())
    values=resolve_semantic_continuation_arguments(frame,continuation)
    assert port.calls==3
    rebound=callable_frame_bindings(frame,grounded_values=values)
    assert run(rebound)=='precision: 4.5'
    assert port.calls==4
    assert initial.get('question.name').value.payload.value=='Borealis'
