from dataclasses import replace
from decimal import Decimal
import pytest

from fervis.lookup.answer_program.api_reads import ApiReadSession
from fervis.lookup.answer_program.model import AnswerProgram, RelationProgram
from fervis.lookup.answer_program.operations import (Operation, SqlQuerySpec, SqlRelationInput,
    SqlColumnBinding, SqlOutputField, SqlNamedInput, ComputeSpec)
from fervis.lookup.answer_program.values import NodeOutputRef, ConstantRef, FactValue, LiteralType
from fervis.lookup.answer_program.relations import Relation, RelationSource, RelationField, FieldBindingRole, SourceKind, EndpointParamBinding
from fervis.lookup.answer_program.result_projection import RelationResultOutput, ResultProjection
from fervis.lookup.question_contract import QueryRequestedFact, QueryRequestedOutput, QueryOperationDeclaration, QuerySourceDeclaration
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.contract_codec import canonical_contract_fingerprint
from fervis.lookup.relational_sql.request_contract import verify_query_request
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_execution import execute_access_program
from tests.lookup.relational_engine.test_dependent_reads import _read


def _source(source):
    return Relation(source.read_id,RelationSource(SourceKind.API_READ,read_id=source.read_id,row_source_id=source.id),
                    (RelationField(source.fields[0].id,(FieldBindingRole.OUTPUT,)),))


def test_request_rejects_changed_source_selection_and_binding_constants():
    catalog = RelationCatalog(reads=(_read('items'),))
    source = build_api_row_source_catalog(catalog).sources[0]
    relation = _source(source)
    operation = Operation('query',SqlQuerySpec('SELECT id FROM items',
        (SqlRelationInput('items',relation.id,(SqlColumnBinding('id',source.fields[0].id),)),),
        (SqlOutputField('id','integer'),)),output_relation='answer')
    projection = RelationResultOutput('r1','answer',field_id='id',role='answer_value')
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Which items?')
    def request_for(selected):
        return QueryRequestedFact('fact_1',origin,(QueryOperationDeclaration('query',canonical_contract_fingerprint(operation)),),
            (QueryRequestedOutput.from_projection('r1',origin,projection,'integer'),),
            sources=(QuerySourceDeclaration(selected.id,canonical_contract_fingerprint(selected)),))
    program = AnswerProgram(relations=(relation,),operations=(operation,),
                            result_projection=ResultProjection(relation_outputs=(projection,)))
    verify_query_request(request_for(relation),program)
    changed = replace(relation,source=replace(relation.source,read_id='another_population'))
    with pytest.raises(VerificationError,match='source differs'):
        verify_query_request(request_for(relation),replace(program,relations=(changed,)))
    def restricted(value):
        constant = FactValue.literal(id='status',literal_type=LiteralType.STRING,value=value)
        return replace(relation,source=replace(relation.source,param_bindings=(
            EndpointParamBinding('status',ConstantRef('status','v1',constant)),)))
    active, archived = restricted('active'), restricted('archived')
    verify_query_request(request_for(active),replace(program,relations=(active,)))
    with pytest.raises(VerificationError,match='source differs'):
        verify_query_request(request_for(active),replace(program,relations=(archived,)))


@pytest.mark.parametrize('producer_kind',['compute','sql'])
def test_derived_sql_parameters_keep_producer_specific_proofs(producer_kind):
    catalog = RelationCatalog(reads=tuple(_read(name) for name in ('items','first','second')))
    sources = {source.read_id:source for source in build_api_row_source_catalog(catalog).sources}
    relations = tuple(_source(source) for source in sources.values())
    def minimum(name):
        source=sources[name]
        return Operation(name,SqlQuerySpec(f'SELECT MIN(id) AS value FROM {name}',
            (SqlRelationInput(name,name,(SqlColumnBinding('id',source.fields[0].id),)),),
            (SqlOutputField('value','number'),),scalar=True),output_relation=name+'_value')
    if producer_kind=='compute':
        value=FactValue.literal(id='cutoff',literal_type=LiteralType.NUMBER,value='1',proof_refs=('external:threshold',))
        producers=(Operation('first',ComputeSpec(ConstantRef('cutoff','v1',value),output_scalar='value')),)
        expected='external:threshold'
    else:
        producers=(minimum('first'),minimum('second'))
        expected='read:first'
    query=Operation('query',SqlQuerySpec('SELECT COUNT(*) AS total FROM items WHERE id>:cutoff',
        (SqlRelationInput('items','items',(SqlColumnBinding('id',sources['items'].fields[0].id),)),),
        (SqlOutputField('total','number'),),(SqlNamedInput('cutoff',NodeOutputRef('first','value')),),scalar=True),
        output_relation='answer')
    class Port:
        def read(self,*,endpoint_name,args):
            rows=[{'id':1},{'id':2},{'id':3}] if endpoint_name=='items' else [{'id':1 if endpoint_name=='first' else 2}]
            return {'responseStatus':200,'responseBody':rows}
    executed=execute_access_program(RelationProgram(relations=relations,operations=(*producers,query)),
                                    catalog=catalog,read_session=ApiReadSession(Port()))
    result=executed.engine_output.relation('answer')
    assert result.rows == ({'total':Decimal('2')},)
    assert expected in result.evidence.proof_refs
    assert 'read:second' not in result.evidence.proof_refs


def test_compiler_rejects_source_population_changes_under_an_unchanged_request():
    from fervis.lookup.relation_catalog.model import CatalogParam
    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.values import BindingSet
    from fervis.lookup.answer_program.model import FactFulfillment
    from fervis.lookup.question_contract import QueryQuestionContract
    catalog=RelationCatalog(reads=(_read('items',params=(CatalogParam('status','status','query','string'),)),))
    source=build_api_row_source_catalog(catalog).sources[0]
    def selected(value):
        constant=FactValue.literal(id='status',literal_type=LiteralType.STRING,value=value,proof_refs=('question:status',))
        return replace(_source(source),source=replace(_source(source).source,param_bindings=(
            EndpointParamBinding('status',ConstantRef('status','v1',constant)),)))
    active=selected('active')
    operation=Operation('query',SqlQuerySpec('SELECT id FROM items',
        (SqlRelationInput('items',active.id,(SqlColumnBinding('id',source.fields[0].id),)),),
        (SqlOutputField('id','integer'),)),output_relation='answer')
    projection=RelationResultOutput('r1','answer',field_id='id',label='item',role='answer_value')
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Which active items?')
    fact=QueryRequestedFact('fact_1',origin,(QueryOperationDeclaration('query',canonical_contract_fingerprint(operation)),),
        (QueryRequestedOutput.from_projection('r1',origin,projection,'integer'),),
        sources=(QuerySourceDeclaration(active.id,canonical_contract_fingerprint(active)),))
    question=QueryQuestionContract((),(fact,))
    program=AnswerProgram(relations=(active,),operations=(operation,),
        fulfillment=(FactFulfillment('fact_1','r1','r1'),),result_projection=ResultProjection(relation_outputs=(projection,)))
    compile_answer_program(program,question_contract=question,catalog=catalog,bindings=BindingSet())
    with pytest.raises(VerificationError,match='source differs'):
        compile_answer_program(replace(program,relations=(selected('archived'),)),
                               question_contract=question,catalog=catalog,bindings=BindingSet())
