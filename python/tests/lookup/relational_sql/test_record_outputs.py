"""Observed API records answer entity questions without fabricating identity authority."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from jsonschema import validate

from fervis.lookup.answer_program.result_projection import RelationResultOutput,ResultProjectionError
from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.model import CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.relational_sql.acquisition import ApiView
from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt,parse_query_answer
from fervis.lookup.relational_sql.outputs import QueryOutput
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.results import ResultContract,ResultOrder
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.contract_codec import canonical_answer_program_json,decode_answer_program
from tests.lookup.relational_engine.test_dependent_reads import _read
from tests.lookup.relational_sql.test_authoring import payload


@pytest.mark.parametrize('name',['Shared','',None])
def test_unannotated_record_projection_preserves_distinct_tied_rows_and_roundtrips(name):
    read=replace(_read('items'),candidate_keys=(),fields=(
        CatalogField('items.id','integer',path='id',row_path_id='root'),
        CatalogField('items.name','string',path='name',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0]
    view=ApiView('items',source.id,{field.path:field.id for field in source.fields if field.path},{})
    compiled=compile_query_answer(question='Which items share the highest score?',
        query='SELECT id,name,1 AS score FROM items',views=(view,),output_types={'id':'integer','name':'string','score':'integer'},
        result_contract=ResultContract('rows',('id','name'),(ResultOrder('score',True),),'first_with_ties'),
        catalog=catalog,public_outputs=(QueryOutput('item',record_fields={'identifier':'id','name':'name'}),))
    persisted=decode_answer_program(canonical_answer_program_json(compiled.program));assert persisted==compiled.program
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':1,'name':name},{'id':2,'name':name}]}
    result=invoke_answer_program(program=persisted,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert result.issue is None
    values=[next(iter(row.values.values())) for row in result.fact_result.outcome.projected_rows]
    assert values==[{'identifier':1,'name':name},{'identifier':2,'name':name}]
    from fervis.lookup.lineage.results import _lineage_value
    assert _lineage_value(values[0])[1]=={'kind':'object','value':values[0]}
    assert compiled.question_contract.requested_facts[0].outputs[0].value_type=='object'
    from fervis.lookup.memory.outcomes import fact_result_answer_addresses
    from fervis.lookup.memory.projection import _project_memory_relation
    from fervis.memory.addresses import FactAddressKind,fact_address_from_payload
    addresses=tuple(fact_address_from_payload(address.to_dict()) for address in fact_result_answer_addresses(result.fact_result))
    relation_address=next(address for address in addresses if address.kind==FactAddressKind.RELATION)
    restored=_project_memory_relation(relation_address,artifact_id='saved',
        rows_by_address={address.address:address for address in addresses if address.kind==FactAddressKind.ROW}).relation
    assert restored.rows==({'id':1,'name':name},{'id':2,'name':name})
    assert restored.field_types=={'id':'integer','name':'string'}
    assert restored.field_answer_output_ids=={'id':('result_1',),'name':('result_1',)}
    assert restored.completeness.status.value=='complete'
    assert all(address.kind!=FactAddressKind.ENTITY for address in addresses)


def test_entity_request_can_declare_record_fields_without_declaring_a_nominal_key():
    meaning=SimpleNamespace(result_kind='qualifying_instances',selection_kind='all_results',
        output_origins=('item',),ordering_origins=(),output_kinds=('identity',))
    tables={'items':{'columns':{'id':{'type':'integer'},'name':{'type':'string'}},'request_parameters':[]}}
    body=payload(query='SELECT id,name FROM items',mode='rows',columns=[{'name':'id','value_type':'integer'},{'name':'name','value_type':'string'}],
        outputs=[{'kind':'record','label':'item','fields':[{'name':'identifier','column':'id'},{'name':'name','column':'name'}]}])
    validate(body,QueryAnswerPrompt(question='Which item?',meaning=meaning,tables=tables,parameters={})._schema())
    authored=parse_query_answer(body,table_names=set(tables),tables=tables,parameter_names=set(),meaning=meaning)
    assert authored.outputs[0].identity is None
    assert authored.outputs[0].columns==('id','name')


@pytest.mark.parametrize('changes',[{'field_id':'id'}, {'record_fields':{}}, {'record_fields':{'':'id'}}])
def test_record_projection_rejects_ambiguous_or_empty_projection(changes):
    with pytest.raises(ResultProjectionError):
        RelationResultOutput('result','rows',**{'record_fields':{'identifier':'id'},**changes})


def test_record_projection_checks_each_observed_field():
    projection=RelationResultOutput('result','rows',record_fields={'identifier':'id','name':'name'})
    with pytest.raises(ResultProjectionError,match='field'):
        projection.project({'id':1})


def test_reference_grounding_cannot_substitute_a_record_for_a_canonical_identity():
    from jsonschema import ValidationError
    from fervis.lookup.relational_sql.reference_planning import ReferenceMeaning,ReferenceQueryPrompt,parse_reference_query
    from fervis.lookup.relational_sql.execution import QueryValidationError
    from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'the configured item')
    meaning=ReferenceMeaning('fact_1','i1','item','the configured item',(origin,),('i1',),
        reference_text='the configured item',reference_kind='description')
    tables={'items':{'columns':{'id':{'type':'integer'}},'request_parameters':[]}}
    prompt=ReferenceQueryPrompt(meaning=meaning,tables=tables,parameters={})
    body=payload(query='SELECT id FROM items',mode='rows',columns=[{'name':'id','value_type':'integer'}],
        outputs=[{'kind':'record','label':'item','fields':[{'name':'identifier','column':'id'}]}])
    body['reference_binding']={'kind':'description','basis':'the configured item'}
    with pytest.raises(ValidationError):validate(body,prompt._schema())
    with pytest.raises(QueryValidationError,match='canonical identity'):
        parse_reference_query(body,prompt=prompt,menu=SimpleNamespace(expressions={},descriptions={}))
