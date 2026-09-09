from dataclasses import replace
import pytest
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.reference_compilation import compile_reference_result,combine_reference_members
from fervis.lookup.relational_sql.acquisition import ApiView
from fervis.lookup.relational_sql.outputs import QueryOutput
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.relational_sql.parameters import query_parameter_menu
from fervis.lookup.answer_program.operations import SqlNamedInput
from fervis.lookup.answer_program.result_projection import EntityKeyProjection,EntityKeyProjectionComponent
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.question_contract import InputTerm,InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,TextType,CollectionType
from fervis.lookup.relation_catalog import RelationCatalog,CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.identity_types import IdentityExecutionFailureReason
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize('data_case',['distinct','same_entity','missing','selected_member'])
def test_reference_collection_resolves_every_member_before_union(data_case):
    read=replace(_read('sites'),fields=(*_read('sites').fields,
        CatalogField('name','string',path='name',row_path_id='root'),
        CatalogField('alias','string',path='alias',row_path_id='root',nullable=True)))
    catalog=RelationCatalog(reads=(read,));source=build_api_row_source_catalog(catalog).sources[0]
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Beta and Alpha')
    term=InputTerm('i1',origin,('Beta','Alpha'),CollectionType(TextType()))
    denotation=InputDenotation('d1','i1','named sites','The names identify sites.','sites',InputDenotationKind.IDENTITY_REFERENCE)
    value=FactValue.string_set(id='names',known_input_id='i1',values=term.operand,proof_refs=('question_input:i1',))
    menu=query_parameter_menu((CanonicalInputValue('names','i1',('fact_1:sql_input:i1',),value,value.proof_refs),))
    members=[]
    for index,name in enumerate(value.payload.values):
        reference_id=f'reference_i1_member_{index}'
        compiled=compile_query_answer(question=f'Identify {name}.',query='SELECT id AS key_id FROM sites WHERE name=$name OR alias=$name',
            views=(ApiView('sites',source.id,{f.path:f.id for f in source.fields if f.path},{}),),catalog=catalog,
            output_types={'key_id':'integer'},result_contract=ResultContract('rows',('key_id',)),
            query_parameters=(SqlNamedInput('name',replace(menu.expressions['p1_1'],item_index=index)),),
            parameters=menu.program_inputs.parameters,bindings=menu.program_inputs.bindings,inputs=(term,),input_denotations=(denotation,),
            public_outputs=(QueryOutput('site',identity=EntityKeyProjection('sites','primary',(EntityKeyProjectionComponent('id','key_id'),))),),
            namespace=reference_id+'.', lookup_input_ref='i1')
        from fervis.lookup.canonical_data import EntityKeyValue,EntityKeyComponentValue
        selected=EntityKeyValue('sites','primary',(EntityKeyComponentValue('id',3),)) if data_case=='selected_member' and name=='Beta' else None
        members.append(compile_reference_result(compiled,input_ref='i1',output_types={'key_id':'integer'},reference_id=reference_id,operand=name,selected_key=selected,selection_proof_ref='clarification_response:choose_beta' if selected else ''))
    combined=combine_reference_members(term,tuple(members))
    final=compile_query_answer(question='Count the referenced sites.',query='SELECT COUNT(*) AS total FROM reference_i1',
        views=(),relation_views=(combined.view,),prerequisites=combined.program,bindings=combined.bindings,catalog=catalog,
        output_types={'total':'integer'},result_contract=ResultContract('scalar'),inputs=(term,),input_denotations=(denotation,),expected_input_refs=('i1',))
    class Port:
        calls=0
        def read(self,**kwargs):
            self.calls+=1
            rows=[{'id':1,'name':'Alpha','alias':'Beta' if data_case=='same_entity' else None}]
            if data_case in {'distinct','selected_member'}:rows.append({'id':2,'name':'Beta','alias':None})
            if data_case=='selected_member':rows.append({'id':3,'name':'Beta','alias':None})
            return {'responseStatus':200,'responseBody':rows}
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
    from fervis.lookup.answer_program.values import BindingSet
    persisted = decode_answer_program(canonical_answer_program_json(final.program))
    if data_case == 'distinct':
        for replacement in [('Alpha', 'Beta', 'Gamma'), ('Alpha', 'Gamma'), ('Alpha',)]:
            changed = FactValue.string_set(id='names', known_input_id='i1', values=replacement, proof_refs=('question_input:i1',))
            rebound = BindingSet.from_bindings(tuple(replace(binding, value=changed) for binding in final.bindings.bindings))
            untouched = Port()
            with pytest.raises(ValueError, match='fixed|certified'):
                invoke_answer_program(program=persisted, bindings=rebound,
                    environment=ExecutionEnvironment(catalog=catalog), ports=RuntimePorts(untouched, LookupMemory()))
            assert untouched.calls == 0
    port=Port()
    result=invoke_answer_program(program=final.program,bindings=final.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert port.calls==1
    if data_case=='missing':
        assert result.fact_result is None
        assert result.issue.reference.reason is IdentityExecutionFailureReason.NOT_FOUND
        assert result.issue.reference.operand=='Beta'
    else:
        assert result.issue is None
        assert next(iter(result.fact_result.outcome.projected_rows[0].values.values()))==(2 if data_case in {'distinct','selected_member'} else 1)
        if data_case=='selected_member':
            rows=next(relation.rows for relation in result.relations if relation.id==combined.view.relation_id)
            assert {row['id'] for row in rows}=={1,3}
            from fervis.lookup.answer_program.values import BindingSet
            changed=FactValue.string_set(id='names',known_input_id='i1',values=('Alpha','Delta'),proof_refs=('question_input:i1',))
            rebound=BindingSet.from_bindings(tuple(replace(binding,value=changed) for binding in final.bindings.bindings))
            before=port.calls
            with pytest.raises(ValueError,match='fixed|certified'):
                invoke_answer_program(program=final.program,bindings=rebound,
                    environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
            assert port.calls==before
