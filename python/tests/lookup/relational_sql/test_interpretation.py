from dataclasses import replace
import pytest

from fervis.lookup.answer_program.values import FactValue, LiteralType, ValueComponent, BindingSet
from fervis.lookup.answer_program.operations import SqlNamedInput
from fervis.lookup.answer_program.inputs import compile_relation_program_inputs
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.persistence import StoredProgramInvocation, ProgramInvocation
from fervis.lookup.contract_codec import canonical_contract_fingerprint, answer_program_id, canonical_answer_program_json, decode_answer_program
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.read_eligibility import SemanticReadEligibilityResult, ReadRequirementAssessment, SemanticReadDecision
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.model import CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.relational_sql.parameters import query_parameter_menu, with_catalog_choices
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.question_contract import InputTerm, InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType
from fervis.lookup.memory.projection import LookupMemory
from fervis.lineage.enums import ProgramInvocationKind
from fervis.memory.conversation_context.semantic_frames import _semantic_frame_projection
from tests.lookup.relational_engine.test_dependent_reads import _read


def test_catalog_interpretation_preserves_original_input_and_cannot_be_silently_rebound():
    read=_read('observations')
    read=replace(read,fields=(*read.fields,CatalogField('active','boolean',path='active',row_path_id='root',
        metadata={'description':'True means accepted; false means failed.'})))
    catalog=RelationCatalog(reads=(read,))
    rows=build_api_row_source_catalog(catalog)
    source=rows.sources[0]
    source_catalog=build_available_source_catalog(rows,read_eligibility=SemanticReadEligibilityResult((
        ReadRequirementAssessment('fact_1',read.id,(source.id,),read.id,tuple(f.field_ref for f in source.fields),
            'The observation rows carry the acceptance state.',SemanticReadDecision.RETAIN),),()))
    original=FactValue.literal(id='category',known_input_id='i1',literal_type=LiteralType.STRING,
        value='accepted',label='accepted',proof_refs=('question_input:i1',))
    menu=with_catalog_choices(query_parameter_menu((CanonicalInputValue(original.id,'i1',
        ('fact_1:sql_input:i1',),original,original.proof_refs),)),source_catalog=source_catalog,source_refs={source.id})
    choice=next(name for name,expr in menu.expressions.items() if name.startswith('c') and
        expr.value.payload.component_value(ValueComponent.VALUE) is True)
    assert menu.descriptions['p1_1']['value'] == 'accepted'
    assert menu.descriptions['p1_1']['may_interpret']
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'How many accepted observations?')
    parameters=tuple(replace(item,fixed_value_fingerprint=canonical_contract_fingerprint(original.payload))
        for item in menu.program_inputs.parameters)
    view_catalog=build_query_view_catalog(catalog)
    compiled=compile_query_answer(question=origin.meaning,query=f'SELECT COUNT(*) AS total FROM "{view_catalog.views[0].name}" WHERE active=${choice}',
        views=view_catalog.views,output_types={'total':'integer'},result_contract=ResultContract('scalar'),catalog=catalog,
        query_parameters=(SqlNamedInput(choice,menu.expressions[choice]),),parameters=parameters,
        bindings=menu.program_inputs.bindings,meaning_inputs=(menu.expressions['p1_1'],),
        inputs=(InputTerm('i1',origin,'accepted',TextType()),),
        input_denotations=(InputDenotation('d1','i1','acceptance category','A category from the question.',None,
            InputDenotationKind.NON_IDENTITY_SCALAR),))
    assert decode_answer_program(canonical_answer_program_json(compiled.program))==compiled.program
    class Port:
        def read(self,**kwargs):
            return {'responseStatus':200,'responseBody':[{'id':1,'active':True},{'id':2,'active':False},{'id':3,'active':True}]}
    executed=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[2]
    stored=StoredProgramInvocation(ProgramInvocation('inv','run',answer_program_id(compiled.program),compiled.bindings,
        ProgramInvocationKind.COMPILED_QUESTION),compiled.program)
    signature=_semantic_frame_projection(compiled.program.fact_template[0],stored=stored).callable
    assert signature.parameters==()
    changed=replace(original,payload=replace(original.payload,value='failed'))
    rebound=BindingSet.from_bindings(tuple(replace(binding,value=changed) for binding in compiled.bindings.bindings))
    with pytest.raises(ValueError,match='fixed|certified|mapping'):
        compile_relation_program_inputs(compiled.program,bindings=rebound)


def test_temporal_symbols_preserve_inclusive_boundary_semantics():
    value=FactValue.time(id='month',known_input_id='i1',expression='this month',
        resolved_start='2026-09-01',resolved_end='2026-09-30',granularity='month',proof_refs=('question_input:i1',))
    menu=query_parameter_menu((CanonicalInputValue(value.id,'i1',('fact_1:sql_input:i1',),value,value.proof_refs),))
    assert menu.descriptions['p1_1']['value_type']=='date'
    assert menu.descriptions['p1_2']['boundary']=='inclusive'
    assert menu.descriptions['p1_2']['granularity']=='month'
