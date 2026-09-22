"""Logical obligations are authored and validated before source realization."""
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from fervis.lookup.question_contract import QuestionContractRequest
from fervis.lookup.question_contract.model import Aggregate, AggregateFunction, TemporalBucket, TemporalGrain
from fervis.lookup.orchestration.logical_planning import author_logical_plan


def case():
    path=Path(__file__).resolve().parents[2]/'conformance/cases/algorithms/semantic_kernel/non_identity_group_value_is_a_result_key.yaml'
    return yaml.safe_load(path.read_text())['input']


@pytest.mark.parametrize('change_grain', [False, True])
def test_logical_plan_preserves_grouping_and_calculation_before_any_api_binding(change_grain):
    fixture=case()
    payload=deepcopy(fixture['payload'])
    if change_grain:
        payload['outcome']['answer_requests'][0]['grouping'][0]['expression']['grain']='week'
    seen=[]
    def turn(purpose,prompt,parse):
        seen.append(type(prompt).__name__)
        return parse(fixture['frame_payload'] if len(seen)==1 else payload)
    request=QuestionContractRequest(fixture['question_context_texts'][0],{})
    if change_grain:
        with pytest.raises(ValueError,match='grain|grouping'):
            author_logical_plan(request,turn=turn)
        return
    parsed=author_logical_plan(request,turn=turn)
    assert seen==['SemanticQuestionFrameTurnPrompt','SemanticQuestionContractTurnPrompt']
    from fervis.lookup.contract_codec import canonical_contract_json,decode_canonical_contract
    from fervis.lookup.question_contract import QuestionContract
    restored=decode_canonical_contract(canonical_contract_json(parsed.contract),QuestionContract)
    fact,=restored.requested_facts
    aggregate,=(node for node in fact.expressions if isinstance(node,Aggregate))
    bucket,=(node for node in fact.expressions if isinstance(node,TemporalBucket))
    assert aggregate.function is AggregateFunction.COUNT
    assert bucket.grain is TemporalGrain.DAY
    assert fact.outputs[0].expression_ref==fact.grouping_refs[0]==fact.ordering[0].expression_ref
    assert fact.outputs[1].expression_ref==aggregate.id
    assert fact.qualification_ref is not None


def test_incomplete_question_never_authors_a_computation():
    calls=[]
    def turn(purpose,prompt,parse):
        calls.append(prompt)
        return parse({'decision_basis':'The desired result is missing.','outcome':{
            'kind':'missing_requested_fact','source_text':'Please check.',
            'why_question_is_incomplete':'No requested result.'}})
    from fervis.lookup.question_contract import QuestionContractNeedsClarification
    assert isinstance(author_logical_plan(QuestionContractRequest('Please check.',{}),turn=turn),QuestionContractNeedsClarification)
    assert len(calls)==1


def test_equal_operands_in_two_facts_keep_independent_input_ownership():
    from dataclasses import replace
    from fervis.lookup.answer_program.values import FactValue, LiteralType
    from fervis.lookup.grounding.semantic import CanonicalInputValue
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.question_contract.model import Comparison
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from fervis.lookup.orchestration.logical_planning import prepare_logical_realizations
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query

    original = employee_query(manager_minimum=True).request
    first = original.index.requested_fact
    first_input = original.index.input_by_ref["minimum"]
    first_denotation = original.index.input_denotation_by_ref["minimum"]
    second_input = replace(first_input, id="other_minimum")
    second_denotation = replace(
        first_denotation, id="other_denotation", input_ref="other_minimum"
    )
    second = replace(
        first,
        id="fact_2",
        expressions=tuple(
            replace(node, right_ref="other_minimum")
            if isinstance(node, Comparison) and node.right_ref == "minimum"
            else node
            for node in first.expressions
        ),
    )
    inputs = {item.id: item for item in (first_input, second_input)}
    denotations = {item.input_ref: item for item in (first_denotation, second_denotation)}
    indexes = tuple(
        analyze_requested_fact(fact, inputs=inputs, input_denotations=denotations)
        for fact in (first, second)
    )
    logical = ParsedSemanticQuestionContract(
        "Compare two independently supplied thresholds.",
        QuestionContract(tuple(inputs.values()), (first, second), tuple(denotations.values())),
        indexes,
    )
    values = tuple(
        CanonicalInputValue(
            f"value_{position}",
            input_ref,
            tuple(use.use_ref for use in index.input_use_sites),
            FactValue.literal(
                id=f"value_{position}",
                literal_type=LiteralType.NUMBER,
                value="150",
                known_input_id=input_ref,
                proof_refs=(f"question_input:{input_ref}",),
            ),
            (f"question_input:{input_ref}",),
        )
        for position, (input_ref, index) in enumerate(
            zip(("minimum", "other_minimum"), indexes, strict=True), start=1
        )
    )
    requests = prepare_logical_realizations(
        logical,
        sources_by_fact={fact.id: original.source_catalog for fact in (first, second)},
        canonical_values=values,
    )
    assert [tuple(value.input_ref for value in item.canonical_values) for item in requests] == [
        ("minimum",),
        ("other_minimum",),
    ]
    assert requests[0].index.requested_fact != requests[1].index.requested_fact


def test_realization_preparation_preserves_plan_and_exposes_sufficient_alternatives():
    from dataclasses import replace
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from fervis.lookup.orchestration.logical_planning import prepare_logical_realizations
    from tests.lookup.source_binding._candidate_fixture import daily_observation_request
    existing=daily_observation_request()
    logical=ParsedSemanticQuestionContract('Daily sums.',QuestionContract((),(existing.index.requested_fact,)),(existing.index,))
    prepared,=prepare_logical_realizations(logical,sources_by_fact={'fact_1':existing.source_catalog},canonical_values=())
    assert prepared.index.requested_fact is logical.contract.requested_facts[0]
    assert prepared.row_references_for_set('fact_1:set:s1')==('raw_observations',)
    assert set(prepared.strategy.branches[0].source_refs)=={s.id for s in existing.source_catalog.sources}
    calendar_only=replace(existing.source_catalog,sources=existing.source_catalog.sources[:1])
    unavailable,=prepare_logical_realizations(logical,sources_by_fact={'fact_1':calendar_only},canonical_values=())
    assert unavailable.strategy.branches==()
    assert unavailable.index.requested_fact==logical.contract.requested_facts[0]


@pytest.mark.parametrize('fact_ids', [(),('foreign',),('fact_1','foreign')])
def test_realization_preparation_cannot_drop_or_introduce_requested_facts(fact_ids):
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from fervis.lookup.orchestration.logical_planning import prepare_logical_realizations
    from tests.lookup.source_binding._candidate_fixture import daily_observation_request
    existing=daily_observation_request()
    logical=ParsedSemanticQuestionContract('Daily sums.',QuestionContract((),(existing.index.requested_fact,)),(existing.index,))
    with pytest.raises(ValueError,match='requested fact'):
        prepare_logical_realizations(logical,sources_by_fact={key:existing.source_catalog for key in fact_ids},canonical_values=())


@pytest.mark.parametrize('drop_qualification', [False, True])
def test_compilation_cannot_replace_the_independently_authored_logical_fact(drop_qualification):
    from dataclasses import replace
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.orchestration.logical_planning import compile_logical_bindings
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query
    verified=employee_query()
    request=verified.request
    original=request.index.requested_fact
    logical=ParsedSemanticQuestionContract('Compare employees with their managers.',
        QuestionContract((),(original,)),(request.index,))
    if drop_qualification:
        changed=replace(original,qualification_ref=None)
        request=replace(request,index=analyze_requested_fact(changed,inputs={},input_denotations={}))
        with pytest.raises(ValueError,match='logical fact'):
            compile_logical_bindings(logical,requests=(request,),bindings_by_fact={original.id:verified.binding_plan})
        return
    compiled=compile_logical_bindings(logical,requests=(request,),bindings_by_fact={original.id:verified.binding_plan})
    from fervis.lookup.contract_codec import canonical_answer_program_json,decode_answer_program
    restored=decode_answer_program(canonical_answer_program_json(compiled.answer_program))
    assert restored.fact_template==(original,)
    assert restored.relation_guarantees
    assert restored.fulfillment
    assert all(op.spec.kind.value!='sql_query' for op in restored.operations)


@pytest.mark.parametrize('source_kind', ['memory', 'rest'])
def test_source_realization_binds_rows_and_preserves_the_typed_count_through_execution(source_kind):
    from fervis.lookup.orchestration.logical_planning import realize_and_compile_logical_plan
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
    contract,source,memory,_,_,original=_compile_memory_count(({'event_id':'a'},{'event_id':'b'}))
    request=original.request
    from fervis.lookup.relation_catalog import RelationCatalog
    catalog=RelationCatalog()
    if source_kind == 'rest':
        from dataclasses import replace
        from tests.lookup.relational_engine.test_dependent_reads import _read
        from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
        from fervis.lookup.available_sources import snapshot_source_catalog
        from fervis.lookup.orchestration.logical_planning import prepare_logical_realizations
        catalog=RelationCatalog(reads=(replace(_read('events'),candidate_keys=()),))
        source,=build_api_row_source_catalog(catalog).sources
        logical=ParsedSemanticQuestionContract('Count events.',contract,(request.index,))
        request,=prepare_logical_realizations(logical,sources_by_fact={contract.requested_facts[0].id:snapshot_source_catalog((source,))},canonical_values=())
    branch=request.strategy.branches[0].branch_id
    set_ref=request.index.subject_obligation.subject_set_ref.token
    turns=[]
    def turn(purpose,prompt,parse):
        turns.append(type(prompt).__name__)
        if len(turns)==1:
            return parse({'set_bindings':{set_ref:[{'branch_id':branch,'rows_ref':source.id,
                'mapping_basis':'Observed event rows represent events.', "record_fields": []}]},'fact_bindings':{},'association_bindings':{}})
        return parse({'populations':{set_ref:[{'branch_id':branch,'logical_set_meaning':contract.requested_facts[0].sets[0].origin.meaning,
            'mapping_basis':'All observed event rows belong to this event population.','population':{'kind':'exact_population'}}]}})
    logical=ParsedSemanticQuestionContract('Count events.',contract,(request.index,))
    compiled=realize_and_compile_logical_plan(logical,
        sources_by_fact={contract.requested_facts[0].id:request.source_catalog},canonical_values=(),turn=turn)
    assert turns==['SemanticSourceRealizationTurnPrompt','SetPopulationTurnPrompt']
    from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.relation_catalog import RelationCatalog
    from fervis.lookup.memory.projection import LookupMemory
    from fervis.lookup.contract_codec import canonical_answer_program_json,decode_answer_program
    program=decode_answer_program(canonical_answer_program_json(compiled.answer_program))
    calls=[]
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append((endpoint_name,args))
            return {'responseStatus':200,'responseBody':[{'id':1},{'id':1}]}
    memories=(memory,) if source_kind=='memory' else ()
    execution=invoke_answer_program(program=program,bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog,memory_relations=memories),
        ports=RuntimePorts(Port(),LookupMemory(relations=memories)))
    assert calls==([] if source_kind=='memory' else [('events',{})])
    assert execution.issue is None
    assert next(iter(execution.fact_result.outcome.projected_rows[0].values.values()))==2


def test_typed_pipeline_stops_before_model_realization_when_sources_are_insufficient():
    from dataclasses import replace
    from fervis.lookup.orchestration.logical_planning import realize_and_compile_logical_plan
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from fervis.lookup.source_binding import SourceRealizationUnavailable
    from tests.lookup.source_binding._candidate_fixture import daily_observation_request
    request=daily_observation_request()
    logical=ParsedSemanticQuestionContract('Daily sums.',QuestionContract((),(request.index.requested_fact,)),(request.index,))
    calendar=replace(request.source_catalog,sources=request.source_catalog.sources[:1])
    def turn(*args):
        pytest.fail('An impossible date-only SUM must not consume a model call')
    outcome=realize_and_compile_logical_plan(logical,sources_by_fact={'fact_1':calendar},canonical_values=(),turn=turn)
    assert isinstance(outcome,SourceRealizationUnavailable)
    assert outcome.requested_fact_id=='fact_1'
    assert outcome.unmet_requirement_refs


def test_compilation_rejects_a_substituted_derived_qualification_index():
    from dataclasses import replace
    from fervis.lookup.orchestration.logical_planning import compile_logical_bindings
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from fervis.lookup.qualification import QualificationDNF
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query
    verified=employee_query()
    original=verified.request.index
    logical=ParsedSemanticQuestionContract('Employee comparison.',QuestionContract((),(original.requested_fact,)),(original,))
    forged=replace(verified.request,index=replace(original,qualification=QualificationDNF.true('fact_1'),boolean_requirements=()))
    with pytest.raises(ValueError,match='analysis'):
        compile_logical_bindings(logical,requests=(forged,),bindings_by_fact={'fact_1':verified.binding_plan})


def test_conversation_request_is_the_only_owner_of_resolved_input_text():
    from fervis.lookup.conversation_resolution.compilation import CompiledConversationResolution,ResolvedLiteralQuestionInput
    fixture=deepcopy(case())
    question='For each day in that month, how many events occurred, ordered by day?'
    fixture['frame_payload']['outcome']['supplied_values']['operands'][0]['non_entity_value']['value']['origin']={
        'kind':'conversation_resolution','resolved_input_ref':'conversation.v1'}
    resolution=CompiledConversationResolution(question,fixture['question_context_texts'][0],(),
        (ResolvedLiteralQuestionInput('conversation.v1','that month','March 2026'),),None,(),())
    calls=[]
    def turn(purpose,prompt,parse):
        calls.append(prompt)
        return parse(fixture['frame_payload'] if len(calls)==1 else fixture['payload'])
    result=author_logical_plan(QuestionContractRequest(question,{},conversation_resolution=resolution),turn=turn)
    assert len(calls)==2
    assert result.contract.inputs[0].operand=='March 2026'
    assert result.contract.inputs[0].origin.resolved_input_ref=='conversation.v1'
