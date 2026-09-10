from types import SimpleNamespace
import pytest
from fervis.lookup import semantic_turn
from fervis.lookup.orchestration import semantic_compilation as compilation
from fervis.lookup.orchestration.pipeline import _SemanticTurnRecorder
from fervis.lookup.model_turn import ModelTurnOutput,ModelTurnGenerationFailure
from fervis.model_io.turn_artifacts import ModelTurnArtifact
from fervis.model_io.turns import ModelTurnPurpose
from fervis.lookup.relational_sql.authoring import QueryAnswerPrompt,parse_query_answer
from fervis.lookup.question_contract import parse_semantic_question_frame
from fervis.lookup.turn_prompts import TurnPromptContext
from tests.lookup.question_contract.test_question_frame import _frame_payload
from tests.lookup.relational_sql.test_authoring import payload


def assert_count_query(answer):
    from fervis.lookup.relational_sql.execution import execute_query, SqlTable
    assert answer.output_types == {'total':'integer'}
    for count in (0,3):
        rows = tuple({'id':i} for i in range(count))
        result = execute_query(answer.query, tables={'items':SqlTable({'id':'BIGINT'}, rows)})
        assert result.rows == ((count,),)


@pytest.mark.parametrize('failure_kind',['repairable','repeated','provider'])
def test_validation_repair_is_bounded_and_observable(monkeypatch,failure_kind):
    question='How many stores?'
    meaning=parse_semantic_question_frame(_frame_payload(),question_context_texts=(question,)).answer_requests[0]
    prompt=QueryAnswerPrompt(question=question,meaning=meaning,tables={'items':{'columns':{'id':{'type':'integer'}}}},parameters={})
    attempts=[];failures=[];completed=[]
    def generate(**kwargs):
        invocation=kwargs['invocation'];attempts.append(invocation)
        arguments=payload(query='SELECT COUNT(*) AS total FROM items()' if len(attempts)==1 or failure_kind=='repeated'
            else 'SELECT COUNT(*) AS total FROM items')
        artifact=ModelTurnArtifact(system_prompt=invocation.system_prompt,prompt_text=invocation.prompt_text,
            provider_schema=invocation.provider_schema,tool_specs=invocation.tool_specs,submitted_payload=arguments)
        if failure_kind=='provider':
            raise ModelTurnGenerationFailure('provider unavailable',{},1,artifact,error_code='provider_rate_limited')
        return ModelTurnOutput(arguments,{'inputTokens':10,'costUsd':0.01},1,artifact)
    monkeypatch.setattr(semantic_turn,'run_one_of_tool_model_turn',generate)
    request=SimpleNamespace(model_port=None,provider='openai',max_thinking_tokens=100,
        validation_failure_observer=lambda purpose,failure:failures.append(failure))
    def run():return compilation._turn(ModelTurnPurpose.SOURCE_REALIZATION,prompt=prompt,
        context=TurnPromptContext(current_question=question),
        parse=lambda value:parse_query_answer(value,table_names={'items'},parameter_names=set()),request=request,
        on_turn=lambda purpose,result:completed.append(result))
    if failure_kind!='repairable':
        with pytest.raises(compilation.SemanticCompilationTurnError):run()
    else:
        assert_count_query(run().result)
        assert 'Compiler diagnostics' in attempts[-1].prompt_text
        assert 'Preserve the original task' in attempts[-1].prompt_text
        assert completed[0].usage['costUsd']==0.01
    assert len(attempts)==(1 if failure_kind=='provider' else 2)
    assert len(failures)==(0 if failure_kind=='provider' else 1)
    if failures:assert failures[0].usage['costUsd']==0.01


def test_attempt_recorder_charges_rejected_and_corrected_calls_once(monkeypatch):
    from fervis.lookup.orchestration import pipeline
    events=[]
    monkeypatch.setattr(pipeline,'_append_model_turn_failed',lambda state,**kw:events.append(('failed',kw['turn'],kw['strict_audit'])))
    monkeypatch.setattr(pipeline,'_append_model_turn_completed',lambda state,**kw:events.append(('completed',kw['turn'])))
    monkeypatch.setattr(pipeline,'_limit_before_next_model_turn',lambda *args:None)
    state=SimpleNamespace(ports=None,request=SimpleNamespace(run_id='run'))
    recorder=_SemanticTurnRecorder(state,1,{}, {})
    recorder.failed(ModelTurnPurpose.SOURCE_REALIZATION,SimpleNamespace(usage={'inputTokens':3,'costUsd':0.01}))
    recorder(ModelTurnPurpose.SOURCE_REALIZATION,SimpleNamespace(usage={'inputTokens':5,'costUsd':0.02}))
    assert events==[('failed',1,True),('completed',2)]
    assert recorder.turn_numbers[ModelTurnPurpose.SOURCE_REALIZATION]==[2]
    assert recorder.next_turn==3
    assert recorder.usage['inputTokens']==8
    assert recorder.usage['costUsd']==pytest.approx(0.03)


def test_correction_preserves_question_independence_and_frame_authority():
    from fervis.lookup.turn_prompts.correction import CorrectionTurnPrompt
    from fervis.lookup.source_reads.access_discovery import AccessDiscoveryRequest,ReadAccessTurnPrompt
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.question_contract import SemanticQuestionFrameTurnPrompt,QuestionContractRequest
    from fervis.lookup.turn_prompts.invocation import ProviderInvocationMetadata
    from tests.lookup.relational_engine.test_dependent_reads import _program
    _,_,catalog=_program()
    sources=build_api_row_source_catalog(catalog)
    discovery=ReadAccessTurnPrompt(AccessDiscoveryRequest(catalog,sources,(sources.sources[-1],)))
    context=TurnPromptContext(current_question='COUNT_ONLY_PAID_ORDERS')
    failure=SimpleNamespace(artifact=SimpleNamespace(submitted_payload={}),error_context={'message':'Invalid field reference'})
    original=discovery.to_model_invocation(context)
    corrected=CorrectionTurnPrompt(discovery,failure).to_model_invocation(context)
    assert 'COUNT_ONLY_PAID_ORDERS' not in original.prompt_text
    assert 'COUNT_ONLY_PAID_ORDERS' not in corrected.prompt_text
    assert original.system_prompt==corrected.system_prompt
    class Frame(SemanticQuestionFrameTurnPrompt):
        def provider_metadata(self):return ProviderInvocationMetadata({'test_marker':'preserved'})
    frame=Frame(QuestionContractRequest(current_question='How many stores?',conversation_context={}))
    original=frame.to_model_invocation(context)
    corrected=CorrectionTurnPrompt(frame,failure).to_model_invocation(context)
    assert original.system_prompt==corrected.system_prompt
    assert original.metadata==corrected.metadata
    assert original.response_contract==corrected.response_contract
    assert original.tool_contract==corrected.tool_contract


def test_real_structured_schema_failure_is_corrected_and_classified():
    from fervis.model_io.structured_output.errors import ModelValidationKind
    question='How many stores?'
    meaning=parse_semantic_question_frame(_frame_payload(),question_context_texts=(question,)).answer_requests[0]
    prompt=QueryAnswerPrompt(question=question,meaning=meaning,tables={'items':{'columns':{'id':{'type':'integer'}}}},parameters={})
    failures=[]
    class Model:
        calls=0
        def generate(self,**kwargs):
            self.calls+=1
            return {'answer':{'tool':'submit_query_answer','arguments':{} if self.calls==1 else payload()},
                'usage':{'inputTokens':3,'costUsd':0.01}}
    model=Model()
    request=SimpleNamespace(model_port=model,provider='openai',max_thinking_tokens=100,
        validation_failure_observer=lambda purpose,failure:failures.append(failure))
    result=compilation._turn(ModelTurnPurpose.SOURCE_REALIZATION,prompt=prompt,context=TurnPromptContext(current_question=question),
        parse=lambda value:parse_query_answer(value,table_names={'items'},parameter_names=set()),request=request,on_turn=None)
    assert_count_query(result.result)
    assert model.calls==2
    assert failures[0].validation_kind is ModelValidationKind.SCHEMA
    assert failures[0].error_context['validator']=='required'
