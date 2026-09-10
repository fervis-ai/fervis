from types import SimpleNamespace
import pytest

from fervis.lookup.conversation_resolution import turn
from fervis.lookup.conversation_resolution.model import ConversationResolutionRequest
from fervis.lookup.conversation_resolution.tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
)


def test_parse_failure_retains_actionable_error_context(monkeypatch):
    artifact = SimpleNamespace(selected_tool_name=CONVERSATION_RESOLUTION_TOOL_NAME, submitted_payload={})
    monkeypatch.setattr(
        turn,
        "run_one_of_tool_model_turn",
        lambda **kwargs: SimpleNamespace(
            arguments={
                "kind": "conversation_resolution",
                "current_question_text": "wrong",
                "outcome": {
                    "kind": "missing_input",
                    "why_unresolved": "missing",
                    "candidate_interpretations": [],
                },
            },
            usage={"inputTokens": 10},
            duration_ms=1,
            artifact=artifact,
        ),
    )
    with pytest.raises(turn.ConversationResolutionGenerationError) as failure:
        turn.generate_conversation_resolution(
            request=ConversationResolutionRequest(
                question="actual", conversation_context={}
            ),
            model_port=None,
            provider="openai",
            model_key="openai:gpt-5.4-mini",
            max_thinking_tokens=16384,
        )
    assert failure.value.error_context == {
        "exception_class": "ValueError",
        "message": "current_question_text must exactly match current question",
    }
    assert failure.value.usage == {"inputTokens": 10}


@pytest.mark.parametrize('failure_kind',['repairable','repeated','provider'])
def test_conversation_validation_uses_one_observable_correction(monkeypatch,failure_kind):
    from fervis.lookup.model_turn import ModelTurnOutput,ModelTurnGenerationFailure
    from fervis.model_io.turn_artifacts import ModelTurnArtifact
    question='And how many failed observations?'
    attempts=[];rejected=[]
    def generate(**kwargs):
        invocation=kwargs['invocation'];attempts.append(invocation)
        rewritten=len(attempts)==1 or failure_kind=='repeated'
        arguments={'kind':'conversation_resolution','current_question_text':question,
            'outcome':{'kind':'resolved','resolution_basis':'The clause supplies the complete request.',
                'clauses':[{'request_shape_source':'current_clause_supplies_request',
                    'current_clause_text':question,'occurrence':1,
                    'request_shape_basis':'No prior request parts are needed.',
                    'resolved_text':'How many failed observations are there?' if rewritten else question,
                    'retained_frame_parts':[],'values':[]}]}}
        artifact=ModelTurnArtifact(system_prompt=invocation.system_prompt,prompt_text=invocation.prompt_text,
            provider_schema=invocation.provider_schema,tool_specs=invocation.tool_specs,
            submitted_payload=arguments,selected_tool_name=CONVERSATION_RESOLUTION_TOOL_NAME)
        if failure_kind=='provider':
            raise ModelTurnGenerationFailure('provider unavailable',{},1,artifact)
        return ModelTurnOutput(arguments,{'costUsd':0.01},1,artifact)
    monkeypatch.setattr(turn,'run_one_of_tool_model_turn',generate)
    def run():
        return turn.generate_conversation_resolution(request=ConversationResolutionRequest(question=question,conversation_context={}),
            model_port=None,provider='openai',model_key='openai:gpt-5.4-mini',max_thinking_tokens=100,
            validation_failure_observer=rejected.append)
    if failure_kind=='repairable':
        result=run()
        assert result.result.outcome.contextualized_question==question
        assert result.usage=={'costUsd':0.01}
    else:
        with pytest.raises(turn.ConversationResolutionGenerationError):run()
    assert len(attempts)==(1 if failure_kind=='provider' else 2)
    assert len(rejected)==(0 if failure_kind=='provider' else 1)
    if rejected:
        assert rejected[0].usage=={'costUsd':0.01}
        assert 'context-free question must pass through without rewriting' in attempts[-1].prompt_text
        assert attempts[-1].system_prompt==attempts[0].system_prompt
