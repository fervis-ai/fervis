from types import SimpleNamespace
import pytest

from fervis.lookup.conversation_resolution import turn
from fervis.lookup.conversation_resolution.model import ConversationResolutionRequest
from fervis.lookup.conversation_resolution.tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
)


def test_parse_failure_retains_actionable_error_context(monkeypatch):
    artifact = SimpleNamespace(selected_tool_name=CONVERSATION_RESOLUTION_TOOL_NAME)
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
