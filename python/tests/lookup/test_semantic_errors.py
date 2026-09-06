from types import SimpleNamespace

import pytest

from fervis.lookup import semantic_turn
from fervis.lookup.model_turn import ModelTurnOutput
from fervis.model_io.turn_artifacts import ModelTurnArtifact
from fervis.model_io.backbone.dto import ToolSpec


def test_parser_failure_preserves_the_first_structural_reason(monkeypatch):
    artifact = ModelTurnArtifact(system_prompt="", prompt_text="", provider_schema={},
                                 tool_specs=(ToolSpec(name="submit_test", description="Test", input_schema={}),), submitted_payload={})
    monkeypatch.setattr(semantic_turn, "run_one_of_tool_model_turn", lambda **kwargs:
                        ModelTurnOutput({}, {"costUsd": 0.01}, 10, artifact))
    prompt = SimpleNamespace(turn_name="source binding", to_model_invocation=lambda context: object())

    def parse(payload):
        raise ValueError("field binding belongs to another source")

    with pytest.raises(semantic_turn.SemanticTurnGenerationError) as captured:
        semantic_turn.generate_semantic_turn(prompt=prompt, context=None, parse=parse,
                                            model_port=None, provider="openai", max_thinking_tokens=100)
    assert captured.value.error_context == {
        "exception_class": "ValueError", "message": "field binding belongs to another source",
    }
    assert captured.value.usage == {"costUsd": 0.01}
