from fervis.lookup.source_binding.prompt import SemanticSourceBindingTurnPrompt
from fervis.lookup.source_binding.model import SourceRealization
from fervis.lookup.turn_prompts import build_turn_prompt_context
from tests.lookup.source_binding.test_choice_requirements import (
    _source_required_choice_request,
)


def test_predicate_binding_receives_question_authority():
    request, _, _ = _source_required_choice_request(choices=("active", "deleted"))
    prompt = SemanticSourceBindingTurnPrompt(SourceRealization(request, {}, {}, {}))
    question = "Count canary-filter canceled workflow runs"
    artifact = prompt.to_model_invocation(
        build_turn_prompt_context(current_question=question, conversation_context={})
    )
    assert question in artifact.prompt_text
