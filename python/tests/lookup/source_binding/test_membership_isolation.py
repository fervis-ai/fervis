from fervis.lookup.source_binding.membership import SourceMembershipTurnPrompt
from fervis.lookup.source_binding.model import SourceRealization
from fervis.lookup.turn_prompts import build_turn_prompt_context
from tests.lookup.source_binding.test_choice_requirements import _source_required_choice_request


def test_ordinary_membership_does_not_receive_question_qualifications():
    request, _, _ = _source_required_choice_request(choices=('active', 'deleted'))
    prompt = SourceMembershipTurnPrompt(SourceRealization(request, {}, {}, {}))
    artifacts = tuple(prompt.to_model_invocation(build_turn_prompt_context(
        current_question=question, conversation_context={},
    )) for question in ('Count canary-filter-A sales', 'Count canary-filter-B canceled sales'))
    assert artifacts[0].prompt_text == artifacts[1].prompt_text
    assert 'canary-filter' not in artifacts[0].prompt_text
    assert 'boolean_requirements' not in artifacts[0].prompt_text
