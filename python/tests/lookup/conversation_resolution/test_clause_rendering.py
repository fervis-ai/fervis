"""One source of truth for the resolved question: attributed clause spans."""
import pytest
from fervis.lookup.conversation_resolution import parse_conversation_resolution, CONVERSATION_RESOLUTION_TOOL_NAME


def _clause(text):
    return {'current_clause_text':text, 'occurrence':1,
            'request_shape_basis':'The current clause supplies the request.',
            'request_shape_source':'current_clause_supplies_request',
            'resolved_text':text, 'retained_frame_parts':[], 'values':[]}


def test_question_is_derived_from_clauses_without_a_second_authored_copy():
    question = 'How many stores?\nWhich are open?'
    payload = {'kind':'conversation_resolution', 'current_question_text':question,
               'outcome':{'kind':'resolved', 'resolution_basis':'Both clauses are complete.',
                          'clauses':[_clause('How many stores?'), _clause('Which are open?')]}}
    result = parse_conversation_resolution(tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
                                           payload=payload, current_question=question)
    assert result.outcome.contextualized_question == question
    assert len(result.outcome.clauses) == 2


def test_overlapping_clause_spans_are_not_resolved_twice():
    question = 'How many stores?'
    payload = {'kind':'conversation_resolution', 'current_question_text':question,
               'outcome':{'kind':'resolved', 'resolution_basis':'One request.',
                          'clauses':[_clause(question), _clause(question)]}}
    with pytest.raises(ValueError, match='resolved clause spans overlap'):
        parse_conversation_resolution(tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
                                      payload=payload, current_question=question)


def test_resolved_request_is_visible_without_becoming_literal_input_authority():
    from fervis.lookup.conversation_resolution.compilation import CompiledConversationResolution, CompiledResolvedClause
    resolution = CompiledConversationResolution(
        current_question_text='What percentage increase is that?',
        contextualized_question='What percentage increase in revenue is that?',
        clauses=(CompiledResolvedClause('What percentage increase is that?',
                                        'What percentage increase in revenue from 999 is that?', (), ()),),
        inputs=(), frame_call=None, used_source_card_ids=(), used_memory_ids=(),
    )
    assert resolution.to_prompt_payload()['clauses'][0]['resolved_request'] == resolution.clauses[0].resolved_text
    assert resolution.context_texts() == ()


def test_unanchored_answer_values_do_not_compete_with_typed_request_frames():
    from fervis.memory.conversation_context import ConversationContextSource, ConversationContextFrame, ConversationFramePart, ConversationFramePartKind
    from fervis.lookup.conversation_resolution.prompt import ConversationResolutionTurnPrompt
    source = ConversationContextSource('answer', 'prior_fervis_answer', 'The revenue was 987654321.12.')
    frame = ConversationContextFrame('request:1', ('answer',),
        (ConversationFramePart('output', ConversationFramePartKind.REQUESTED_OUTPUT, 'revenue'),))
    prompt = ConversationResolutionTurnPrompt(question='What percentage increase is that?',
                                              context_sources=(source,), context_frames=(frame,)).to_model_invocation().prompt_text
    assert '987654321.12' not in prompt
    assert 'request_frame_ids' in prompt
    assert source.text == 'The revenue was 987654321.12.'
