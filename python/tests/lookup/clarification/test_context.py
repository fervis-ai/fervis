from fervis.lookup.clarification import (
    ClarificationAnnotation,
    ClarificationResponseSource,
    ConversationResolutionResponse,
)
from fervis.lookup.clarification.context import (
    active_clarification,
    active_clarification_from_source,
    clarification_context_source,
    extend_clarification_chain,
)


def test_active_clarification_round_trip_preserves_order_and_identity() -> None:
    responses = (
        _response(
            response_id="response_1",
            clarification_id="clarification_1",
            original_question="How many stores are in Nairobi?",
            question="Which matching place should I use?",
            answer="Nairobi",
        ),
        _response(
            response_id="response_2",
            clarification_id="clarification_2",
            original_question="Nairobi",
            question="Which kind of place should I use?",
            answer="Nairobi",
        ),
    )

    clarification = active_clarification(responses)
    source = clarification_context_source(clarification)
    restored = active_clarification_from_source(source)

    assert restored == clarification
    assert restored.lineage_refs == (
        "clarification_response:response_1",
        "clarification_response:response_2",
    )
    assert source.meaning_anchors[-1].occurrence == 3


def test_extending_chain_discards_non_annotation_responses() -> None:
    first = _response(
        response_id="response_1",
        clarification_id="clarification_1",
        original_question="How many stores are there?",
        question="Which place should I use?",
        answer="Nairobi",
    )
    selected_option = ConversationResolutionResponse(
        source=ClarificationResponseSource(
            response_id="selected_option",
            clarification_id="clarification_option",
            exact_user_text="Area: Nairobi",
        )
    )
    second = _response(
        response_id="response_2",
        clarification_id="clarification_2",
        original_question="Nairobi",
        question="Which kind of place should I use?",
        answer="Area",
    )

    chain = extend_clarification_chain((first, selected_option), second)

    assert chain == (first, second)


def _response(
    *,
    response_id: str,
    clarification_id: str,
    original_question: str,
    question: str,
    answer: str,
) -> ConversationResolutionResponse:
    return ConversationResolutionResponse(
        source=ClarificationResponseSource(
            response_id=response_id,
            clarification_id=clarification_id,
            exact_user_text=answer,
        ),
        annotation=ClarificationAnnotation(
            suspended_question_text=original_question,
            clarification_question_text=question,
        ),
    )
