from __future__ import annotations

from fervis.lookup.continuations import (
    ContinuationCarriedInput,
    ContinuationPlan,
    ContinuationPlanKind,
    ContinuationReplacement,
)
from fervis.lookup.conversation_resolution import (
    ConversationResolutionOverlay,
    LiteralQuestionInputOverlay,
    conversation_input_provenance_from,
)
from fervis.lookup.question_inputs import LiteralInputRole
from fervis.memory.conversation_context import ConversationReplaceablePart


def test_input_provenance_merges_current_resolved_reference_with_carried_prior_input():
    overlay = ConversationResolutionOverlay(
        current_question="What about Mombasa on that day?",
        value_frames=(),
        references=(),
        scopes=(),
        activated_memory_ids=(),
        used_source_card_ids=(),
        resolved_question_inputs=(
            LiteralQuestionInputOverlay(
                source_text="that day",
                resolved_input_ref="cr_input_1",
                resolved_value_text="Tuesday",
                value_meaning_hint="time scope",
                role=LiteralInputRole.TIME_VALUE,
            ),
        ),
    )
    plan = ContinuationPlan(
        kind=ContinuationPlanKind.SAME_FACT_INPUT_REPLACEMENT,
        current_question="What about Mombasa on that day?",
        resolved_request_text="sales count in Mombasa on Tuesday",
        frame_id="context_frame_1",
        prior_answer_fact="sales count in Nairobi on Tuesday",
        replacements=(
            ContinuationReplacement(
                part=ConversationReplaceablePart(
                    part_id="q_place",
                    kind="entity_identity",
                    text="Nairobi",
                ),
                current_text="Mombasa",
            ),
        ),
        carried_inputs=(
            ContinuationCarriedInput(
                part=ConversationReplaceablePart(
                    part_id="q_time",
                    kind="time_scope",
                    text="Tuesday",
                ),
                resolved_value_text="Tuesday",
                value_meaning_hint="time scope",
            ),
        ),
    )

    provenance = conversation_input_provenance_from(
        overlay=overlay,
        continuation_plan=plan,
    )

    assert provenance.to_prompt_payload() == {
        "question_context_kind": "prior_question_continuation",
        "resolved_request_text": "sales count in Mombasa on Tuesday",
        "inputs": [
            {
                "input_ref": "q_time",
                "kind": "literal_text",
                "question_input_source": "conversation_resolution",
                "value_source_text": "that day",
                "resolved_value_text": "Tuesday",
                "role": "time_value",
                "value_meaning_hint": "time scope",
                "sources": ["resolved_question_input", "continuation_carried"],
            },
            {
                "input_ref": "q_place",
                "kind": "literal_text",
                "question_input_source": "question_context",
                "value_source_text": "Mombasa",
                "resolved_value_text": "Mombasa",
                "role": "reference_value",
                "sources": ["continuation_replacement"],
            },
        ],
    }
