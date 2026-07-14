from __future__ import annotations

import pytest

from fervis.lookup.clarification import (
    AmbiguousQuestionInterpretation,
    ClarificationEvidence,
    ClarificationEvidenceKind,
    ConversationInterpretationCandidate,
    ConversationInterpretationEvidence,
    clarification_from_payload,
    clarification_payload,
    clarify,
)


def test_clarification_payload_round_trips_ambiguous_interpretation() -> None:
    clarification = clarify(
        AmbiguousQuestionInterpretation(
            clarification_id="clarify_period",
            requested_fact_id="question_contract",
            source_text="last March",
            candidates=(
                ConversationInterpretationCandidate(
                    id="period_2025",
                    contextualized_question="March 2025",
                    source_evidence=(
                        ConversationInterpretationEvidence(
                            source_id="question_contract",
                            exact_source_texts=("last March",),
                        ),
                    ),
                ),
                ConversationInterpretationCandidate(
                    id="period_2026",
                    contextualized_question="March 2026",
                    source_evidence=(
                        ConversationInterpretationEvidence(
                            source_id="question_contract",
                            exact_source_texts=("last March",),
                        ),
                    ),
                ),
            ),
            evidence=(
                ClarificationEvidence(
                    kind=ClarificationEvidenceKind.QUESTION_CONTRACT,
                    id="question_contract:needs_clarification",
                ),
            ),
        )
    )

    assert (
        clarification_from_payload(clarification_payload(clarification))
        == clarification
    )


@pytest.mark.parametrize("missing_key", ("owner", "continuation"))
def test_clarification_payload_rejects_incomplete_owner_spec(
    missing_key: str,
) -> None:
    clarification = clarify(
        AmbiguousQuestionInterpretation(
            clarification_id="clarify_period",
            requested_fact_id="question_contract",
            source_text="last March",
            accepts_free_text=True,
        )
    )
    payload = clarification_payload(clarification)
    del payload[missing_key]

    with pytest.raises(ValueError):
        clarification_from_payload(payload)
