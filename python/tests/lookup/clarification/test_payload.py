from __future__ import annotations

from fervis.lookup.clarification import (
    AmbiguousQuestionInterpretation,
    ClarificationEvidence,
    ClarificationEvidenceKind,
    ClarificationOption,
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
            options=(
                ClarificationOption(
                    id="period_2025",
                    label="March 2025",
                    value="2025-03",
                ),
                ClarificationOption(
                    id="period_2026",
                    label="March 2026",
                    value="2026-03",
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

    assert clarification_from_payload(clarification_payload(clarification)) == clarification
