from fervis.lookup.clarification import (
    TargetReferenceUnsupported,
    clarify,
    render_clarification_question,
)


def test_unsupported_reference_question_does_not_claim_the_value_was_found() -> None:
    clarification = clarify(
        TargetReferenceUnsupported(
            clarification_id="clarification_staff_id",
            requested_fact_id="fact_1",
            known_input_id="staff_id_qi_1",
            source_text="51515151-0000-0000-0002-000000000002",
            target_label="staff member identifier",
        )
    )

    assert render_clarification_question(clarification) == (
        "I could not resolve the staff member identifier "
        '"51515151-0000-0000-0002-000000000002". '
        "Which staff member identifier should I use?"
    )
