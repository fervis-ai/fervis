from fervis.lookup.grounding.model import GroundingIssue, GroundingTerminalKind
from fervis.lookup.orchestration.terminal_results import _grounding_issue_fact_result
from fervis.lookup.outcomes.model import NeedsClarification


def test_grounding_clarification_carries_grounding_evidence_ref():
    result = _grounding_issue_fact_result(
        (
            GroundingIssue(
                kind=GroundingTerminalKind.UNRESOLVED_REFERENCE,
                known_input_id="q1_staff_id",
                requested_fact_id="fact_1",
                known_input_text="staff-9999",
                proof_refs=("known_input:q1_staff_id",),
            ),
        )
    )

    assert isinstance(result.outcome, NeedsClarification)
    assert result.outcome.clarifications[0].evidence_refs == (
        "known_input:q1_staff_id",
        "grounding:unresolved_reference",
    )
