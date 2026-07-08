from fervis.lookup.grounding.model import GroundingIssue, GroundingTerminalKind
from fervis.lookup.clarification import clarification_payload
from fervis.lookup.orchestration.terminal_results import _grounding_issue_fact_result
from fervis.lookup.outcomes.model import NeedsClarification


def test_grounding_clarification_payload_carries_resolver_evidence():
    result = _grounding_issue_fact_result(
        (
            GroundingIssue(
                kind=GroundingTerminalKind.UNRESOLVED_REFERENCE,
                known_input_id="q1_staff_id",
                requested_fact_id="fact_1",
                known_input_text="staff-9999",
                known_input_description="staff",
                proof_refs=("known_input:q1_staff_id",),
                resolver_read_id="get_staff_detail",
                resolver_endpoint_name="get_staff_detail",
                resolver_field_id="staff_id",
                identity_field="staff_id",
            ),
        )
    )

    assert isinstance(result.outcome, NeedsClarification)
    payload = clarification_payload(result.outcome.clarifications[0])
    assert payload["need"] == "target_reference"
    assert payload["reason"] == "unresolved_reference"
    assert payload["subjects"] == [
        {
            "kind": "question_input",
            "id": "q1_staff_id",
            "label": "staff",
            "sourceText": "staff-9999",
            "options": [],
        }
    ]
    assert {
        "kind": "resolver_read",
        "id": "read:get_staff_detail",
        "readId": "get_staff_detail",
        "endpointName": "get_staff_detail",
        "fieldId": "staff_id",
        "identityField": "staff_id",
    } in payload["evidence"]
