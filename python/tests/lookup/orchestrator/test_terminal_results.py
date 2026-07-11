from fervis.lookup.grounding.model import (
    GroundingCandidate,
    GroundingIssue,
    GroundingTerminalKind,
)
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


def test_grounding_issues_for_one_input_produce_one_complete_clarification():
    result = _grounding_issue_fact_result(
        tuple(
            GroundingIssue(
                kind=GroundingTerminalKind.AMBIGUOUS_REFERENCE,
                known_input_id="q1_store",
                requested_fact_id="fact_1",
                known_input_text="Central",
                known_input_description="store",
                candidate_options=(
                    GroundingCandidate(
                        id=candidate_id,
                        label=candidate_id,
                        resolver_read_id=resolver_read_id,
                    ),
                ),
                resolver_read_id=resolver_read_id,
                resolver_endpoint_name=resolver_read_id,
            )
            for candidate_id, resolver_read_id in (
                ("store_1", "list_stores"),
                ("location_1", "list_locations"),
            )
        )
    )

    assert isinstance(result.outcome, NeedsClarification)
    assert len(result.outcome.clarifications) == 1
    payload = clarification_payload(result.outcome.clarifications[0])
    assert [
        option["id"] for option in payload["subjects"][0]["options"]
    ] == ["store_1", "location_1"]
    assert [
        item["readId"]
        for item in payload["evidence"]
        if item["kind"] == "resolver_read"
    ] == [
        "list_stores",
        "list_locations",
    ]
