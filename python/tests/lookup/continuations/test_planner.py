from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from fervis.lookup.continuations import (
    ContinuationPlanKind,
    derive_continuation_plan,
)
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    parse_conversation_resolution,
)
from fervis.lookup.memory.projection import project_conversation_memory_cards
from fervis.memory.addresses import EvidenceRef, FactAddress
from fervis.memory.artifacts import FactOutcome, build_fact_artifact


def test_shape_changing_continuation_carries_prior_inputs_for_inspection():
    projection = _sales_count_projection()
    resolution = _parse_continuation(
        projection=projection,
        current_question="How about locations?",
        replacement_part_id="answer_subject",
        replacement_text="locations",
        resolved_clause_text="count of locations in Nairobi this month",
    )

    plan = derive_continuation_plan(
        resolution=resolution.outcome,
        memory_projection=projection,
    )

    assert plan.kind == ContinuationPlanKind.SHAPE_CHANGING
    assert [item.to_payload() for item in plan.replacements] == [
        {
            "part_id": "answer_subject",
            "kind": "answer_subject",
            "text": "sales",
            "prior_text": "sales",
            "current_text": "locations",
        }
    ]
    assert [item.to_payload() for item in plan.carried_inputs] == [
        {
            "part_id": "q_place",
            "kind": "entity_identity",
            "text": "Nairobi",
            "resolved_value_text": "Nairobi",
            "field_label_text": "location",
            "value_meaning_hint": "location identity",
            "binding": {
                "value_kind": "entity_identity",
                "source_lineage": ["turn_sales.entity.q_place"],
                "display": "Nairobi",
                "identity_type": "location",
                "canonical_values": {"location_id": "loc_nairobi"},
            },
        },
        {
            "part_id": "q_time",
            "kind": "time_scope",
            "text": "this month",
            "resolved_value_text": "this month",
            "value_meaning_hint": "time scope",
            "binding": {
                "value_kind": "time_scope",
                "source_lineage": ["turn_sales.value.q_time"],
                "value": "this month",
                "display": "this month",
                "resolved_start": "2026-07-01",
                "resolved_end": "2026-07-31",
                "granularity": "month",
            },
        },
    ]


def test_same_fact_input_replacement_is_reusable_when_lineage_is_complete():
    projection = _sales_count_projection()
    resolution = _parse_continuation(
        projection=projection,
        current_question="What about Mombasa?",
        replacement_part_id="q_place",
        replacement_text="Mombasa",
        resolved_clause_text="sales count in Mombasa this month",
    )

    plan = derive_continuation_plan(
        resolution=resolution.outcome,
        memory_projection=projection,
    )

    assert plan.kind == ContinuationPlanKind.SAME_FACT_INPUT_REPLACEMENT
    assert [item.part_id for item in plan.carried_inputs] == ["q_time"]


def test_continuation_semantics_do_not_depend_on_private_card_serialization():
    projection = _sales_count_projection()
    private_cards = deepcopy(projection.private_cards)
    assert private_cards is not None
    prior_request = projection.prior_requests[0]
    private_cards[prior_request.memory_id]["request_shape"]["slots"] = {
        "presentation": "changed"
    }
    projection = replace(projection, private_cards=private_cards)
    resolution = _parse_continuation(
        projection=projection,
        current_question="How about locations?",
        replacement_part_id="answer_subject",
        replacement_text="locations",
        resolved_clause_text="count of locations in Nairobi this month",
    )

    plan = derive_continuation_plan(
        resolution=resolution.outcome,
        memory_projection=projection,
    )

    assert [item.part_id for item in plan.carried_inputs] == ["q_place", "q_time"]


def _parse_continuation(
    *,
    projection,
    current_question: str,
    replacement_part_id: str,
    replacement_text: str,
    resolved_clause_text: str,
):
    return parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload={
            "kind": "conversation_resolution",
            "current_question_text": current_question,
            "clause_resolutions": [
                {
                    "current_clause_text": current_question,
                    "occurrence": 1,
                    "requested_value_frame": {
                        "current_value_surface": {
                            "text": replacement_text,
                            "kind": "broad_current_value",
                        },
                        "context_frame_choices": [
                            {
                                "frame_id": "context_frame_1",
                                "choice": "use_frame",
                                "current_conflict_quotes": [],
                            }
                        ],
                    },
                    "continuation": {
                        "kind": "continue_prior_question",
                        "frame_id": "context_frame_1",
                        "replacements": [
                            {
                                "part_id": replacement_part_id,
                                "current_text": replacement_text,
                            }
                        ],
                    },
                    "dependencies": [],
                    "resolved_clause_text": resolved_clause_text,
                }
            ],
            "unresolved": {
                "unresolved_kind": "none",
                "why_unresolved": "",
                "candidate_interpretations": [],
            },
        },
        current_question=current_question,
        context_frames=projection.context_frames,
        context_sources=projection.context_sources,
    )


def _sales_count_projection():
    artifact = build_fact_artifact(
        artifact_id="turn_sales",
        outcome=FactOutcome.ANSWERED,
        source_question="How many sales did we make in Nairobi this month?",
        source_answer="8",
        provenance={
            "question_contract": {
                "kind": "question_contract",
                "answer_requests_count": 1,
                "question_inputs": [
                    {
                        "id": "q_place",
                        "kind": "literal_text",
                        "role": "reference_value",
                        "text": "Nairobi",
                        "resolved_value_text": "Nairobi",
                        "field_label_text": "location",
                        "value_meaning_hint": "location identity",
                    },
                    {
                        "id": "q_time",
                        "kind": "literal_text",
                        "role": "time_value",
                        "text": "this month",
                        "resolved_value_text": "this month",
                        "value_meaning_hint": "time scope",
                    },
                ],
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "sales count in Nairobi this month",
                        "answer_subject": {"subject_text": "sales"},
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "sales count",
                            }
                        ],
                        "used_question_inputs": ["q_place", "q_time"],
                    }
                ],
            }
        },
        addresses=tuple(
            item
            for item in (
                FactAddress.entity(
                    address="entity.q_place",
                    resource="location",
                    reference_text="Nairobi",
                    identity={"location_id": "loc_nairobi"},
                    evidence=EvidenceRef(step_ids=("known_input:q_place",)),
                ),
                FactAddress.value(
                    address="value.q_time",
                    value={
                        "type": "time_scope",
                        "value": "this month",
                        "expression": "this month",
                        "resolvedStart": "2026-07-01",
                        "resolvedEnd": "2026-07-31",
                        "granularity": "month",
                    },
                    display="this month",
                    evidence=EvidenceRef(step_ids=("known_input:q_time",)),
                ),
                FactAddress.value(
                    address="value.answer_1",
                    value={
                        "type": "integer",
                        "value": 8,
                        "answer_output_ids": ["answer_1"],
                    },
                ),
            )
        ),
    )
    return project_conversation_memory_cards(
        {"factArtifacts": [artifact.to_dict()]},
        current_question="What about Mombasa?",
    )
