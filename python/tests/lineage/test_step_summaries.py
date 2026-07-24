from __future__ import annotations

from fervis.lineage.step_summary import (
    StepSummaryDetail,
    StepSummaryItem,
    StepSemanticItem,
    step_summary_json,
    step_semantic_items_from_json,
    step_summary_items_from_json,
)
from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.lookup.lineage.step_summaries import (
    model_turn_output_summary,
)
from fervis.model_io.turns import ModelTurnPurpose
from fervis.observability.event_contracts import EventPayloadKey


def test_source_binding_model_turn_summary_projects_semantic_mapping_bases() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.SOURCE_BINDING,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "set_bindings": {
                    "fact_1:set:s1": [
                        {
                            "mapping_basis": "Sale rows realize the requested set.",
                            "source_ref": "source_1",
                        }
                    ]
                },
                "fact_bindings": {
                    "fact_1:fact:f1": [
                        {
                            "mapping_basis": "The amount field supplies revenue.",
                            "source_ref": "source_1",
                        }
                    ]
                },
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="Sale rows realize the requested set.",
            is_explanation=True,
            basis="Sale rows realize the requested set.",
            path=("set_bindings", "fact_1:set:s1", "0", "mapping_basis"),
        ),
        StepSummaryItem(
            text="The amount field supplies revenue.",
            is_explanation=True,
            basis="The amount field supplies revenue.",
            path=("fact_bindings", "fact_1:fact:f1", "0", "mapping_basis"),
        ),
    )


def test_conversation_resolution_model_turn_summary_projects_clause_semantics() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.CONVERSATION_RESOLUTION,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "kind": "conversation_resolution",
                "current_question_text": "what about last month?",
                "contextualized_question": (
                    "how many completed in-person sales last month?"
                ),
                "clauses": [
                    {
                        "current_clause_text": "what about last month?",
                        "resolved_text": (
                            "how many completed in-person sales last month?"
                        ),
                        "values": [
                            {
                                "value_id": "time_scope",
                                "resolved_text": "last month",
                            }
                        ],
                    }
                ],
            },
            EventPayloadKey.DERIVED_ARGUMENTS: {},
        }
    )

    assert [item.to_json() for item in step_semantic_items_from_json(summary)] == [
        {
            "kind": "conversation_clause",
            "payload": {
                "current_clause_text": "what about last month?",
                "resolved_text": "how many completed in-person sales last month?",
                "resolved_values": ("last month",),
            },
        }
    ]


def test_model_turn_summary_projects_generic_explanation_fields() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.READ_ELIGIBILITY,
            EventPayloadKey.DERIVED_ARGUMENTS: lineage_explanation_metadata(
                (
                    "requested_fact_assessments",
                    "*",
                    "read_candidate_reviews",
                    "*",
                    "retention_basis",
                ),
            ),
            EventPayloadKey.PARSED_ARGUMENTS: {
                "identity_outcomes": {
                    "grounding_task_1": {
                        "canonical_option_assessments": [],
                        "canonical_option_basis": (
                            "The input denotes the Area primary identity."
                        ),
                        "canonical_option_id": "Area.primary_key",
                        "resolver_route_assessments": [],
                        "resolver_route_basis": (
                            "The route validates the supplied primary key."
                        ),
                        "resolver_route_id": "get_area_detail",
                        "evidence_refs": [],
                        "outcome": "SELECTED",
                    }
                },
                "read_assessments_by_requested_fact": {
                    "fact_1": {
                        "source_1": {
                            "assessment_basis": (
                                "Area rows can ground the named London population scope."
                            ),
                            "relevant_field_refs": ["data.area_id", "data.name"],
                            "decision": "RETAIN",
                        }
                    }
                },
            },
        }
    )

    assert step_summary_items_from_json(summary) == (
        StepSummaryItem(
            text="Read eligibility: retained 1 source candidates, dropped 0.",
        ),
        StepSummaryItem(
            text=(
                "source_1: RETAIN - fields=2 - "
                "Area rows can ground the named London population scope."
            ),
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
            subject="source_1",
            disposition="RETAIN",
            basis="Area rows can ground the named London population scope.",
        ),
    )
    assert step_semantic_items_from_json(summary) == (
        StepSemanticItem(
            kind="identity_selection",
            payload={
                "input_id": "grounding_task_1",
                "canonical_option_id": "Area.primary_key",
                "resolver_route_id": "get_area_detail",
                "basis": (
                    "The input denotes the Area primary identity. "
                    "The route validates the supplied primary key."
                ),
                "outcome": "SELECTED",
            },
        ),
    )


def test_plan_selection_model_turn_summary_projects_reviewed_candidates() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.PLAN_SELECTION,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "source_assessments_by_requested_fact": {
                    "fact_1": {
                        "source_1": {
                            "basis": "Location rows provide the complete fact.",
                            "alignment": "DIRECT",
                        }
                    }
                }
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="Plan selection assessed sources: source_1.",
        ),
        StepSummaryItem(
            text="source_1: DIRECT - Location rows provide the complete fact.",
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
            subject="source_1",
            disposition="DIRECT",
            basis="Location rows provide the complete fact.",
        ),
    )


def test_question_contract_summary_projects_semantic_requested_facts_and_known_inputs() -> (
    None
):
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.QUESTION_CONTRACT,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "decision_basis": "The question requests one result.",
                "outcome": {
                    "kind": "question_meaning",
                    "answer_requests": [
                        {
                            "result_kind": "scalar",
                            "qualifying_row_kind": {
                                "meaning": "sales",
                                "origin": {
                                    "kind": "question",
                                    "resolved_input_ref": None,
                                },
                            },
                            "grouping_meanings": [],
                            "return_request_basis": (
                                "The question asks for the number of sales."
                            ),
                            "returned_result": {"kind": "values"},
                            "answer_values": [
                                {
                                    "value_ref": "v1",
                                    "meaning": "sale count",
                                    "origin": {
                                        "kind": "question",
                                        "resolved_input_ref": None,
                                    },
                                }
                            ],
                            "returned_value_refs": ["v1"],
                            "ordering_value_refs": [],
                            "selection": {"kind": "all_results", "limit": None},
                            "universal_shape": "none",
                        },
                    ],
                    "supplied_values": [
                        {
                            "meaning": "the named store",
                            "denotation": {
                                "basis": "ABC Mall names a store.",
                                "kind": "identity_reference",
                                "instance_kind": "store",
                            },
                            "value": {
                                "operands": ["ABC Mall"],
                                "value_type": {"kind": "identity_name_or_code"},
                                "origin": {
                                    "kind": "question",
                                    "resolved_input_ref": None,
                                },
                            },
                        },
                        {
                            "meaning": "the reporting period",
                            "denotation": {
                                "basis": "this month states a time interval.",
                                "kind": "scalar",
                                "instance_kind": None,
                            },
                            "value": {
                                "operands": ["this month"],
                                "value_type": {"kind": "temporal_scope"},
                                "origin": {
                                    "kind": "resolved_context",
                                    "resolved_input_ref": "time_scope",
                                },
                            },
                        },
                    ],
                },
            },
        }
    )

    assert step_semantic_items_from_json(summary) == (
        StepSemanticItem(
            kind="requested_fact",
            payload={
                "requested_fact_id": "fact_1",
                "description": "The question asks for the number of sales.",
            },
        ),
        StepSemanticItem(
            kind="known_input",
            payload={
                "input_id": "ABC Mall",
                "text": "ABC Mall",
                "kind": "IDENTITY_REFERENCE",
                "role": "",
                "description": "the named store",
                "resolved_value_text": "ABC Mall",
            },
        ),
        StepSemanticItem(
            kind="known_input",
            payload={
                "input_id": "time_scope",
                "text": "this month",
                "kind": "NON_IDENTITY_SCALAR",
                "role": "",
                "description": "the reporting period",
                "resolved_value_text": "this month",
            },
        ),
    )


def test_enrichment_and_grounding_summaries_project_semantic_resolver_records() -> None:
    enrichment_summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.QUERY_ENRICHMENT,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "recall_bucket_matches": [],
                "input_resource_search_terms": [
                    {
                        "input_use_ref": "fact_1:input_use:e1:1",
                        "catalog_search_terms": ["location"],
                    }
                ],
            },
        }
    )
    grounding_summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.GROUNDING,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "time_resolutions": {},
                "reference_reviews": {
                    "grounding_task_1": {
                        "identifier_kind_basis": "ABC Mall is a descriptive name.",
                        "identifier_kind": "DESCRIPTIVE",
                        "resource_type_reviews": {
                            "location": {
                                "compatibility_basis": (
                                    "A location could be the place named ABC Mall."
                                ),
                                "compatibility": "POSSIBLE_DENOTED_KIND",
                                "route_reviews": {
                                    "bind_fact_1_entity_1_1": {
                                        "assessment_basis": (
                                            "The resolver can search location records "
                                            "by the provided lookup text."
                                        ),
                                        "resolution": {
                                            "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                            "lookup_request_params": ["name"],
                                            "returned_identity_verification_fields": [
                                                "name"
                                            ],
                                        },
                                    }
                                },
                            }
                        },
                    }
                },
            },
        }
    )

    assert step_semantic_items_from_json(enrichment_summary) == (
        StepSemanticItem(
            kind="resource_recall",
            payload={
                "input_use_ref": "fact_1:input_use:e1:1",
                "resource_name": "location",
            },
        ),
    )
    assert step_semantic_items_from_json(grounding_summary) == (
        StepSemanticItem(
            kind="resolver_candidate",
            payload={
                "input_id": "grounding_task_1",
                "resolver_read_id": "bind_fact_1_entity_1_1",
                "resolver_label": "Bind Fact 1 Entity 1 1",
                "basis": (
                    "The resolver can search location records "
                    "by the provided lookup text."
                ),
            },
        ),
    )


def test_grounding_summary_projects_time_interpretations_as_semantic_inputs() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.GROUNDING,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "reference_reviews": {},
                "time_resolutions": {
                    "time_task_1": {
                        "date_intent": {
                            "expression": "this month",
                            "intent": {"time_shape": "period_named", "unit": "month"},
                        }
                    }
                },
            },
        }
    )

    assert step_semantic_items_from_json(summary) == (
        StepSemanticItem(
            kind="interpreted_input",
            payload={
                "input_id": "time_task_1",
                "input_text": "this month",
                "kind": "time",
                "value": "this month",
                "label": "this month",
                "detail": "month",
            },
        ),
    )
