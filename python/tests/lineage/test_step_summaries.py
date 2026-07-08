from __future__ import annotations

from fervis.lineage.step_summary import (
    StepSummaryDetail,
    StepSummaryItem,
    StepSemanticItem,
    step_summary_json,
    step_semantic_items_from_json,
)
from fervis.lookup.fact_plan.values import FactValue, LiteralType
from fervis.lookup.grounding.model import CanonicalInputLedger
from fervis.lookup.lineage.explanation_metadata import (
    lineage_explanation_metadata,
)
from fervis.lookup.lineage.step_summaries import (
    add_grounding_result_semantics,
    model_turn_output_summary,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)
from fervis.model_io.turns import ModelTurnPurpose
from fervis.observability.event_contracts import EventPayloadKey


def test_source_binding_model_turn_summary_projects_metric_fit_basis() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.SOURCE_BINDING,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "outcome": {
                    "kind": "source_bindings",
                    "metric_fit_bases": {
                        "fact_1": {
                            "metric_1": {
                                "fit_basis": "row-level payroll amount",
                            }
                        }
                    },
                    "fit_basis_interpretations": {
                        "fact_1": {
                            "metric_1": {
                                "interpretation": "FITS_REQUESTED_ANSWER",
                            }
                        }
                    },
                },
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="metric_1: row-level payroll amount -> FITS_REQUESTED_ANSWER",
            is_explanation=True,
            path=("outcome", "metric_fit_bases", "fact_1", "metric_1", "fit_basis"),
        )
    )


def test_conversation_resolution_model_turn_summary_projects_clause_semantics() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.CONVERSATION_RESOLUTION,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "kind": "conversation_resolution",
                "current_question_text": "what about last month?",
                "clause_resolutions": [
                    {
                        "current_clause_text": "what about last month?",
                        "requested_value_frame": {
                            "current_value_surface": {
                                "text": "what about last month?",
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
                        "dependencies": [],
                        "resolved_clause_text": (
                            "how many completed in-person sales last month?"
                        ),
                    }
                ],
            },
            EventPayloadKey.DERIVED_ARGUMENTS: {
                "value_frames": [
                    {
                        "current_clause_text": "what about last month?",
                        "current_value_text": "what about last month?",
                        "current_value_kind": "broad_current_value",
                        "resolved_frame_text": "count of completed in-person sales",
                        "must_preserve_terms": ["completed in-person sales"],
                        "used_context_frame_ids": ["context_frame_1"],
                    }
                ]
            },
        }
    )

    assert [item.to_json() for item in step_semantic_items_from_json(summary)] == [
        {
            "kind": "conversation_clause",
            "payload": {
                "current_clause_text": "what about last month?",
                "current_value_text": "what about last month?",
                "resolved_frame_text": "count of completed in-person sales",
                "resolved_clause_text": "how many completed in-person sales last month?",
            },
        }
    ]


def test_source_binding_model_turn_summary_projects_decision_basis() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.SOURCE_BINDING,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "outcome": {
                    "kind": "source_bindings",
                    "metric_fit_bases": {},
                    "fit_basis_interpretations": {},
                        "source_invocations": [
                            {
                                "binding_target_id": "target.source_5",
                                "answer_population": {
                                    "match_basis_explanation": (
                                        "Payroll summary rows match the requested "
                                    "staff population."
                                )
                            },
                            "fulfillment_decisions": {
                                "answer_1": {
                                    "fulfillment_choice_id": "choice_staff_name",
                                    "match_basis_explanation": (
                                        "staff_name identifies the returned staff."
                                    ),
                                }
                            },
                            "param_decisions": {
                                "month": {
                                    "param_decision_id": "param_month",
                                    "match_basis_explanation": (
                                        "The requested month must bind the month param."
                                    ),
                                }
                            },
                        }
                    ],
                },
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="Source binding target.source_5",
            detail=StepSummaryDetail.VERBOSE,
        ),
        StepSummaryItem(
            text="Population basis: Payroll summary rows match the requested staff population.",
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
            path=("answer_population", "match_basis_explanation"),
        ),
        StepSummaryItem(
            text=(
                "Fulfillment basis answer_1/choice_staff_name: "
                "staff_name identifies the returned staff."
            ),
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
            path=(
                "fulfillment_decisions",
                "answer_1",
                "match_basis_explanation",
            ),
        ),
        StepSummaryItem(
            text=(
                "Param basis month/param_month: "
                "The requested month must bind the month param."
            ),
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
            path=("param_decisions", "month", "match_basis_explanation"),
        ),
    )


def test_model_turn_summary_projects_generic_explanation_fields() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.READ_ELIGIBILITY,
            EventPayloadKey.DERIVED_ARGUMENTS: lineage_explanation_metadata(
                ("read_candidate_reviews", "*", "retention_basis"),
            ),
            EventPayloadKey.PARSED_ARGUMENTS: {
                "read_candidate_reviews": [
                    {
                        "source_candidate_id": "source_1",
                        "read_id": "list_area_list",
                        "retention_basis": (
                            "Area rows can ground the named London population scope."
                        ),
                        "relevant_row_path_tokens": ["source_1.row"],
                        "relevant_field_tokens": ["name"],
                        "decision": "RETAIN",
                    }
                ]
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="Read eligibility: retained 1 source candidates, dropped 0.",
        ),
        StepSummaryItem(
            text=(
                "source_1 list_area_list: RETAIN - rows=1 - fields=1 - "
                "Area rows can ground the named London population scope."
            ),
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
        ),
    )


def test_plan_selection_model_turn_summary_projects_reviewed_candidates() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.PLAN_SELECTION,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "outcome": {
                    "kind": "source_alignment_reviews",
                    "reviews_by_requested_fact": {
                        "fact_1": {
                            "source_1": {
                                "source_candidate_id": "source_1",
                                "basis": (
                                    "Location rows do not expose compensation measures."
                                ),
                                "source_alignment": "NOT_ALIGNED",
                            },
                            "source_8": {
                                "source_candidate_id": "source_8",
                                "basis": (
                                    "Payout rows expose amounts but no location key."
                                ),
                                "source_alignment": "PARTIAL",
                            },
                        }
                    },
                }
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="Plan selection reviewed source candidates: source_1, source_8.",
        ),
        StepSummaryItem(
            text=(
                "source_1: NOT_ALIGNED - "
                "Location rows do not expose compensation measures."
            ),
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
        ),
        StepSummaryItem(
            text="source_8: PARTIAL - Payout rows expose amounts but no location key.",
            detail=StepSummaryDetail.VERBOSE,
            is_explanation=True,
        ),
    )


def test_question_contract_summary_projects_semantic_requested_facts_and_known_inputs() -> (
    None
):
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.QUESTION_CONTRACT,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "kind": "question_contract",
                "question_inputs": [
                    {
                        "input_ref": "fact_1_entity_1",
                        "kind": "literal_text",
                        "role": "reference_value",
                        "source_text": "ABC Mall",
                        "value_meaning_hint": "store",
                        "resolved_value_text": "ABC Mall",
                    },
                    {
                        "input_ref": "fact_1_time_1",
                        "kind": "literal_text",
                        "role": "time_value",
                        "source_text": "this month",
                        "resolved_value_text": "this month",
                    },
                ],
                "answer_requests": [
                    {
                        "answer_fact": "sales at ABC Mall this month",
                    }
                ],
            },
        }
    )

    assert step_semantic_items_from_json(summary) == (
        StepSemanticItem(
            kind="requested_fact",
            payload={
                "requested_fact_id": "fact_1",
                "description": "sales at ABC Mall this month",
            },
        ),
        StepSemanticItem(
            kind="known_input",
            payload={
                "input_id": "fact_1_entity_1",
                "text": "ABC Mall",
                "kind": "literal_text",
                "role": "reference_value",
                "description": "store",
                "resolved_value_text": "ABC Mall",
            },
        ),
        StepSemanticItem(
            kind="known_input",
            payload={
                "input_id": "fact_1_time_1",
                "text": "this month",
                "kind": "literal_text",
                "role": "time_value",
                "description": "",
                "resolved_value_text": "this month",
            },
        ),
    )


def test_enrichment_and_grounding_summaries_project_semantic_resolver_records() -> None:
    enrichment_summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.QUERY_ENRICHMENT,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "entity_target_catalog_search_terms": [
                    {
                        "target_id": "fact_1_entity_1",
                        "catalog_search_terms": [
                            {
                                "term": "location",
                                "basis": (
                                    "location can identify ABC Mall because "
                                    "target meaning is store or location."
                                ),
                            },
                        ],
                    }
                ]
            },
        }
    )
    grounding_summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.GROUNDING,
            EventPayloadKey.PARSED_ARGUMENTS: {
                "known_input_binding_reviews": {
                    "fact_1_entity_1": {
                        "option_reviews": {
                            "bind_fact_1_entity_1_1": {
                                "because": (
                                    "The resolver can search location records "
                                    "by the provided lookup text."
                                ),
                                "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                            }
                        }
                    }
                }
            },
        }
    )

    assert step_semantic_items_from_json(enrichment_summary) == (
        StepSemanticItem(
            kind="resolver_candidate",
            payload={
                "input_id": "fact_1_entity_1",
                "resolver_read_id": "",
                "resolver_label": "Location",
                "basis": (
                    "location can identify ABC Mall because "
                    "target meaning is store or location."
                ),
            },
        ),
    )
    assert step_semantic_items_from_json(grounding_summary) == (
        StepSemanticItem(
            kind="resolver_candidate",
            payload={
                "input_id": "fact_1_entity_1",
                "resolver_read_id": "",
                "resolver_label": "",
                "basis": (
                    "The resolver can search location records "
                    "by the provided lookup text."
                ),
            },
        ),
    )


def test_grounding_summary_projects_time_interpretations_as_semantic_inputs() -> None:
    summary = add_grounding_result_semantics(
        {},
        ledger=CanonicalInputLedger(
            values=(
                FactValue.time(
                    id="value_time_1",
                    expression="this month",
                    resolved_start="2026-06-01",
                    resolved_end="2026-06-30",
                    granularity="month",
                    proof_refs=("known_input:fact_1_time_1",),
                ),
            )
        ),
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="sales at ABC Mall this month",
                    answer_outputs=(RequestedFactAnswerOutput("answer_1"),),
                    known_inputs=(
                        RequestedFactLiteralInput(
                            id="fact_1_time_1",
                            source=KnownInputSource.QUESTION_CONTEXT,
                            text="this month",
                            role=LiteralInputRole.TIME_VALUE,
                            resolved_value_text="this month",
                        ),
                    ),
                ),
            )
        ),
    )

    assert step_semantic_items_from_json(summary) == (
        StepSemanticItem(
            kind="interpreted_input",
            payload={
                "input_id": "fact_1_time_1",
                "input_text": "this month",
                "kind": "time",
                "value": "2026-06-01 to 2026-06-30",
                "label": "this month",
                "detail": "month",
            },
        ),
    )


def test_grounding_summary_projects_literal_interpretations_as_semantic_inputs() -> (
    None
):
    summary = add_grounding_result_semantics(
        {},
        ledger=CanonicalInputLedger(
            values=(
                FactValue.literal(
                    id="value_limit_1",
                    literal_type=LiteralType.NUMBER,
                    value="10",
                    label="top 10",
                    proof_refs=("known_input:fact_1_limit_1",),
                ),
            )
        ),
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="top 10 salespeople this month",
                    answer_outputs=(RequestedFactAnswerOutput("answer_1"),),
                    known_inputs=(
                        RequestedFactLiteralInput(
                            id="fact_1_limit_1",
                            source=KnownInputSource.QUESTION_CONTEXT,
                            text="top 10",
                            role=LiteralInputRole.RESULT_LIMIT,
                            resolved_value_text="10",
                            value_meaning_hint="rank limit",
                        ),
                    ),
                ),
            )
        ),
    )

    assert step_semantic_items_from_json(summary) == (
        StepSemanticItem(
            kind="interpreted_input",
            payload={
                "input_id": "fact_1_limit_1",
                "input_text": "top 10",
                "kind": "literal_number",
                "value": "10",
                "label": "top 10",
                "detail": "",
            },
        ),
    )


def test_fact_planning_model_turn_summary_projects_selected_binding() -> None:
    summary = model_turn_output_summary(
        {
            EventPayloadKey.PURPOSE: ModelTurnPurpose.FACT_PLAN,
            EventPayloadKey.ARGUMENTS: {
                "outcome": {
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "metric": {"field_id": "calculated_pay"},
                            "function": {"value": "sum"},
                        }
                    ]
                }
            },
        }
    )

    assert summary == step_summary_json(
        StepSummaryItem(
            text="Binding: metric=calculated_pay function=sum"
        )
    )
