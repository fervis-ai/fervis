from __future__ import annotations

import pytest

from fervis.lookup.question_contract import (
    KnownInputKind,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
    RequestedFactKnownInput,
    parse_question_contract,
)


def _time_input(input_id: str, text: str) -> RequestedFactKnownInput:
    return RequestedFactKnownInput(
        id=input_id,
        kind=KnownInputKind.LITERAL,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=text,
        role=LiteralInputRole.TIME_VALUE,
        satisfies_requirement_id=f"{input_id}_req",
    )


def test_requested_fact_answer_output_serializes_description_only():
    output = RequestedFactAnswerOutput(
        id="answer_1",
        description="the amount she made yesterday",
    )
    fact = RequestedFact(
        id="fact_1",
        description="total sales amount for Alice yesterday",
        answer_outputs=(output,),
    )

    assert output.to_model_dict() == {
        "id": "answer_1",
        "description": "the amount she made yesterday",
    }
    assert fact.answer_request_model_dict()["answer_outputs"] == [
        {
            "description": "the amount she made yesterday",
        }
    ]


def test_question_contract_model_materializes_input_refs_as_fact_known_inputs():
    period = RequestedFactKnownInput(
        id="period",
        kind=KnownInputKind.LITERAL,
        source=KnownInputSource.QUESTION_CONTEXT,
        text="yesterday",
        resolved_value_text="yesterday",
        role=LiteralInputRole.TIME_VALUE,
        satisfies_requirement_id="period_req",
    )
    contract = QuestionContract(
        question_inputs=(period,),
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales yesterday",
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
                ),
                answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
                input_refs=("period",),
            ),
        ),
    )

    assert contract.requested_facts[0].known_inputs == (period,)


def test_question_contract_serializes_fact_local_known_inputs_as_question_inputs():
    period = _time_input("period", "yesterday")
    contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales yesterday",
                answer_expression=RequestedFactAnswerExpression(
                    family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
                ),
                answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
                known_inputs=(period,),
            ),
        ),
    )

    assert contract.to_model_dict() == {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "answer_requests": [
            {
                "id": "fact_1",
                "answer_fact": "sales yesterday",
                "answer_expression": {"family": "scalar_aggregate"},
                "input_requirements": {"time_requirements": []},
                "answer_outputs": [
                    {
                        "description": "answer_1",
                    }
                ],
                "input_decisions": [
                    {
                        "input_ref": "period",
                        "use_input": True,
                    }
                ],
            }
        ],
        "question_inputs": [
            {
                "id": "period",
                "kind": "literal_text",
                "source": "question_context",
                "text": "yesterday",
                "satisfies_requirement_id": "period_req",
                "resolved_value_text": "yesterday",
                "role": "time_value",
            }
        ],
    }


def test_question_contract_parser_fails_closed_on_unparsed_fields():
    payload = {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
        "requested_facts": [
            {
                "id": "model_authored_fact",
            }
        ],
        "answer_requests": [
            {
                "answer_fact": "sales today",
                "answer_expression": {
                    "family": "scalar_aggregate",
                    "extra": "not part of the contract",
                },
                "answer_subject": {
                    "subject_text": "sales",
                    "instance_interpretation": {
                        "kind": "NORMAL_BUSINESS_INSTANCE",
                    },
                },
                "input_requirements": {"time_requirements": []},
                "answer_population": {
                    "population_label": "sales",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                        }
                    ],
                },
                "answer_outputs": [{"description": "sales count"}],
                "input_decisions": [],
                "resolver_choice": "not part of the contract",
            }
        ],
    }

    with pytest.raises(ValueError, match="unparsed"):
        parse_question_contract(
            tool_name="submit_answer_request_contract",
            payload=payload,
            question_context="How many sales today?",
        )


def test_row_set_reference_known_input_cannot_carry_literal_fields():
    with pytest.raises(ValueError, match="row set reference"):
        RequestedFactKnownInput(
            id="prior_rows",
            kind=KnownInputKind.ROW_SET_REFERENCE,
            source=KnownInputSource.CONVERSATION_RESOLUTION,
            text="those sales",
            occurrence=1,
            resolved_input_ref="cr_input_1",
            role=LiteralInputRole.TIME_VALUE,
            resolved_value_text="today",
        )


def test_conversation_resolution_resolved_text_requires_conversation_source():
    payload = {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            {
                "input_ref": "input_staff",
                "kind": "literal_text",
                "source": "question_context",
                "source_text": "Alice Smith",
                "resolved_value_text": "Alice Smith",
                "role": "reference_value",
                "inventory_check": {
                    "why_this_is_an_input": "Alice Smith is the resolved staff value"
                },
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
        "answer_requests": [
            {
                "answer_fact": "her sales",
                "answer_expression": {"family": "scalar_aggregate"},
                "answer_subject": {
                    "subject_text": "sales",
                    "instance_interpretation": {
                        "kind": "NORMAL_BUSINESS_INSTANCE",
                    },
                },
                "input_requirements": {"time_requirements": []},
                "answer_population": {
                    "population_label": "her sales",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                        }
                    ],
                },
                "answer_outputs": [{"description": "sales total"}],
                "input_decisions": [
                    {
                        "input_ref": "input_staff",
                        "use_input": True,
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="source_text"):
        parse_question_contract(
            tool_name="submit_answer_request_contract",
            payload=payload,
            question_context="What were her sales?",
            question_context_texts=("Alice Smith",),
        )
