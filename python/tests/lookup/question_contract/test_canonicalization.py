from __future__ import annotations

import pytest

from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactGroupKey,
    RequestedFactAnswerOutput,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
    ResultSelectionKind,
    build_answer_request_contract_schema,
    parse_question_contract,
)


def _decision_payload(outcome: dict[str, object]) -> dict[str, object]:
    return {
        "decision_basis": "The current wording identifies the requested fact.",
        "outcome": outcome,
    }


def _single_input_payload(question_input: dict[str, object]) -> dict[str, object]:
    input_ref = str(question_input["input_ref"])
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [question_input],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
        "answer_requests": [
            {
                "answer_fact": "sales for the specified input",
                "answer_expression": {"family": "scalar_aggregate"},
                "answer_subject": {
                    "subject_text": "sales",
                    "instance_interpretation": {
                        "kind": "NORMAL_BUSINESS_INSTANCE",
                    },
                },
                "answer_population": {
                    "population_label": "sales for specified input",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                            "owned_question_input_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "sales total", "role": "ANSWER_VALUE"}
                ],
                "used_question_inputs": [input_ref],
            }
        ],
    }


def _time_input(input_id: str, text: str) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=text,
        role=LiteralInputRole.TIME_VALUE,
    )


def test_requested_fact_answer_output_serializes_its_computation_role():
    output = RequestedFactAnswerOutput(
        id="answer_1",
        description="the amount she made yesterday",
        role="MEASURED_VALUE",
    )
    fact = RequestedFact(
        id="fact_1",
        description="total sales amount for Alice yesterday",
        answer_outputs=(output,),
    )

    assert output.to_model_dict() == {
        "id": "answer_1",
        "description": "the amount she made yesterday",
        "role": "MEASURED_VALUE",
    }
    assert fact.answer_request_model_dict()["answer_outputs"] == [
        {
            "description": "the amount she made yesterday",
            "role": "MEASURED_VALUE",
        }
    ]


def test_answer_output_schema_requires_a_computation_role():
    schema = build_answer_request_contract_schema()
    answer_output_schema = schema["properties"]["answer_requests"]["items"][
        "properties"
    ]["answer_outputs"]["items"]

    assert answer_output_schema["required"] == ["description", "role"]


def test_ranked_selection_remains_one_catalog_blind_answer_shape():
    question = "Which salesperson made the most revenue today?"
    payload = _single_input_payload(
        {
            "input_ref": "input_today",
            "kind": "literal_text",
            "source": "question_context",
            "value_source_text": "today",
            "operand_text": "today",
            "role": "time_value",
            "inventory_check": {
                "why_this_is_an_input": "today constrains the requested result"
            },
        }
    )
    payload["answer_requests"][0]["answer_expression"] = {
        "family": "ranked_selection"
    }

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=question,
    )

    expression = parsed.outcome.requested_facts[0].answer_expression

    assert expression is not None
    assert expression.family is RequestedFactAnswerExpressionFamily.RANKED_SELECTION
    assert expression.selection_kind is ResultSelectionKind.LIMITED_RESULTS


def test_grouped_aggregate_requires_expression_group_key():
    with pytest.raises(ValueError, match="grouped_aggregate requires group key"):
        RequestedFact(
            id="fact_1",
            description="sales count by store",
            answer_expression=RequestedFactAnswerExpression(
                family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            ),
            answer_outputs=(
                RequestedFactAnswerOutput(
                    id="answer_count",
                    description="sales count",
                    role="ROW_COUNT",
                ),
            ),
        )


def test_grouped_aggregate_provider_schema_requires_expression_group_key():
    schema = build_answer_request_contract_schema()
    answer_expression_schema = schema["properties"]["answer_requests"]["items"][
        "properties"
    ]["answer_expression"]
    grouped_branch = next(
        branch
        for branch in answer_expression_schema["oneOf"]
        if branch["properties"]["family"]["enum"] == ["grouped_aggregate"]
    )

    assert grouped_branch["required"] == ["family", "group_key"]


def test_grouped_aggregate_serializes_group_key_on_answer_expression():
    fact = RequestedFact(
        id="fact_1",
        description="sales count for each specified staff member today",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            group_key=RequestedFactGroupKey(
                description="staff member",
                domain=GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS,
                question_input_refs=("qi_staff_1", "qi_staff_2"),
            ),
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_count",
                description="sales count",
                role="ROW_COUNT",
            ),
        ),
        input_refs=("qi_staff_1", "qi_staff_2"),
    )

    assert fact.answer_request_model_dict()["answer_expression"] == {
        "family": "grouped_aggregate",
        "group_key": {
            "description": "staff member",
            "domain": "SPECIFIED_QUESTION_INPUTS",
            "question_input_refs": ["qi_staff_1", "qi_staff_2"],
        },
    }
    assert fact.answer_request_model_dict()["answer_outputs"] == [
        {
            "description": "sales count",
            "role": "ROW_COUNT",
        }
    ]


def test_parse_question_contract_accepts_group_key_on_grouped_expression():
    payload = {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            {
                "source": "question_context",
                "role": "reference_value",
                "kind": "literal_text",
                "input_ref": "qi_staff_1",
                "value_source_text": "51515151-0000-0000-0002-000000000001",
                "operand_text": "51515151-0000-0000-0002-000000000001",
                "field_label_text": "staff id",
                "occurrence": 1,
                "inventory_check": {
                    "why_this_is_an_input": "first requested staff key",
                },
            },
            {
                "source": "question_context",
                "role": "reference_value",
                "kind": "literal_text",
                "input_ref": "qi_staff_2",
                "value_source_text": "51515151-0000-0000-0002-000000000002",
                "operand_text": "51515151-0000-0000-0002-000000000002",
                "field_label_text": "staff id",
                "occurrence": 1,
                "inventory_check": {
                    "why_this_is_an_input": "second requested staff key",
                },
            },
            {
                "source": "question_context",
                "role": "time_value",
                "kind": "literal_text",
                "input_ref": "qi_today",
                "value_source_text": "today",
                "operand_text": "today",
                "occurrence": 1,
                "inventory_check": {
                    "why_this_is_an_input": "requested time constraint",
                },
            },
        ],
        "answer_requests": [
            {
                "answer_fact": "sales count for each specified staff member today",
                "answer_expression": {
                    "family": "grouped_aggregate",
                    "group_key": {
                        "description": "staff member",
                        "domain": "SPECIFIED_QUESTION_INPUTS",
                        "question_input_refs": ["qi_staff_1", "qi_staff_2"],
                    },
                },
                "answer_subject": {
                    "subject_text": "sales",
                    "instance_interpretation": {
                        "kind": "NORMAL_BUSINESS_INSTANCE",
                    },
                },
                "answer_population": {
                    "population_label": "sales today",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                            "owned_question_input_refs": [],
                        },
                        {
                            "test_id": "test_2",
                            "kind": "EXPLICIT_USER_CONSTRAINT",
                            "polarity": "MUST_PASS",
                            "test_question": "Did the sale occur today?",
                            "owned_question_input_refs": ["qi_today"],
                        },
                        {
                            "test_id": "test_3",
                            "kind": "NORMAL_INSTANCE_GUARD",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this an ordinary business sale?",
                            "owned_question_input_refs": [],
                        },
                    ],
                },
                "answer_outputs": [{"description": "sales count", "role": "ROW_COUNT"}],
                "used_question_inputs": ["qi_staff_1", "qi_staff_2", "qi_today"],
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
    }

    result = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=(
            "How many sales did the staff members with ids: "
            "51515151-0000-0000-0002-000000000001 and "
            "51515151-0000-0000-0002-000000000002 sell each today?"
        ),
    )

    fact = result.outcome.requested_facts[0]
    assert fact.answer_expression is not None
    assert fact.answer_expression.group_key is not None
    assert fact.answer_expression.group_key.question_input_refs == (
        "qi_staff_1",
        "qi_staff_2",
    )
    assert tuple(output.role for output in fact.answer_outputs) == ("ROW_COUNT",)


def test_requested_fact_rejects_duplicate_row_population_outputs():
    with pytest.raises(ValueError, match="at most one row population"):
        RequestedFact(
            id="fact_1",
            description="sales count",
            answer_outputs=(
                RequestedFactAnswerOutput(
                    id="answer_1",
                    description="count for first staff member",
                    role="ROW_COUNT",
                ),
                RequestedFactAnswerOutput(
                    id="answer_2",
                    description="count for second staff member",
                    role="ROW_COUNT",
                ),
            ),
        )


def test_literal_input_requires_explicit_role():
    with pytest.raises(TypeError):
        RequestedFactLiteralInput(
            id="store",
            source=KnownInputSource.QUESTION_CONTEXT,
            text="BBS Mall",
            resolved_value_text="BBS Mall",
        )


def test_question_contract_model_materializes_input_refs_as_fact_known_inputs():
    period = RequestedFactLiteralInput(
        id="period",
        source=KnownInputSource.QUESTION_CONTEXT,
        text="yesterday",
        resolved_value_text="yesterday",
        role=LiteralInputRole.TIME_VALUE,
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
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),
                ),
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
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer_1", role="ANSWER_VALUE"),
                ),
                known_inputs=(period,),
            ),
        ),
    )

    assert contract.to_model_dict() == {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            {
                "id": "period",
                "kind": "literal_text",
                "source": "question_context",
                "text": "yesterday",
                "resolved_value_text": "yesterday",
                "role": "time_value",
            }
        ],
        "answer_requests": [
            {
                "id": "fact_1",
                "answer_fact": "sales yesterday",
                "answer_expression": {"family": "scalar_aggregate"},
                "answer_outputs": [
                    {
                        "id": "answer_1",
                        "description": "answer_1",
                        "role": "ANSWER_VALUE",
                    }
                ],
                "used_question_inputs": ["period"],
            }
        ],
    }


def test_question_contract_parser_accepts_positive_used_question_inputs():
    staff_a = {
        "input_ref": "staff_a",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000001",
        "operand_text": "51515151-0000-0000-0002-000000000001",
        "field_label_text": "staff id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "first staff id constrains the requested sales"
        },
    }
    staff_b = {
        "input_ref": "staff_b",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000002",
        "operand_text": "51515151-0000-0000-0002-000000000002",
        "field_label_text": "staff id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "second staff id constrains the requested sales"
        },
    }
    today = {
        "input_ref": "today",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "today",
        "operand_text": "today",
        "role": "time_value",
        "inventory_check": {
            "why_this_is_an_input": "today constrains the requested sales"
        },
    }
    payload = _single_input_payload(staff_a)
    payload["answer_requests_count"] = 2
    payload["question_inputs"] = [staff_a, staff_b, today]
    first_request = payload["answer_requests"][0]
    first_request["answer_fact"] = "sales for first staff member today"
    first_request["used_question_inputs"] = ["staff_a", "today"]
    second_request = {
        **first_request,
        "answer_fact": "sales for second staff member today",
        "used_question_inputs": ["staff_b", "today"],
    }
    payload["answer_requests"] = [first_request, second_request]

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=(
            "How many sales did the staff members with ids: "
            "51515151-0000-0000-0002-000000000001 and "
            "51515151-0000-0000-0002-000000000002 sell each today?"
        ),
    )

    assert [fact.input_refs for fact in parsed.outcome.requested_facts] == [
        ("staff_a", "today"),
        ("staff_b", "today"),
    ]


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
                "answer_population": {
                    "population_label": "sales",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                            "owned_question_input_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "sales count", "role": "ANSWER_VALUE"}
                ],
                "used_question_inputs": [],
                "resolver_choice": "not part of the contract",
            }
        ],
    }

    with pytest.raises(ValueError, match="unparsed"):
        parse_question_contract(
            tool_name="submit_question_contract_outcome",
            payload=_decision_payload(payload),
            question_context="How many sales today?",
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
                "value_source_text": "Alice Smith",
                "operand_text": "Alice Smith",
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
                "answer_population": {
                    "population_label": "her sales",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                            "owned_question_input_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "sales total", "role": "ANSWER_VALUE"}
                ],
                "used_question_inputs": ["input_staff"],
            }
        ],
    }

    with pytest.raises(ValueError, match="value_source_text"):
        parse_question_contract(
            tool_name="submit_question_contract_outcome",
            payload=_decision_payload(payload),
            question_context="What were her sales?",
            question_context_texts=("Alice Smith",),
        )


def test_result_limit_requires_canonical_digit_text_at_parse_boundary():
    payload = {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            {
                "input_ref": "input_limit",
                "kind": "literal_text",
                "source": "question_context",
                "value_source_text": "top five",
                "operand_text": "five",
                "role": "result_limit",
                "inventory_check": {
                    "why_this_is_an_input": "top five supplies the result limit"
                },
            }
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
        "answer_requests": [
            {
                "answer_fact": "top five sales",
                "answer_expression": {"family": "ranked_list"},
                "answer_subject": {
                    "subject_text": "sales",
                    "instance_interpretation": {
                        "kind": "NORMAL_BUSINESS_INSTANCE",
                    },
                },
                "answer_population": {
                    "population_label": "sales",
                    "counted_unit": "sale",
                    "membership_tests": [
                        {
                            "test_id": "test_1",
                            "kind": "SUBJECT_IDENTITY",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this a sale?",
                            "owned_question_input_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "top sales", "role": "ANSWER_VALUE"}
                ],
                "used_question_inputs": ["input_limit"],
            }
        ],
    }

    with pytest.raises(ValueError, match="canonical positive integer digits"):
        parse_question_contract(
            tool_name="submit_question_contract_outcome",
            payload=_decision_payload(payload),
            question_context="Show top five sales.",
        )


def test_result_limit_input_requires_canonical_digit_text_when_constructed_directly():
    with pytest.raises(ValueError, match="canonical positive integer digits"):
        RequestedFactLiteralInput(
            id="input_limit",
            source=KnownInputSource.QUESTION_CONTEXT,
            text="top five",
            resolved_value_text="five",
            role=LiteralInputRole.RESULT_LIMIT,
        )


def test_question_contract_parser_rejects_partial_uuid_segment_input_span():
    question = "How much did staff_id 51515151-0000-0000-0002-000000000001 make today?"
    payload = _single_input_payload(
        {
            "input_ref": "input_staff",
            "kind": "literal_text",
            "source": "question_context",
            "value_source_text": "0002",
            "operand_text": "0002",
            "role": "reference_value",
            "value_meaning_hint": "staff member",
            "inventory_check": {
                "why_this_is_an_input": "0002 is a partial identifier fragment"
            },
        }
    )

    with pytest.raises(ValueError, match="value_source_text"):
        parse_question_contract(
            tool_name="submit_question_contract_outcome",
            payload=_decision_payload(payload),
            question_context=question,
        )


def test_question_contract_parser_rejects_partial_snake_case_input_span():
    question = "How much did staff_id 51515151-0000-0000-0002-000000000001 make today?"
    payload = _single_input_payload(
        {
            "input_ref": "input_field",
            "kind": "literal_text",
            "source": "question_context",
            "value_source_text": "id",
            "operand_text": "id",
            "role": "reference_value",
            "value_meaning_hint": "field fragment",
            "inventory_check": {
                "why_this_is_an_input": "id is a partial field-name fragment"
            },
        }
    )

    with pytest.raises(ValueError, match="value_source_text"):
        parse_question_contract(
            tool_name="submit_question_contract_outcome",
            payload=_decision_payload(payload),
            question_context=question,
        )


def test_question_contract_parser_allows_field_label_hint_for_id_values():
    question = (
        "How many sales did the staff members with ids: "
        "51515151-0000-0000-0002-000000000001 and "
        "51515151-0000-0000-0002-000000000002 sell each today?"
    )
    first = {
        "input_ref": "input_staff_a",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000001",
        "operand_text": "51515151-0000-0000-0002-000000000001",
        "field_label_text": "staff member id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "first staff id constrains the requested sales"
        },
    }
    second = {
        "input_ref": "input_staff_b",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000002",
        "operand_text": "51515151-0000-0000-0002-000000000002",
        "field_label_text": "staff member id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "second staff id constrains the requested sales"
        },
    }
    payload = _single_input_payload(first)
    payload["question_inputs"] = [first, second]
    payload["answer_requests"][0]["used_question_inputs"] = [
        "input_staff_a",
        "input_staff_b",
    ]

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=question,
    )

    assert tuple(
        input_.field_label_text for input_ in parsed.outcome.question_inputs
    ) == (
        "staff member id",
        "staff member id",
    )


def test_question_contract_parser_allows_field_label_scoped_over_coordinated_values():
    question = (
        "How much did staff_id 51515151-0000-0000-0002-000000000001 "
        "and 51515151-0000-0000-0002-000000000002 make today?"
    )
    first = {
        "input_ref": "input_staff_a",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000001",
        "operand_text": "51515151-0000-0000-0002-000000000001",
        "field_label_text": "staff_id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "first staff id constrains the requested sales"
        },
    }
    second = {
        "input_ref": "input_staff_b",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000002",
        "operand_text": "51515151-0000-0000-0002-000000000002",
        "field_label_text": "staff_id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "second staff id constrains the requested sales"
        },
    }
    payload = _single_input_payload(first)
    payload["question_inputs"] = [first, second]
    payload["answer_requests"][0]["used_question_inputs"] = [
        "input_staff_a",
        "input_staff_b",
    ]

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=question,
    )

    assert tuple(
        input_.field_label_text for input_ in parsed.outcome.question_inputs
    ) == (
        "staff_id",
        "staff_id",
    )


def test_question_contract_parser_allows_repeated_field_label_for_later_value():
    question = (
        "How much did staff_id 51515151-0000-0000-0002-000000000001 "
        "and staff_id 51515151-0000-0000-0002-000000000002 make today?"
    )
    first = {
        "input_ref": "input_staff_a",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000001",
        "operand_text": "51515151-0000-0000-0002-000000000001",
        "field_label_text": "staff_id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "first staff id constrains the requested sales"
        },
    }
    second = {
        "input_ref": "input_staff_b",
        "kind": "literal_text",
        "source": "question_context",
        "value_source_text": "51515151-0000-0000-0002-000000000002",
        "operand_text": "51515151-0000-0000-0002-000000000002",
        "field_label_text": "staff_id",
        "role": "reference_value",
        "value_meaning_hint": "staff member",
        "inventory_check": {
            "why_this_is_an_input": "second staff id constrains the requested sales"
        },
    }
    payload = _single_input_payload(first)
    payload["question_inputs"] = [first, second]
    payload["answer_requests"][0]["used_question_inputs"] = [
        "input_staff_a",
        "input_staff_b",
    ]

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=question,
    )

    assert tuple(input_.text for input_ in parsed.outcome.question_inputs) == (
        "51515151-0000-0000-0002-000000000001",
        "51515151-0000-0000-0002-000000000002",
    )


def test_question_contract_parser_rejects_answer_subject_literal_input():
    question = "How many unverified cash deposits are there?"
    payload = _single_input_payload(
        {
            "input_ref": "subject",
            "kind": "literal_text",
            "source": "question_context",
            "value_source_text": "cash deposits",
            "operand_text": "cash deposits",
            "role": "reference_value",
            "value_meaning_hint": "cash deposits",
            "inventory_check": {
                "why_this_is_an_input": "cash deposits is a named reference"
            },
        }
    )
    payload["answer_requests"][0]["answer_fact"] = "count of unverified cash deposits"
    payload["answer_requests"][0]["answer_subject"]["subject_text"] = "cash deposits"
    payload["answer_requests"][0]["answer_population"]["population_label"] = (
        "cash deposits"
    )
    payload["answer_requests"][0]["answer_population"]["counted_unit"] = "cash deposits"

    with pytest.raises(ValueError, match="answer subject"):
        parse_question_contract(
            tool_name="submit_question_contract_outcome",
            payload=_decision_payload(payload),
            question_context=question,
        )
