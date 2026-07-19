from __future__ import annotations

import pytest

from fervis.lookup.clarification import ActiveClarification, ClarificationExchange
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
    ResolvedLiteralQuestionInput,
)
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactOrderingDirection,
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
    use_id = f"use_{input_ref}"
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
                            "question_input_use_refs": [],
                        },
                        {
                            "test_id": "test_input",
                            "kind": "EXPLICIT_USER_CONSTRAINT",
                            "polarity": "MUST_PASS",
                            "test_question": "Does this sale match the specified input?",
                            "question_input_use_refs": [use_id],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "sales total", "role": "ANSWER_VALUE"}
                ],
                "question_input_uses": [
                    {
                        "use_id": use_id,
                        "input_ref": input_ref,
                        "owner_kind": "POPULATION_TESTS",
                    }
                ],
            }
        ],
    }


def _set_population_inputs(
    answer_request: dict[str, object],
    *input_refs: str,
) -> None:
    uses = [
        {
            "use_id": f"use_{input_ref}",
            "input_ref": input_ref,
            "owner_kind": "POPULATION_TESTS",
        }
        for input_ref in input_refs
    ]
    answer_request["question_input_uses"] = uses
    population = answer_request["answer_population"]
    assert isinstance(population, dict)
    tests = population["membership_tests"]
    assert isinstance(tests, list)
    explicit = next(
        test
        for test in tests
        if isinstance(test, dict)
        and test.get("kind") == "EXPLICIT_USER_CONSTRAINT"
    )
    explicit["question_input_use_refs"] = [use["use_id"] for use in uses]


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


def test_question_contract_inherits_complete_clarification_lineage() -> None:
    question = "How many sales happened today?"
    payload = _single_input_payload(
        {
            "input_ref": "input_today",
            "kind": "literal_text",
            "source": "conversation_resolution",
            "resolved_input_ref": "input_today",
            "value_source_text": "today",
            "operand_text": "today",
            "role": "time_value",
            "inventory_check": {
                "why_this_is_an_input": "today constrains the requested result"
            },
        }
    )
    resolution = CompiledConversationResolution(
        current_question_text="today",
        contextualized_question=question,
        clauses=(),
        inputs=(
            ResolvedLiteralQuestionInput(
                input_ref="input_today",
                value_source_text="today",
                resolved_value_text="today",
                role=LiteralInputRole.TIME_VALUE,
            ),
        ),
        frame_call=None,
        used_source_card_ids=(),
        used_memory_ids=(),
        active_clarification=ActiveClarification(
            original_question="How many sales happened?",
            exchanges=(
                ClarificationExchange(
                    response_id="response_1",
                    clarification_id="clarification_1",
                    question="Which day should I use?",
                    answer="today",
                ),
            ),
        ),
    )

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=question,
        conversation_resolution=resolution,
    )

    assert parsed.outcome.clarification_lineage_refs == (
        "clarification_response:response_1",
    )


def test_answer_output_schema_requires_a_computation_role():
    schema = build_answer_request_contract_schema()
    answer_output_schema = schema["properties"]["answer_requests"]["items"][
        "properties"
    ]["answer_outputs"]["items"]

    assert answer_output_schema["required"] == ["description", "role"]


def test_ordered_take_one_is_orthogonal_to_list_rows():
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
        "family": "list_rows",
        "ordering": {"basis": "revenue", "direction": "descending"},
        "selection": {"kind": "take_one"},
    }

    parsed = parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision_payload(payload),
        question_context=question,
    )

    expression = parsed.outcome.requested_facts[0].answer_expression

    assert expression is not None
    assert expression.family is RequestedFactAnswerExpressionFamily.LIST_ROWS
    assert expression.ordering_basis == "revenue"
    assert (
        expression.ordering_direction
        is RequestedFactOrderingDirection.DESCENDING
    )
    assert expression.selection_kind is ResultSelectionKind.TAKE_ONE


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

    assert grouped_branch["required"] == ["family", "selection", "group_key"]


def test_group_key_rejects_repeated_question_inputs():
    with pytest.raises(ValueError, match="group key repeats question input"):
        RequestedFactGroupKey(
            description="staff member",
            domain=GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS,
            question_input_refs=("qi_staff_1", "qi_staff_1"),
        )


def test_grouped_aggregate_serializes_group_key_on_answer_expression():
    fact = RequestedFact(
        id="fact_1",
        description="sales count for each specified staff member today",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            selection_kind=ResultSelectionKind.ALL_RESULTS,
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
        "selection_kind": "all_results",
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
                        "selection": {"kind": "all_results"},
                    "group_key": {
                        "description": "staff member",
                        "domain": "SPECIFIED_QUESTION_INPUTS",
                    },
                },
                "question_input_uses": [
                    {
                        "use_id": "use_staff_1",
                        "input_ref": "qi_staff_1",
                        "owner_kind": "GROUP_KEY",
                    },
                    {
                        "use_id": "use_staff_2",
                        "input_ref": "qi_staff_2",
                        "owner_kind": "GROUP_KEY",
                    },
                    {
                        "use_id": "use_today",
                        "input_ref": "qi_today",
                        "owner_kind": "POPULATION_TESTS",
                    },
                ],
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
                            "question_input_use_refs": [],
                        },
                        {
                            "test_id": "test_2",
                            "kind": "EXPLICIT_USER_CONSTRAINT",
                            "polarity": "MUST_PASS",
                            "test_question": "Did the sale occur today?",
                            "question_input_use_refs": ["use_today"],
                        },
                        {
                            "test_id": "test_3",
                            "kind": "NORMAL_INSTANCE_GUARD",
                            "polarity": "MUST_PASS",
                            "test_question": "Is this an ordinary business sale?",
                            "question_input_use_refs": [],
                        },
                    ],
                },
                "answer_outputs": [{"description": "sales count", "role": "ROW_COUNT"}],
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


def test_question_contract_parser_accepts_fact_local_question_input_uses():
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
    _set_population_inputs(first_request, "staff_a", "today")
    second_request = {
        **first_request,
        "answer_fact": "sales for second staff member today",
    }
    second_request["answer_population"] = {
        **first_request["answer_population"],
        "membership_tests": [
            dict(test)
            for test in first_request["answer_population"]["membership_tests"]
        ],
    }
    _set_population_inputs(second_request, "staff_b", "today")
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
                            "question_input_use_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "sales count", "role": "ANSWER_VALUE"}
                ],
                "question_input_uses": [],
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
                            "question_input_use_refs": [],
                        },
                        {
                            "test_id": "test_staff",
                            "kind": "EXPLICIT_USER_CONSTRAINT",
                            "polarity": "MUST_PASS",
                            "test_question": "Does this sale belong to the staff member?",
                            "question_input_use_refs": ["use_input_staff"],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "sales total", "role": "ANSWER_VALUE"}
                ],
                "question_input_uses": [
                    {
                        "use_id": "use_input_staff",
                        "input_ref": "input_staff",
                        "owner_kind": "POPULATION_TESTS",
                    }
                ],
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
                "answer_expression": {"family": "unsupported_family"},
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
                            "question_input_use_refs": [],
                        }
                    ],
                },
                "answer_outputs": [
                    {"description": "top sales", "role": "ANSWER_VALUE"}
                ],
                "question_input_uses": [
                    {
                        "use_id": "use_input_limit",
                        "input_ref": "input_limit",
                        "owner_kind": "RESULT_LIMIT",
                    }
                ],
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
    _set_population_inputs(
        payload["answer_requests"][0],
        "input_staff_a",
        "input_staff_b",
    )

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
    _set_population_inputs(
        payload["answer_requests"][0],
        "input_staff_a",
        "input_staff_b",
    )

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
    _set_population_inputs(
        payload["answer_requests"][0],
        "input_staff_a",
        "input_staff_b",
    )

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
