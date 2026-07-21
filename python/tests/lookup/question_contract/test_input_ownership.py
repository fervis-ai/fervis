from __future__ import annotations

from copy import deepcopy

import pytest

from fervis.lookup.question_contract import (
    build_answer_request_contract_schema,
    parse_question_contract,
)


_QUESTION = (
    "How many sales did the staff members with ids "
    "51515151-0000-0000-0002-000000000001 and "
    "51515151-0000-0000-0002-000000000002 sell each today?"
)


def _decision(outcome: dict[str, object]) -> dict[str, object]:
    return {
        "decision_basis": "The question asks for fact-local grouped sales counts.",
        "outcome": outcome,
    }


def _input(
    *,
    input_ref: str,
    text: str,
    role: str,
    field_label_text: str = "",
    comparison_operator: str = "",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_ref": input_ref,
        "source": "question_context",
        "value_source_text": text,
        "operand_text": text,
        "role": role,
        "inventory_check": {
            "why_this_is_an_input": f"{text} constrains the requested fact",
        },
        "kind": "literal_text",
    }
    if field_label_text:
        payload["field_label_text"] = field_label_text
    if comparison_operator:
        payload["comparison_operator"] = comparison_operator
    return payload


def _membership_test(
    *,
    question: str,
    question_input_refs: list[str],
    comparison_operator: str = "",
) -> dict[str, object]:
    payload = {
        "population_use_refs": [
            f"use_{input_ref}" for input_ref in question_input_refs
        ],
        "polarity": "MUST_PASS",
        "test_question": question,
    }
    if comparison_operator:
        payload["comparison_operator"] = comparison_operator
    return payload


def _population_use(input_ref: str) -> dict[str, object]:
    return {
        "use_id": f"use_{input_ref}",
        "input_ref": input_ref,
        "owner_kind": "POPULATION_TESTS",
    }


def _answer_request(
    *,
    answer_fact: str,
    expression: dict[str, object],
    membership_tests: list[dict[str, object]],
    question_input_uses: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "answer_fact": answer_fact,
        "answer_expression": expression,
        "question_input_uses": question_input_uses,
        "answer_subject": {
            "subject_text": "sales",
            "instance_interpretation": {"kind": "NORMAL_BUSINESS_INSTANCE"},
        },
        "answer_population": {
            "membership_tests": membership_tests,
        },
        "answer_outputs": [{"description": "sales count", "role": "ROW_COUNT"}],
    }


def _grouped_staff_contract() -> dict[str, object]:
    staff_1 = "51515151-0000-0000-0002-000000000001"
    staff_2 = "51515151-0000-0000-0002-000000000002"
    return {
        "kind": "question_contract",
        "answer_requests_count": 1,
        "question_inputs": [
            _input(
                input_ref="qi_staff_1",
                text=staff_1,
                role="reference_value",
                field_label_text="staff id",
            ),
            _input(
                input_ref="qi_staff_2",
                text=staff_2,
                role="reference_value",
                field_label_text="staff id",
            ),
            _input(input_ref="qi_today", text="today", role="time_value"),
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
        "answer_requests": [
            _answer_request(
                answer_fact="sales count for each specified staff member today",
                expression={
                    "family": "grouped_aggregate",
                    "selection": {"kind": "all_results"},
                    "group_key": {
                        "description": "staff member",
                        "value_source": {
                            "kind": "specified_question_inputs",
                        },
                    },
                },
                membership_tests=[
                    _membership_test(
                        question="Did the sale occur today?",
                        question_input_refs=["qi_today"],
                    ),
                ],
                question_input_uses=[
                    {
                        "input_ref": "qi_staff_1",
                        "owner_kind": "GROUP_KEY",
                    },
                    {
                        "input_ref": "qi_staff_2",
                        "owner_kind": "GROUP_KEY",
                    },
                    _population_use("qi_today"),
                ],
            )
        ],
    }


def _parse(
    payload: dict[str, object],
    *,
    question: str = _QUESTION,
):
    return parse_question_contract(
        tool_name="submit_question_contract_outcome",
        payload=_decision(payload),
        question_context=question,
    )


def test_single_ownership_lowers_to_the_existing_requested_fact_contract() -> None:
    result = _parse(_grouped_staff_contract())

    fact = result.outcome.requested_facts[0]
    assert fact.input_refs == ("qi_staff_1", "qi_staff_2", "qi_today")
    assert fact.answer_expression is not None
    assert fact.answer_expression.group_key is not None
    assert fact.answer_expression.group_key.question_input_refs == (
        "qi_staff_1",
        "qi_staff_2",
    )
    assert tuple(
        (test.id, test.owned_question_input_refs)
        for test in fact.answer_population.membership_tests
    ) == (
        ("pop_test_1", ()),
        ("pop_test_2", ()),
        ("explicit_user_constraint_1", ("qi_today",)),
    )


def test_population_schema_authors_only_explicit_user_constraints() -> None:
    schema = build_answer_request_contract_schema()
    population = schema["properties"]["answer_requests"]["items"]["properties"][
        "answer_population"
    ]

    assert population["required"] == ["membership_tests"]
    assert "counted_unit" not in population["properties"]
    membership_test = population["properties"]["membership_tests"]["items"]
    assert membership_test["required"] == [
        "population_use_refs",
        "polarity",
        "test_question",
    ]
    assert set(membership_test["properties"]) == {
        "population_use_refs",
        "polarity",
        "test_question",
    }


def test_population_membership_reuses_the_owned_input_reference() -> None:
    payload = _grouped_staff_contract()

    result = _parse(payload)

    assert result.outcome.requested_facts[0].answer_population.membership_tests[
        2
    ].owned_question_input_refs == ("qi_today",)


def test_one_input_may_supply_multiple_population_tests() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_expression"] = {
        "family": "list_rows",
        "selection": {"kind": "all_results"},
    }
    request["answer_population"]["membership_tests"] = [
        _membership_test(
            question="Did the sale start after today?",
            question_input_refs=["qi_today"],
        ),
        _membership_test(
            question="Did the sale finish after today?",
            question_input_refs=["qi_today"],
        ),
    ]
    request["question_input_uses"] = [_population_use("qi_today")]
    payload["question_inputs"] = [payload["question_inputs"][2]]

    result = _parse(payload)

    tests = result.outcome.requested_facts[0].answer_population.membership_tests
    assert tuple(test.owned_question_input_refs for test in tests) == (
        (),
        (),
        ("qi_today",),
        ("qi_today",),
    )


def test_multiple_inputs_may_supply_one_population_test() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_expression"] = {"family": "scalar_aggregate"}
    request["answer_population"]["membership_tests"] = [
        _membership_test(
            question="Is the sale associated with a requested staff member?",
            question_input_refs=["qi_staff_1", "qi_staff_2"],
        ),
    ]
    request["question_input_uses"] = [
        _population_use("qi_staff_1"),
        _population_use("qi_staff_2"),
    ]
    payload["question_inputs"] = payload["question_inputs"][:2]

    result = _parse(payload)

    assert result.outcome.requested_facts[0].answer_population.membership_tests[
        2
    ].owned_question_input_refs == ("qi_staff_1", "qi_staff_2")


def test_one_declared_input_may_be_owned_independently_by_multiple_facts() -> None:
    payload = _grouped_staff_contract()
    today = payload["question_inputs"][2]
    request = payload["answer_requests"][0]
    request["answer_expression"] = {"family": "scalar_aggregate"}
    request["answer_population"]["membership_tests"] = [
        _membership_test(
            question="Did the sale occur today?",
            question_input_refs=["qi_today"],
        ),
    ]
    request["question_input_uses"] = [_population_use("qi_today")]
    second = deepcopy(request)
    second["answer_fact"] = "refund count today"
    second["answer_subject"]["subject_text"] = "refunds"
    payload["question_inputs"] = [today]
    payload["answer_requests_count"] = 2
    payload["answer_requests"] = [request, second]

    result = _parse(payload)

    assert tuple(fact.input_refs for fact in result.outcome.requested_facts) == (
        ("qi_today",),
        ("qi_today",),
    )


def test_result_limit_use_lowers_to_the_existing_expression_field() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(input_ref="qi_limit", text="3", role="result_limit")
    ]
    payload["answer_requests"] = [
        _answer_request(
            answer_fact="top 3 sales",
            expression={
                "family": "list_rows",
                "ordering": {"basis": "sales amount", "direction": "descending"},
                "selection": {"kind": "take"},
            },
            membership_tests=[
            ],
            question_input_uses=[
                {
                    "input_ref": "qi_limit",
                    "owner_kind": "RESULT_LIMIT",
                }
            ],
        )
    ]

    result = _parse(payload, question="Show the top 3 sales.")

    expression = result.outcome.requested_facts[0].answer_expression
    assert expression is not None
    assert expression.limit_input_ref == "qi_limit"


def test_formula_value_is_owned_by_the_computed_scalar_expression() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(input_ref="qi_rate", text="10%", role="formula_value")
    ]
    payload["answer_requests"] = [
        _answer_request(
            answer_fact="10% of the total measured value",
            expression={"family": "computed_scalar"},
            membership_tests=[
            ],
            question_input_uses=[
                {
                    "input_ref": "qi_rate",
                    "owner_kind": "COMPUTE_EXPRESSION",
                }
            ],
        )
    ]

    result = _parse(payload, question="What is 10% of the total measured value?")

    expression = result.outcome.requested_facts[0].answer_expression
    assert expression is not None
    assert expression.compute_input_refs == ("qi_rate",)


def test_formula_value_cannot_be_owned_by_population_tests() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(input_ref="qi_rate", text="10%", role="formula_value")
    ]
    payload["answer_requests"] = [
        _answer_request(
            answer_fact="10% of the total measured value",
            expression={"family": "computed_scalar"},
            membership_tests=[
                _membership_test(
                    question="Does this value equal 10%?",
                    question_input_refs=["qi_rate"],
                ),
            ],
            question_input_uses=[
                _population_use("qi_rate")
            ],
        )
    ]

    with pytest.raises(ValueError, match="formula_value.*COMPUTE_EXPRESSION"):
        _parse(payload, question="What is 10% of the total measured value?")


def test_formula_value_rejects_unit_bearing_text_without_typed_unit_authority() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(input_ref="qi_amount", text="10 widgets", role="formula_value")
    ]

    with pytest.raises(ValueError, match="formula_value requires a numeric scalar"):
        _parse(payload, question="What is the total plus 10 widgets?")


def test_temporal_group_key_owns_its_grain_without_a_question_input() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = []
    payload["answer_requests"] = [
        _answer_request(
            answer_fact="event count per day",
            expression={
                "family": "grouped_aggregate",
                "group_key": {
                    "description": "event day",
                    "value_source": {"kind": "temporal_bucket", "grain": "day"},
                },
                "selection": {"kind": "all_results"},
            },
            membership_tests=[
            ],
            question_input_uses=[],
        )
    ]

    result = _parse(payload, question="How many events were recorded per day?")

    expression = result.outcome.requested_facts[0].answer_expression
    assert expression is not None
    assert expression.group_key is not None
    assert expression.group_key.source_kind.value == "temporal_bucket"
    assert expression.group_key.temporal_grain == "day"
    assert result.outcome.question_inputs == ()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: payload["answer_requests"][0][
                "question_input_uses"
            ].append({"input_ref": "missing", "owner_kind": "GROUP_KEY"}),
            "unknown question input",
        ),
        (
            lambda payload: payload["answer_requests"][0][
                "question_input_uses"
            ].append({"input_ref": "qi_staff_1", "owner_kind": "GROUP_KEY"}),
            "duplicates question input",
        ),
        (
            lambda payload: payload["answer_requests"][0]["answer_population"][
                "membership_tests"
            ][0].update({"population_use_refs": ["missing"]}),
            "references unknown population input use",
        ),
        (
            lambda payload: payload["answer_requests"][0]["answer_population"][
                "membership_tests"
            ][0].update(
                {"population_use_refs": ["use_qi_today", "use_qi_today"]}
            ),
            "duplicates population input use",
        ),
    ],
)
def test_parser_rejects_invalid_population_ownership(mutate, error: str) -> None:
    payload = _grouped_staff_contract()
    mutate(payload)

    with pytest.raises(ValueError, match=error):
        _parse(payload)


def test_parser_rejects_explicit_population_test_without_an_operand() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_population"]["membership_tests"][0][
        "population_use_refs"
    ] = []

    with pytest.raises(ValueError, match="requires at least one question input"):
        _parse(payload)


def test_parser_rejects_group_members_without_a_specified_input_group_key() -> None:
    payload = _grouped_staff_contract()
    payload["answer_requests"][0]["answer_expression"] = {"family": "scalar_aggregate"}

    with pytest.raises(
        ValueError,
        match="GROUP_KEY requires a specified_question_inputs group key",
    ):
        _parse(payload)


def test_parser_rejects_specified_input_group_key_without_owned_inputs() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["question_input_uses"] = request["question_input_uses"][2:]

    with pytest.raises(ValueError, match="SPECIFIED_QUESTION_INPUTS.*GROUP_KEY"):
        _parse(payload)


def test_parser_rejects_result_limit_owner_for_non_limit_input() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_expression"] = {
        "family": "list_rows",
        "ordering": {"basis": "sale amount", "direction": "descending"},
        "selection": {"kind": "take"},
    }
    request["answer_population"]["membership_tests"] = []
    request["question_input_uses"] = [
        {
            "input_ref": "qi_today",
            "owner_kind": "RESULT_LIMIT",
        }
    ]
    payload["question_inputs"] = [payload["question_inputs"][2]]

    with pytest.raises(ValueError, match="RESULT_LIMIT.*result_limit"):
        _parse(payload)


def test_parser_rejects_non_limit_owner_for_result_limit_input() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(input_ref="qi_limit", text="3", role="result_limit")
    ]
    payload["answer_requests"] = [
        _answer_request(
            answer_fact="top 3 sales",
            expression={
                "family": "list_rows",
                "ordering": {"basis": "sale amount", "direction": "descending"},
                "selection": {"kind": "take"},
            },
            membership_tests=[
                _membership_test(
                    question="Is this sale within the requested limit?",
                    question_input_refs=["qi_limit"],
                ),
            ],
            question_input_uses=[
                _population_use("qi_limit")
            ],
        )
    ]

    with pytest.raises(ValueError, match="result_limit.*RESULT_LIMIT"):
        _parse(payload, question="Show the top 3 sales.")


def test_parser_rejects_multiple_result_limits_for_one_fact() -> None:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(input_ref="qi_limit_3", text="3", role="result_limit"),
        _input(input_ref="qi_limit_5", text="5", role="result_limit"),
    ]
    payload["answer_requests"] = [
        _answer_request(
            answer_fact="top sales",
            expression={
                "family": "list_rows",
                "ordering": {"basis": "sale amount", "direction": "descending"},
                "selection": {"kind": "take"},
            },
            membership_tests=[
            ],
            question_input_uses=[
                {
                    "input_ref": "qi_limit_3",
                    "owner_kind": "RESULT_LIMIT",
                },
                {
                    "input_ref": "qi_limit_5",
                    "owner_kind": "RESULT_LIMIT",
                },
            ],
        )
    ]

    with pytest.raises(ValueError, match="at most one result limit"):
        _parse(payload, question="Show the top 3 or top 5 sales.")


def test_legacy_provider_ownership_fields_are_rejected() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    del request["question_input_uses"]
    request["used_question_inputs"] = ["qi_staff_1", "qi_staff_2", "qi_today"]
    request["answer_expression"]["group_key"]["question_input_refs"] = [
        "qi_staff_1",
        "qi_staff_2",
    ]

    with pytest.raises(ValueError, match="unparsed fields|missing required field"):
        _parse(payload)


def test_schema_exposes_only_the_single_ownership_ledger() -> None:
    schema = build_answer_request_contract_schema()
    top_level_properties = list(schema["properties"])
    assert top_level_properties.index("answer_requests") < top_level_properties.index(
        "question_inputs"
    )
    answer_request = schema["properties"]["answer_requests"]["items"]
    properties = answer_request["properties"]
    property_names = list(properties)
    assert property_names.index("question_input_uses") < property_names.index(
        "answer_population"
    )

    assert "question_input_uses" in properties
    use_schema = properties["question_input_uses"]["items"]
    assert [
        branch["properties"]["owner_kind"]["enum"][0] for branch in use_schema["oneOf"]
    ] == [
        "GROUP_KEY",
        "POPULATION_TESTS",
        "COMPUTE_EXPRESSION",
        "RESULT_LIMIT",
    ]
    branches_by_owner = {
        branch["properties"]["owner_kind"]["enum"][0]: branch
        for branch in use_schema["oneOf"]
    }
    assert (
        "use_id" in branches_by_owner["POPULATION_TESTS"]["properties"]
    )
    assert all(
        "use_id" not in branch["properties"]
        for owner, branch in branches_by_owner.items()
        if owner != "POPULATION_TESTS"
    )
    assert "used_question_inputs" not in properties
    grouped_expression = next(
        branch
        for branch in properties["answer_expression"]["oneOf"]
        if branch["properties"]["family"]["enum"] == ["grouped_aggregate"]
    )
    specified_group_key = grouped_expression["properties"]["group_key"]["oneOf"][0]
    assert (
        "question_input_refs"
        not in specified_group_key["properties"]["value_source"]["properties"]
    )
    membership_test = properties["answer_population"]["properties"][
        "membership_tests"
    ]["items"]
    assert "population_label" not in properties["answer_population"]["properties"]
    assert "owned_question_input_refs" not in membership_test["properties"]
    assert "population_use_refs" in membership_test["properties"]


def test_schema_places_comparison_operator_only_on_threshold_inputs() -> None:
    schema = build_answer_request_contract_schema()
    input_branches = schema["properties"]["question_inputs"]["items"]["oneOf"]
    branches_by_role = {
        branch["properties"]["role"]["enum"][0]: branch for branch in input_branches
    }

    threshold = branches_by_role["threshold_value"]
    assert "comparison_operator" in threshold["properties"]
    assert "comparison_operator" in threshold["required"]
    assert all(
        "comparison_operator" not in branch["properties"]
        for role, branch in branches_by_role.items()
        if role != "threshold_value"
    )

    membership_test = schema["properties"]["answer_requests"]["items"]["properties"][
        "answer_population"
    ]["properties"]["membership_tests"]["items"]
    assert "comparison_operator" not in membership_test["properties"]


def _predicate_contract(
    *,
    text: str,
    role: str,
    comparison_operator: str = "",
) -> dict[str, object]:
    payload = _grouped_staff_contract()
    payload["question_inputs"] = [
        _input(
            input_ref="qi_operand",
            text=text,
            role=role,
            comparison_operator=comparison_operator,
        )
    ]
    request = payload["answer_requests"][0]
    request["answer_expression"] = {"family": "scalar_aggregate"}
    request["question_input_uses"] = [_population_use("qi_operand")]
    request["answer_population"]["membership_tests"] = [
        _membership_test(
            question="Does the sale satisfy the supplied predicate?",
            question_input_refs=["qi_operand"],
        ),
    ]
    return payload


def test_predicate_value_is_owned_by_population_tests() -> None:
    result = _parse(
        _predicate_contract(text="completed", role="predicate_value"),
        question="How many completed sales were recorded?",
    )

    known = result.outcome.question_inputs[0]
    test = result.outcome.requested_facts[0].answer_population.membership_tests[2]
    assert known.is_predicate_value
    assert not known.is_reference_value
    assert test.owned_question_input_refs == ("qi_operand",)
    assert test.comparison_operator is None


@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
def test_threshold_value_carries_one_ordered_comparison(operator: str) -> None:
    result = _parse(
        _predicate_contract(
            text="1000",
            role="threshold_value",
            comparison_operator=operator,
        ),
        question="How many records have a measured value compared with 1000?",
    )

    known = result.outcome.question_inputs[0]
    test = result.outcome.requested_facts[0].answer_population.membership_tests[2]
    assert known.is_threshold_value
    assert test.comparison_operator.value == operator


def test_threshold_value_without_comparison_fails_closed() -> None:
    with pytest.raises(ValueError, match="comparison_operator"):
        _parse(
            _predicate_contract(text="1000", role="threshold_value"),
            question="How many records have a measured value compared with 1000?",
        )


def test_non_threshold_comparison_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires threshold_value"):
        _parse(
            _predicate_contract(
                text="completed",
                role="predicate_value",
                comparison_operator="gt",
            ),
            question="How many completed sales were recorded?",
        )
