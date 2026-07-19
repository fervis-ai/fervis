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
    return payload


def _membership_test(
    *,
    test_id: str,
    kind: str,
    question: str,
) -> dict[str, object]:
    return {
        "test_id": test_id,
        "kind": kind,
        "polarity": "MUST_PASS",
        "test_question": question,
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
        "answer_subject": {
            "subject_text": "sales",
            "instance_interpretation": {"kind": "NORMAL_BUSINESS_INSTANCE"},
        },
        "answer_population": {
            "population_label": "requested sales",
            "counted_unit": "sale",
            "membership_tests": membership_tests,
        },
        "answer_outputs": [{"description": "sales count", "role": "ROW_COUNT"}],
        "question_input_uses": question_input_uses,
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
        "answer_requests": [
            _answer_request(
                answer_fact="sales count for each specified staff member today",
                expression={
                    "family": "grouped_aggregate",
                    "group_key": {
                        "description": "staff member",
                        "domain": "SPECIFIED_QUESTION_INPUTS",
                    },
                },
                membership_tests=[
                    _membership_test(
                        test_id="t_subject",
                        kind="SUBJECT_IDENTITY",
                        question="Is this a sale?",
                    ),
                    _membership_test(
                        test_id="t_today",
                        kind="EXPLICIT_USER_CONSTRAINT",
                        question="Did the sale occur today?",
                    ),
                ],
                question_input_uses=[
                    {"input_ref": "qi_staff_1", "owner_kind": "GROUP_KEY"},
                    {"input_ref": "qi_staff_2", "owner_kind": "GROUP_KEY"},
                    {
                        "input_ref": "qi_today",
                        "owner_kind": "POPULATION_TESTS",
                        "membership_test_ids": ["t_today"],
                    },
                ],
            )
        ],
        "question_input_inventory_check": {
            "all_input_like_phrases_declared": True,
        },
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
        ("t_subject", ()),
        ("t_today", ("qi_today",)),
    )


def test_one_input_may_supply_multiple_population_tests() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_expression"] = {"family": "list_rows"}
    request["answer_population"]["membership_tests"] = [
        _membership_test(
            test_id="t_subject",
            kind="SUBJECT_IDENTITY",
            question="Is this a sale?",
        ),
        _membership_test(
            test_id="t_started",
            kind="EXPLICIT_USER_CONSTRAINT",
            question="Did the sale start after today?",
        ),
        _membership_test(
            test_id="t_finished",
            kind="EXPLICIT_USER_CONSTRAINT",
            question="Did the sale finish after today?",
        ),
    ]
    request["question_input_uses"] = [
        {
            "input_ref": "qi_today",
            "owner_kind": "POPULATION_TESTS",
            "membership_test_ids": ["t_started", "t_finished"],
        }
    ]
    payload["question_inputs"] = [payload["question_inputs"][2]]

    result = _parse(payload)

    tests = result.outcome.requested_facts[0].answer_population.membership_tests
    assert tuple(test.owned_question_input_refs for test in tests) == (
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
            test_id="t_subject",
            kind="SUBJECT_IDENTITY",
            question="Is this a sale?",
        ),
        _membership_test(
            test_id="t_staff",
            kind="EXPLICIT_USER_CONSTRAINT",
            question="Is the sale associated with a requested staff member?",
        ),
    ]
    request["question_input_uses"] = [
        {
            "input_ref": "qi_staff_1",
            "owner_kind": "POPULATION_TESTS",
            "membership_test_ids": ["t_staff"],
        },
        {
            "input_ref": "qi_staff_2",
            "owner_kind": "POPULATION_TESTS",
            "membership_test_ids": ["t_staff"],
        },
    ]
    payload["question_inputs"] = payload["question_inputs"][:2]

    result = _parse(payload)

    assert result.outcome.requested_facts[0].answer_population.membership_tests[
        1
    ].owned_question_input_refs == ("qi_staff_1", "qi_staff_2")


def test_one_declared_input_may_be_owned_independently_by_multiple_facts() -> None:
    payload = _grouped_staff_contract()
    today = payload["question_inputs"][2]
    request = payload["answer_requests"][0]
    request["answer_expression"] = {"family": "scalar_aggregate"}
    request["answer_population"]["membership_tests"] = [
        _membership_test(
            test_id="t_subject",
            kind="SUBJECT_IDENTITY",
            question="Is this a sale?",
        ),
        _membership_test(
            test_id="t_today",
            kind="EXPLICIT_USER_CONSTRAINT",
            question="Did the sale occur today?",
        ),
    ]
    request["question_input_uses"] = [
        {
            "input_ref": "qi_today",
            "owner_kind": "POPULATION_TESTS",
            "membership_test_ids": ["t_today"],
        }
    ]
    second = deepcopy(request)
    second["answer_fact"] = "refund count today"
    second["answer_subject"]["subject_text"] = "refunds"
    second["answer_population"]["counted_unit"] = "refund"
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
            expression={"family": "ranked_selection"},
            membership_tests=[
                _membership_test(
                    test_id="t_subject",
                    kind="SUBJECT_IDENTITY",
                    question="Is this a sale?",
                )
            ],
            question_input_uses=[
                {"input_ref": "qi_limit", "owner_kind": "RESULT_LIMIT"}
            ],
        )
    ]

    result = _parse(payload, question="Show the top 3 sales.")

    expression = result.outcome.requested_facts[0].answer_expression
    assert expression is not None
    assert expression.limit_input_ref == "qi_limit"


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda payload: payload["answer_requests"][0]["question_input_uses"].append(
                {"input_ref": "missing", "owner_kind": "GROUP_KEY"}
            ),
            "unknown question input",
        ),
        (
            lambda payload: payload["answer_requests"][0]["question_input_uses"].append(
                {"input_ref": "qi_staff_1", "owner_kind": "GROUP_KEY"}
            ),
            "duplicates question input",
        ),
        (
            lambda payload: payload["answer_requests"][0]["question_input_uses"][
                2
            ].update({"membership_test_ids": ["missing"]}),
            "unknown membership test",
        ),
        (
            lambda payload: payload["answer_requests"][0]["question_input_uses"][
                2
            ].update({"membership_test_ids": ["t_subject"]}),
            "non-explicit membership test",
        ),
        (
            lambda payload: payload["answer_requests"][0]["question_input_uses"][
                2
            ].update({"membership_test_ids": ["t_today", "t_today"]}),
            "duplicates membership test",
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
    request["question_input_uses"] = request["question_input_uses"][:2]
    payload["question_inputs"] = payload["question_inputs"][:2]

    with pytest.raises(ValueError, match="requires at least one question input"):
        _parse(payload)


def test_parser_rejects_group_owner_without_a_specified_input_group_key() -> None:
    payload = _grouped_staff_contract()
    payload["answer_requests"][0]["answer_expression"] = {"family": "scalar_aggregate"}

    with pytest.raises(ValueError, match="GROUP_KEY.*SPECIFIED_QUESTION_INPUTS"):
        _parse(payload)


def test_parser_rejects_specified_input_group_key_without_owned_inputs() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["question_input_uses"] = [request["question_input_uses"][2]]
    payload["question_inputs"] = [payload["question_inputs"][2]]

    with pytest.raises(ValueError, match="requires at least one GROUP_KEY"):
        _parse(payload)


def test_parser_rejects_result_limit_owner_for_non_limit_input() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_expression"] = {"family": "ranked_selection"}
    request["answer_population"]["membership_tests"] = [
        request["answer_population"]["membership_tests"][0]
    ]
    request["question_input_uses"] = [
        {"input_ref": "qi_today", "owner_kind": "RESULT_LIMIT"}
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
            expression={"family": "ranked_selection"},
            membership_tests=[
                _membership_test(
                    test_id="t_subject",
                    kind="SUBJECT_IDENTITY",
                    question="Is this a sale?",
                ),
                _membership_test(
                    test_id="t_limit",
                    kind="EXPLICIT_USER_CONSTRAINT",
                    question="Is this sale within the requested limit?",
                ),
            ],
            question_input_uses=[
                {
                    "input_ref": "qi_limit",
                    "owner_kind": "POPULATION_TESTS",
                    "membership_test_ids": ["t_limit"],
                }
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
            expression={"family": "ranked_selection"},
            membership_tests=[
                _membership_test(
                    test_id="t_subject",
                    kind="SUBJECT_IDENTITY",
                    question="Is this a sale?",
                )
            ],
            question_input_uses=[
                {"input_ref": "qi_limit_3", "owner_kind": "RESULT_LIMIT"},
                {"input_ref": "qi_limit_5", "owner_kind": "RESULT_LIMIT"},
            ],
        )
    ]

    with pytest.raises(ValueError, match="at most one result limit"):
        _parse(payload, question="Show the top 3 or top 5 sales.")


def test_legacy_provider_ownership_fields_are_rejected() -> None:
    payload = _grouped_staff_contract()
    request = payload["answer_requests"][0]
    request["answer_population"]["membership_tests"].insert(
        1,
        _membership_test(
            test_id="t_staff",
            kind="EXPLICIT_USER_CONSTRAINT",
            question="Is the sale associated with a requested staff member?",
        ),
    )
    del request["question_input_uses"]
    request["used_question_inputs"] = ["qi_staff_1", "qi_staff_2", "qi_today"]
    request["answer_expression"]["group_key"]["question_input_refs"] = [
        "qi_staff_1",
        "qi_staff_2",
    ]
    for test in request["answer_population"]["membership_tests"]:
        if test["test_id"] == "t_staff":
            test["owned_question_input_refs"] = ["qi_staff_1", "qi_staff_2"]
        elif test["test_id"] == "t_today":
            test["owned_question_input_refs"] = ["qi_today"]
        else:
            test["owned_question_input_refs"] = []

    with pytest.raises(ValueError, match="unparsed fields|missing required field"):
        _parse(payload)


def test_schema_exposes_only_the_single_ownership_ledger() -> None:
    schema = build_answer_request_contract_schema()
    answer_request = schema["properties"]["answer_requests"]["items"]
    properties = answer_request["properties"]

    assert "question_input_uses" in properties
    assert "used_question_inputs" not in properties
    assert (
        "question_input_refs"
        not in properties["answer_expression"]["oneOf"][0]["properties"]["group_key"][
            "oneOf"
        ][0]["properties"]
    )
    membership_variants = properties["answer_population"]["properties"][
        "membership_tests"
    ]["items"]["oneOf"]
    assert all(
        "owned_question_input_refs" not in variant["properties"]
        for variant in membership_variants
    )
