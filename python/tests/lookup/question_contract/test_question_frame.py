import pytest

from fervis.lookup.question_contract.clarification import (
    IncompleteFactualRequestKind,
    QuestionContractNeedsClarification,
)
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.question_contract.parser import (
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_frame,
)
from fervis.lookup.question_contract.prompt import SemanticQuestionContractTurnPrompt
from fervis.lookup.question_contract.request import QuestionContractRequest
from fervis.lookup.semantic_types import CollectionType, TemporalScopeType, TextType
from fervis.lookup.turn_prompts import build_turn_prompt_context


def test_question_frame_parses_requested_meaning_and_identity_once() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            supplied_values=[
                {
                    "meaning": "the area named by the question",
                    "denotation_basis": "Nairobi refers to one particular area.",
                    "entity_reference": {
                        "instance_kind": "area",
                        "value": {
                            "operands": ["Nairobi"],
                            "origin": {"kind": "question"},
                        },
                    },
                }
            ]
        ),
        question_context_texts=("How many stores are in Nairobi?",),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert tuple(item.id for item in parsed.inputs) == ("i1",)
    assert parsed.inputs[0].operand == "Nairobi"
    assert isinstance(parsed.inputs[0].value_type, TextType)
    assert parsed.input_denotations[0].input_ref == "i1"
    assert parsed.input_denotations[0].kind is InputDenotationKind.IDENTITY_REFERENCE
    assert parsed.input_denotations[0].denoted_instance_kind == "area"


def test_question_frame_rejects_resolved_input_ref_as_identity_value() -> None:
    payload = _frame_payload(
        supplied_values=[
            {
                "meaning": "Nadia Wanjiku",
                "denotation_basis": "Prior context resolves she to Nadia Wanjiku.",
                "entity_reference": {
                    "instance_kind": "staff",
                    "value": {
                        "operands": ["conversation.v1"],
                        "origin": {
                            "kind": "conversation_resolution",
                            "resolved_input_ref": "conversation.v1",
                        },
                    },
                },
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="conversation input operand must copy its resolved value",
    ):
        parse_semantic_question_frame(
            payload,
            question_context_texts=(
                "Where did she work on her first two shifts?",
            ),
            conversation_text_by_resolved_input_ref={
                "conversation.v1": "Nadia Wanjiku",
            },
        )


def test_question_frame_parses_entity_reference_without_repeated_identity_type() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            supplied_values=[
                {
                    "meaning": "the supplied staff identity",
                    "denotation_basis": "The UUID identifies one staff entity.",
                    "entity_reference": {
                        "instance_kind": "staff",
                        "value": {
                            "operands": [
                                "51515151-0000-0000-0002-000000000001"
                            ],
                            "origin": {"kind": "question"},
                        },
                    },
                }
            ]
        ),
        question_context_texts=(
            "How many sales did staff 51515151-0000-0000-0002-000000000001 make?",
        ),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert (
        parsed.input_denotations[0].kind
        is InputDenotationKind.IDENTITY_REFERENCE
    )
    assert parsed.input_denotations[0].denoted_instance_kind == "staff"


def test_question_frame_rejects_a_missing_supplied_value_branch() -> None:
    payload = _frame_payload(
        supplied_values=[
            {
                "meaning": "the named staff member",
                "denotation_basis": "Nadia Wanjiku names one staff member.",
                "entity_reference": None,
            }
        ]
    )

    with pytest.raises(ValueError):
        parse_semantic_question_frame(
            payload,
            question_context_texts=("Which staff member is named Nadia Wanjiku?",),
        )


def test_question_frame_lowers_alternatives_to_one_collection_input() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            result_kind="one_per_group",
            grouping_meanings=["staff member identity"],
            returned_meanings=[
                {
                    "meaning_ref": "r1",
                    "meaning": "staff member identity",
                    "origin": {"kind": "question"},
                },
                {
                    "meaning_ref": "r2",
                    "meaning": "sale count",
                    "origin": {"kind": "question"},
                }
            ],
            supplied_values=[
                {
                    "meaning": "the specified staff members",
                    "denotation_basis": (
                        "Each supplied value refers to one staff member."
                    ),
                    "entity_reference": {
                        "instance_kind": "staff member",
                        "value": {
                            "operands": ["member A", "member B"],
                            "origin": {"kind": "question"},
                        },
                    },
                }
            ],
        ),
        question_context_texts=("How many sales did member A and member B make each?",),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert parsed.inputs[0].operand == ("member A", "member B")
    assert isinstance(parsed.inputs[0].value_type, CollectionType)
    assert isinstance(parsed.inputs[0].value_type.element_type, TextType)


def test_question_frame_preserves_group_key_ordering_owner() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            result_kind="one_per_group",
            grouping_meanings=["event day"],
            ordering=[
                {
                    "ownership_basis": "The existing day key determines order.",
                    "kind": "group_ref",
                    "group_ref": "g1",
                }
            ],
        ),
        question_context_texts=("Return daily event counts ordered by day.",),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert parsed.answer_requests[0].ordering_group_refs == ("g1",)


def test_question_frame_keeps_distinct_values_with_the_same_meaning() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            supplied_values=[
                {
                    "meaning": "staff member id",
                    "denotation_basis": "The first UUID identifies one staff member.",
                    "entity_reference": {
                        "instance_kind": "staff member",
                        "value": {
                            "operands": ["staff-a"],
                            "origin": {"kind": "question"},
                        },
                    },
                },
                {
                    "meaning": "staff member id",
                    "denotation_basis": "The second UUID identifies one staff member.",
                    "entity_reference": {
                        "instance_kind": "staff member",
                        "value": {
                            "operands": ["staff-b"],
                            "origin": {"kind": "question"},
                        },
                    },
                },
            ]
        ),
        question_context_texts=("Compare staff-a and staff-b.",),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert tuple(item.operand for item in parsed.inputs) == ("staff-a", "staff-b")


def test_question_frame_parses_temporal_scope_without_second_classification() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            supplied_values=[
                {
                    "meaning": "the requested period",
                    "denotation_basis": "Today supplies a time period.",
                    "non_entity_value": {
                        "kind": "temporal_scope",
                        "value": {
                            "operands": ["today"],
                            "origin": {"kind": "question"},
                        }
                    },
                }
            ]
        ),
        question_context_texts=("How many stores opened today?",),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert isinstance(parsed.inputs[0].value_type, TemporalScopeType)
    assert parsed.input_denotations[0].kind is InputDenotationKind.NON_IDENTITY_SCALAR
    assert parsed.input_denotations[0].denoted_instance_kind is None


def test_question_frame_owns_bounded_selection_limit_once() -> None:
    payload = _frame_payload()
    result = payload["outcome"]["answer_requests"][0]["request"]["result"]
    result.clear()
    result.update({
        "kind": "one_result_per_qualifying_row",
        "result_candidates": {
            "instance_kind": "observations",
            "origin": {"kind": "question"},
        },
        "projection": {
            "projection_basis": "Return each qualifying observation identity.",
            "candidate_identity": "returned",
            "explicitly_requested_values": [],
        },
        "result_order": {
            "ordering_request_basis": "Measured value determines result order.",
            "ordering": {
                "kind": "ordered_by",
                "values": [
                    {
                        "ownership_basis": "Measured value is not returned.",
                        "kind": "unreturned_ordering_meaning",
                        "meaning": "measured value",
                        "origin": {"kind": "question"},
                    }
                ],
            },
            "selection": {"kind": "take_with_boundary_ties"},
        },
    })
    payload["outcome"]["supplied_values"]["selection_limits"] = [
        {
            "answer_request_number": 1,
            "meaning": "requested number of results",
            "denotation_basis": "Five supplies the result limit.",
            "non_entity_value": {
                "value": {
                    "operands": ["5"],
                    "value_type": {"kind": "integer"},
                    "origin": {"kind": "question"},
                }
            },
        }
    ]
    parsed = parse_semantic_question_frame(
        payload,
        question_context_texts=(
            "Which five observations had the largest measured value?",
        ),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert tuple(item.operand for item in parsed.inputs) == ("5",)
    assert parsed.answer_requests[0].selection_limit_input_ref == "i1"


def test_question_frame_preserves_returned_and_ordering_meanings_once() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            result_kind="one_per_group",
            grouping_meanings=["salesperson identity"],
                projection={
                    "projection_basis": "Return each salesperson and their revenue.",
                    "returned_grouping_keys": "all",
                    "explicitly_requested_values": [
                    {
                        "value_ref": "v1",
                        "value_kind_basis": "Revenue is a requested value.",
                        "value_kind": "value",
                        "meaning": "revenue",
                        "origin": {"kind": "question"},
                    }
                ],
            },
            ordering=[
                {
                    "ownership_basis": "Revenue is returned and orders the groups.",
                    "kind": "requested_value_ref",
                    "value_ref": "v1",
                }
            ],
        ),
        question_context_texts=(
            "Which salesperson made the most revenue, and how much?",
        ),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    request = parsed.answer_requests[0]
    assert tuple(item.meaning for item in request.grouping_origins) == (
        "salesperson identity",
    )
    assert tuple(item.meaning for item in request.ordering_origins) == ("revenue",)
    assert tuple(item.meaning for item in request.output_origins) == (
        "salesperson identity",
        "revenue",
    )
    assert request.output_kinds == ("identity", "value")


def test_relational_turn_receives_question_frame_decision_bases() -> None:
    question = "Which salesperson made the most revenue, and how much?"
    parsed = parse_semantic_question_frame(
        _frame_payload(
            result_kind="one_per_group",
            grouping_meanings=["salesperson identity"],
            ordering=[
                {
                    "ownership_basis": "Revenue orders the groups.",
                    "kind": "unreturned_ordering_meaning",
                    "meaning": "revenue",
                    "origin": {"kind": "question"},
                }
            ],
        ),
        question_context_texts=(question,),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    prompt = SemanticQuestionContractTurnPrompt(
        QuestionContractRequest(
            current_question=question,
            conversation_context={},
        ),
        meaning=parsed,
    ).to_model_invocation(
        build_turn_prompt_context(
            current_question=question,
            conversation_context={},
        )
    ).prompt_text

    assert '"return_request_basis": "The answer asks for the declared result."' in prompt
    assert '"relational_shape_basis": "This is an ordinary relational request."' in prompt
    assert (
        '"result_grain_basis": "The declared result grain answers the question."'
        in prompt
    )
    assert '"ordering_request_basis": "The declared values determine order."' in prompt
    assert "Grouping assigns a key to every qualifying row." in prompt
    assert "Qualification selects the requested keys." in prompt


def test_question_frame_preserves_temporal_grouping_value_shape() -> None:
    question = "What was total revenue on each date?"
    payload = _frame_payload(
        result_kind="one_per_group",
        grouping_meanings=["date"],
    )
    [grouping] = payload["outcome"]["answer_requests"][0]["request"]["result"][
        "grouping_meanings"
    ]
    grouping["grouping_kind"] = "non_identity_value"
    grouping["grouping_value"] = {
        "kind": "temporal_bucket",
        "grain": "day",
    }

    parsed = parse_semantic_question_frame(
        payload,
        question_context_texts=(question,),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    [request] = parsed.answer_requests
    assert request.grouping_value_shapes == (("temporal_bucket", "day"),)


def test_question_frame_returns_typed_clarification() -> None:
    parsed = parse_semantic_question_frame(
        {
            "decision_basis": "No factual result is stated.",
            "outcome": {
                "kind": "missing_requested_fact",
                "source_text": "Hey, can you check?",
                "why_question_is_incomplete": "The requested result is missing.",
            },
        },
        question_context_texts=("Hey, can you check?",),
    )

    assert isinstance(parsed, QuestionContractNeedsClarification)
    assert (
        parsed.missing[0].missing_kind
        is IncompleteFactualRequestKind.MISSING_REQUESTED_FACT
    )


def _frame_payload(
    *,
    result_kind: str = "one_for_population",
    grouping_meanings: list[str] | None = None,
    returned_meanings: list[dict[str, object]] | None = None,
    projection: dict[str, object] | None = None,
    supplied_values: list[dict[str, object]] | None = None,
    ordering: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    results = returned_meanings or [
        {
            "meaning_ref": "r1",
            "meaning": "store count",
            "origin": {"kind": "question"},
        }
    ]
    provider_result_kind = {
        "one_for_population": "one_value_for_population",
        "one_per_candidate": "one_result_per_qualifying_row",
        "one_per_group": "one_result_per_group",
    }[result_kind]
    result: dict[str, object] = {
        "kind": provider_result_kind,
        {
            "one_for_population": "population_rows",
            "one_per_candidate": "result_candidates",
            "one_per_group": "grouped_observation_rows",
        }[result_kind]: {
            "instance_kind": "store",
            "origin": {"kind": "question"},
        },
    }
    if result_kind == "one_per_group":
        result["grouping_meanings"] = [
            {
                "group_ref": f"g{index}",
                "grouping_basis": "The related staff identity defines each group.",
                "meaning": meaning,
                "origin": {"kind": "question"},
                "grouping_kind": "related_entity_identity",
            }
            for index, meaning in enumerate(grouping_meanings or [], start=1)
        ]
    if result_kind == "one_for_population":
        result["returned_meanings"] = results
    else:
        result["projection"] = projection or {
            "projection_basis": "Return the result identity.",
            **(
                {
                    "returned_grouping_keys": "all",
                    "explicitly_requested_values": [],
                }
                if result_kind == "one_per_group"
                else {
                    "candidate_identity": "returned",
                    "explicitly_requested_values": [],
                }
            ),
        }
    result["result_order"] = {
        "ordering_request_basis": (
            "The declared values determine order." if ordering else "No order is requested."
        ),
        "ordering": (
            {"kind": "ordered_by", "values": ordering}
            if ordering
            else {"kind": "no_ordering_requested"}
        ),
        "selection": {"kind": "all_results"},
    }
    request: dict[str, object] = {
        "return_request_basis": "The answer asks for the declared result.",
        "relational_shape_basis": "This is an ordinary relational request.",
        "request": {
            "relational_shape": "ordinary",
            "result_grain_basis": "The declared result grain answers the question.",
            "result": result,
        },
    }
    operands = []
    for item in supplied_values or []:
        normalized = dict(item)
        reference = normalized.get("entity_reference")
        if isinstance(reference, dict):
            reference = dict(reference)
            value = reference.get("value")
            if isinstance(value, dict) and isinstance(value.get("operands"), list):
                identity_values = value["operands"]
                reference["value"] = (
                    {
                        "kind": "single_identity",
                        "identity_value": identity_values[0],
                        "origin": value.get("origin"),
                    }
                    if len(identity_values) == 1
                    else {
                        "kind": "identity_alternatives",
                        "identity_values": identity_values,
                        "origin": value.get("origin"),
                    }
                )
                normalized["entity_reference"] = reference
        operands.append(normalized)
    return {
        "decision_basis": "The question states one complete factual request.",
        "outcome": {
            "kind": "question_meaning",
            "answer_requests": [request],
            "supplied_values": {
                "operands": operands,
                "selection_limits": [],
            },
            "question_input_inventory_check": {
                "all_input_like_phrases_declared": True,
            },
        },
    }


@pytest.mark.parametrize('result_kind', ('one_per_candidate', 'one_per_group'))
def test_related_entity_output_obeys_declared_result_grain(result_kind):
    from jsonschema import ValidationError, validate
    from fervis.lookup.question_contract.schema import build_semantic_question_frame_schema

    grouped = result_kind == 'one_per_group'
    projection = {
        'projection_basis': 'Return the related person.',
        **({'returned_grouping_keys': 'all'} if grouped else {'candidate_identity': 'returned'}),
        'explicitly_requested_values': [{
            'value_ref': 'v1', 'value_kind_basis': 'An individual related person.',
            'value_kind': 'related_entity', 'meaning': 'worker',
            'origin': {'kind': 'question'},
        }],
    }
    payload = _frame_payload(result_kind=result_kind,
                             grouping_meanings=['store'] if grouped else None,
                             projection=projection)
    schema = build_semantic_question_frame_schema()
    if grouped:
        with pytest.raises(ValidationError):
            validate(payload, schema)
        with pytest.raises(ValueError, match='grouped projection cannot return an ungrouped entity'):
            parse_semantic_question_frame(payload, question_context_texts=('List workers at each store.',))
    else:
        validate(payload, schema)
        assert isinstance(parse_semantic_question_frame(payload, question_context_texts=('List workers at each store.',)), ParsedSemanticQuestionMeaning)
