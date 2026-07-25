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
from fervis.lookup.semantic_types import CollectionType, TemporalScopeType, TextType


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
            result_kind="grouped_results",
            grouping_meanings=["staff member identity"],
            answer_values=[
                {
                    "value_ref": "v1",
                    "meaning": "sale count",
                    "origin": {"kind": "question"},
                }
            ],
            returned_value_refs=["v1"],
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


def test_question_frame_parses_temporal_scope_without_second_classification() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            supplied_values=[
                {
                    "meaning": "the requested period",
                    "denotation_basis": "Today supplies a time period.",
                    "non_entity_value": {
                        "value": {
                            "operands": ["today"],
                            "value_type": {"kind": "temporal_scope"},
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
    request = payload["outcome"]["answer_requests"][0]
    request["result_kind"] = "qualifying_instances"
    request["returned_candidate_identity"] = {
        "meaning": "observation identity",
        "origin": {"kind": "question"},
    }
    request["returned_result"] = {"kind": "identities"}
    request["returned_value_refs"] = []
    request["answer_values"] = [
        {
            "value_ref": "v1",
            "meaning": "measured value",
            "origin": {"kind": "question"},
        }
    ]
    request["ordering_value_refs"] = ["v1"]
    request["selection"] = {
        "kind": "take_with_boundary_ties",
        "limit": {
            "meaning": "requested number of results",
            "denotation_basis": "Five supplies the result limit.",
            "non_entity_value": {
                "value": {
                    "operands": ["5"],
                    "value_type": {"kind": "integer"},
                    "origin": {"kind": "question"},
                }
            },
        },
    }
    parsed = parse_semantic_question_frame(
        payload,
        question_context_texts=(
            "Which five observations had the largest measured value?",
        ),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    assert tuple(item.operand for item in parsed.inputs) == ("5",)
    assert parsed.answer_requests[0].selection_limit_input_ref == "i1"


def test_question_frame_owns_row_identity_ordering_and_projection_once() -> None:
    parsed = parse_semantic_question_frame(
        _frame_payload(
            result_kind="grouped_results",
            grouping_meanings=["salesperson identity"],
            answer_values=[
                {
                    "value_ref": "v1",
                    "meaning": "revenue",
                    "origin": {"kind": "question"},
                },
            ],
            returned_value_refs=["v1"],
            ordering_value_refs=["v1"],
        ),
        question_context_texts=(
            "Which salesperson made the most revenue, and how much?",
        ),
    )

    assert isinstance(parsed, ParsedSemanticQuestionMeaning)
    request = parsed.answer_requests[0]
    assert request.row_identity_origin is None
    assert tuple(item.meaning for item in request.grouping_origins) == (
        "salesperson identity",
    )
    assert tuple(item.meaning for item in request.ordering_origins) == ("revenue",)
    assert tuple(item.meaning for item in request.output_origins) == (
        "salesperson identity",
        "revenue",
    )


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
    result_kind: str = "scalar",
    grouping_meanings: list[str] | None = None,
    answer_values: list[dict[str, object]] | None = None,
    supplied_values: list[dict[str, object]] | None = None,
    returned_value_refs: list[str] | None = None,
    ordering_value_refs: list[str] | None = None,
) -> dict[str, object]:
    values = answer_values or [
        {
            "value_ref": "v1",
            "meaning": "store count",
            "origin": {"kind": "question"},
        }
    ]
    request: dict[str, object] = {
        "result_kind": result_kind,
        "qualifying_row_kind": {
            "meaning": "store",
            "origin": {"kind": "question"},
        },
        "grouping_meanings": [
            {
                "meaning": meaning,
                "origin": {"kind": "question"},
                "grouping_kind": "related_entity_identity",
            }
            for meaning in grouping_meanings or []
        ],
        "return_request_basis": "The answer asks for the declared result.",
        "returned_result": {
            "kind": (
                "values"
                if result_kind == "scalar"
                else "identities_and_values"
                if returned_value_refs
                else "identities"
            )
        },
        "answer_values": values,
        "returned_value_refs": (
            ["v1"]
            if result_kind == "scalar"
            else returned_value_refs or []
        ),
        "ordering_value_refs": ordering_value_refs or [],
        "selection": {"kind": "all_results"},
        "universal_shape": "none",
    }
    if result_kind == "qualifying_instances":
        request["returned_candidate_identity"] = {
            "meaning": "store identity",
            "origin": {"kind": "question"},
        }
    return {
        "decision_basis": "The question states one complete factual request.",
        "outcome": {
            "kind": "question_meaning",
            "answer_requests": [request],
            "supplied_values": supplied_values or [],
        },
    }
