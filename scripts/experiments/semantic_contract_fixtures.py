"""Canonical semantic Question Contract fixtures shared by step experiments."""

from __future__ import annotations

from fervis.lookup.question_contract.semantic_parser import (
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)


def count_contract(
    *,
    question: str,
    operand: str | list[str],
    candidate_meaning: str,
    fact_meaning: str,
    fact_value_type: dict[str, object],
    input_value_type: dict[str, object],
    operator: str,
    identity_set_meaning: str | None = None,
) -> ParsedSemanticQuestionContract:
    collection = isinstance(operand, list)
    identity_origin = (
        _origin(identity_set_meaning) if identity_set_meaning is not None else None
    )
    count_meaning = f"{candidate_meaning} count"
    meaning = parse_semantic_question_frame(
        {
            "decision_basis": "The request asks for one count.",
            "outcome": {
                "kind": "question_meaning",
                "answer_requests": [
                    {
                        "result_kind": "grouped_results" if collection else "scalar",
                        "qualifying_row_kind": _frame_meaning(candidate_meaning),
                        "grouping_meanings": (
                            [_frame_meaning(identity_set_meaning)]
                            if collection
                            else []
                        ),
                        "return_request_basis": (
                            "The answer returns each group identity and its count."
                            if collection
                            else "The answer returns the count."
                        ),
                        "returned_result": {
                            "kind": (
                                "identities_and_values" if collection else "values"
                            )
                        },
                        "answer_values": [
                            {
                                "value_ref": "v1",
                                "meaning": count_meaning,
                                "origin": {"kind": "question"},
                            }
                        ],
                        "returned_value_refs": ["v1"],
                        "ordering_value_refs": [],
                        "selection": {"kind": "all_results"},
                        "universal_shape": "none",
                    }
                ],
                "supplied_values": [
                    {
                        "meaning": fact_meaning,
                        "denotation": {
                            "basis": (
                                "The supplied value identifies one instance."
                                if identity_origin is not None
                                else "The supplied value is a scalar constraint."
                            ),
                            "kind": (
                                "identity_reference"
                                if identity_origin is not None
                                else "scalar"
                            ),
                            "instance_kind": identity_set_meaning,
                        },
                        "value": {
                            "operands": operand if collection else [operand],
                            "value_type": input_value_type,
                            "origin": {"kind": "question"},
                        },
                    }
                ],
            },
        },
        question_context_texts=(question,),
    )
    if not isinstance(meaning, ParsedSemanticQuestionMeaning):
        raise ValueError("experiment requires complete requested-result meaning")
    other_sets: list[dict[str, object]] = []
    other_associations: list[dict[str, object]] = []
    scalar_fact = {
        "kind": "fact",
        "observed_for_ref": "s1",
        "value_type": fact_value_type,
        "origin": _origin(fact_meaning),
    }
    input_ref = {"kind": "input_ref", "input_ref": "i1"}
    identity_fact: dict[str, object] | None = None
    if identity_origin is not None:
        identity_set_ref = "s1"
        observed_for_ref = "s1"
        if identity_set_meaning != candidate_meaning:
            identity_set_ref = "s2"
            if not collection:
                other_sets.append({"id": identity_set_ref, "origin": identity_origin})
                other_associations.append(
                    {
                        "id": "a1",
                        "from_set_ref": "s1",
                        "to_set_ref": identity_set_ref,
                        "origin": _origin(fact_meaning),
                    }
                )
            observed_for_ref = "a1"
        identity_fact = {
            "kind": "fact",
            "observed_for_ref": observed_for_ref,
            "value_type": {"kind": "identifier", "set_ref": identity_set_ref},
            "origin": _origin(fact_meaning),
        }
    qualification = (
        {
            "kind": "input_comparison",
            "input": input_ref,
            "operator": "in" if collection else "equals",
            "fact": identity_fact,
        }
        if identity_fact is not None
        else {"kind": "within", "value": scalar_fact, "scope": input_ref}
        if operator == "within"
        else {
            "kind": "input_comparison",
            "input": input_ref,
            "operator": operator,
            "fact": scalar_fact,
        }
    )
    grouping = (
        [
            {
                "id": "g1",
                "kind": "related_instance_identity",
                "grouping_basis": (
                    f"One result group for each exact {identity_set_meaning} identity."
                ),
                "identified_set": {"id": identity_set_ref},
                "association": {
                    "id": "a1",
                    "origin": _origin(fact_meaning),
                },
            }
        ]
        if collection and identity_fact is not None
        else [
            {
                "id": "g1",
                "kind": "value",
                "grouping_basis": f"One result group for each {fact_meaning} value.",
                "expression": scalar_fact,
            }
        ]
        if collection
        else []
    )
    count = {
        "kind": "aggregate",
        "function": "count",
        "argument": {"kind": "set_ref", "set_ref": "s1"},
        "distinct_argument": False,
    }
    outputs = (
        [
            {"expression": {"kind": "group_ref", "ref": "g1"}},
            {"expression": count},
        ]
        if collection
        else [{"expression": count}]
    )
    parsed = parse_semantic_question_contract(
        {
            "decision_basis": "The relational contract counts qualifying rows.",
            "outcome": {
                "kind": "question_contract",
                "answer_requests": [
                    {
                        "requested_fact_ref": "fact_1",
                        "origin": _origin(f"qualifying {candidate_meaning}"),
                        "other_sets": other_sets,
                        "other_associations": other_associations,
                        "candidate_set": {
                            "instance_interpretation": "normal_business_instance"
                        },
                        "qualification": qualification,
                        "grouping": grouping,
                        "ordering": [],
                        "selection": None,
                        "distinct_by": [],
                        "outputs": outputs,
                    }
                ],
            },
        },
        meaning=meaning,
        question_context_texts=(question,),
    )
    if not isinstance(parsed, ParsedSemanticQuestionContract):
        raise ValueError("experiment requires a complete semantic contract")
    return parsed


def _origin(meaning: str | None) -> dict[str, object]:
    if meaning is None:
        raise ValueError("semantic origin meaning is required")
    return {
        "source": "question_context",
        "meaning": meaning,
        "resolved_input_ref": None,
    }


def _frame_meaning(meaning: str | None) -> dict[str, object]:
    if meaning is None:
        raise ValueError("semantic frame meaning is required")
    return {
        "meaning": meaning,
        "origin": {"kind": "question"},
    }


__all__ = ["count_contract"]
