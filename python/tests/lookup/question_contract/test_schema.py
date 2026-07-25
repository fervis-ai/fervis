from jsonschema import Draft7Validator

from fervis.lookup.question_contract.schema import (
    build_semantic_question_contract_schema_for_meaning,
    build_semantic_question_contract_schema,
    build_semantic_question_frame_schema,
)
from fervis.lookup.question_contract.parser import (
    AnswerRequestMeaning,
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
)
from fervis.lookup.question_contract.model import (
    InputDenotation,
    InputDenotationKind,
    InputTerm,
)
from fervis.lookup.semantic_types import (
    SourceOrigin,
    SourceOriginKind,
    TextType,
)


def test_semantic_question_contract_schema_is_valid_json_schema():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), 0, 1, "all_results", None, "none"),
        ),
        input_refs=("i1",),
        temporal_scope_input_refs=("i1",),
    )
    Draft7Validator.check_schema(schema)
    complete = schema["properties"]["outcome"]["oneOf"][0]
    answer_request_ref = complete["properties"]["answer_requests"]["items"]["oneOf"][0][
        "$ref"
    ]
    answer_request = schema["$defs"][answer_request_ref.rsplit("/", 1)[-1]]
    assert tuple(complete["properties"]) == ("kind", "answer_requests")
    assert tuple(answer_request["properties"]) == (
        "requested_fact_ref",
        "origin",
        "candidate_set",
        "grouping",
        "other_sets",
        "other_associations",
        "qualification",
        "ordering",
        "selection",
        "distinct_by",
        "outputs",
    )
    assert answer_request["properties"]["ordering"]["maxItems"] == 0
    assert answer_request["properties"]["distinct_by"]["maxItems"] == 0
    assert answer_request["properties"]["grouping"]["items"] == {"type": "null"}


def test_semantic_contract_exposes_only_operations_with_declared_input_operands():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), 0, 1, "all_results", None, "none"),
        ),
        input_refs=("i1",),
        temporal_scope_input_refs=("i1",),
    )

    assert "collection_input_ref_value" not in schema["$defs"]
    assert _condition_kinds(schema) >= {"within"}
    assert "coverage" not in _condition_kinds(schema)
    assert "in" not in _comparison_operators(schema)
    assert "take_with_boundary_ties" not in _selection_kinds(schema)


def test_semantic_contract_exposes_input_operations_when_their_operands_exist():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "qualifying_instances",
                (),
                1,
                1,
                "take_with_boundary_ties",
                "i3",
                "none",
            ),
        ),
        input_refs=("i1", "i2", "i3"),
        collection_input_refs=("i1",),
        temporal_scope_input_refs=("i2",),
    )

    assert schema["$defs"]["collection_input_ref_value"]["enum"] == ["i1"]
    assert schema["$defs"]["temporal_scope_input_ref_value"]["enum"] == ["i2"]
    assert schema["$defs"]["input_ref"]["properties"]["input_ref"]["enum"] == ["i3"]
    assert "within" in _condition_kinds(schema)
    assert "in" in _comparison_operators(schema)
    assert "take_with_boundary_ties" in _selection_kinds(schema)


def test_semantic_contract_exposes_coverage_only_for_declared_coverage_shape():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "qualifying_instances",
                (),
                0,
                1,
                "all_results",
                None,
                "every_required_member_has_observation",
            ),
        ),
        input_refs=(),
    )

    assert "coverage" in _condition_kinds(schema)


def test_collection_and_temporal_inputs_are_absent_from_generic_value_leaves():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), 0, 1, "all_results", None, "none"),
        ),
        input_refs=("i1", "i2"),
        collection_input_refs=("i1",),
        temporal_scope_input_refs=("i2",),
    )

    assert "input_ref" not in schema["$defs"]
    for definition_name in ("row_value_expression_0", "value_expression_0"):
        branches = schema["$defs"][definition_name]["oneOf"]
        assert all(branch.get("$ref") != "#/$defs/input_ref" for branch in branches)


def test_semantic_contract_without_inputs_has_no_input_expression_branch():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), 0, 1, "all_results", None, "none"),
        ),
        input_refs=(),
    )

    assert "input_ref" not in schema["$defs"]
    assert "within" not in _condition_kinds(schema)
    for definition_name in ("row_value_expression_0", "value_expression_0"):
        branches = schema["$defs"][definition_name]["oneOf"]
        assert all(branch.get("$ref") != "#/$defs/input_ref" for branch in branches)


def test_identity_input_is_consumed_by_an_identifier_comparison():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), 0, 1, "all_results", None, "none"),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )
    valid = _identity_count_contract()

    assert not tuple(Draft7Validator(schema).iter_errors(valid))
    assert "equals" in _comparison_operators(schema)
    assert schema["$defs"]["identity_input_ref_value"]["enum"] == ["i1"]


def test_candidate_set_ref_cannot_be_redeclared_as_an_additional_set():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), 0, 1, "all_results", None, "none"),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )
    payload = _identity_count_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["other_sets"].append(
        {"id": "s1", "origin": answer_request["origin"]}
    )

    assert tuple(Draft7Validator(schema).iter_errors(payload))


def test_equal_origins_do_not_collapse_distinct_authored_sets() -> None:
    payload = _identity_count_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["other_sets"][0]["origin"] = answer_request["origin"]
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        answer_request["origin"]["meaning"],
    )
    input_origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "Nairobi")
    meaning = ParsedSemanticQuestionMeaning(
        decision_basis="The question asks for one count.",
        answer_requests=(
            AnswerRequestMeaning(
                requested_fact_id="fact_1",
                result_kind="scalar",
                candidate_set_origin=origin,
                grouping_origins=(),
                grouping_kinds=(),
                row_identity_origin=None,
                ordering_origins=(),
                output_origins=(origin,),
                selection_kind="all_results",
                selection_limit_input_ref=None,
                universal_shape="none",
            ),
        ),
        inputs=(InputTerm("i1", input_origin, "Nairobi", TextType()),),
        input_denotations=(
            InputDenotation(
                "input_denotation_1",
                "i1",
                "the supplied place",
                "Nairobi names one place.",
                "place",
                InputDenotationKind.IDENTITY_REFERENCE,
            ),
        ),
    )

    parsed = parse_semantic_question_contract(
        payload,
        meaning=meaning,
        question_context_texts=("How many observations are related to Nairobi?",),
    )

    assert isinstance(parsed, ParsedSemanticQuestionContract)
    [fact] = parsed.contract.requested_facts
    assert tuple(item.id for item in fact.sets) == ("s1", "s2")
    assert tuple(
        (item.id, item.from_set_ref, item.to_set_ref)
        for item in fact.associations
    ) == (("a1", "s1", "s2"),)


def test_semantic_question_frame_schema_is_valid_and_basis_first():
    schema = build_semantic_question_frame_schema()
    Draft7Validator.check_schema(schema)
    complete = schema["properties"]["outcome"]["oneOf"][0]
    result = complete["properties"]["answer_requests"]["items"]["oneOf"][0]
    assert tuple(complete["properties"]) == (
        "kind",
        "answer_requests",
        "supplied_values",
    )
    assert tuple(result["properties"]) == (
        "result_kind",
        "qualifying_row_kind",
        "grouping_meanings",
        "return_request_basis",
        "returned_result",
        "answer_values",
        "returned_value_refs",
        "ordering_value_refs",
        "selection",
        "universal_shape",
    )
    assert result["properties"]["selection"]["properties"]["kind"]["enum"] == [
        "all_results"
    ]
    supplied = complete["properties"]["supplied_values"]["items"]
    branches_by_choice = {
        next(
            name
            for name in ("entity_reference", "non_entity_value")
            if name in branch["properties"]
        ): branch
        for branch in supplied["oneOf"]
    }
    assert set(branches_by_choice) == {"entity_reference", "non_entity_value"}
    assert all(
        tuple(branch["properties"])[:2] == ("meaning", "denotation_basis")
        for branch in branches_by_choice.values()
    )
    assert all(
        choice in branches_by_choice[choice]["required"]
        for choice in branches_by_choice
    )
    identity = branches_by_choice["entity_reference"]["properties"][
        "entity_reference"
    ]
    assert tuple(identity["properties"]) == ("instance_kind", "value")
    assert "value_type" not in identity["properties"]["value"]["properties"]


def test_grouping_schema_does_not_infer_identity_relation_from_origin_text():
    candidate = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "invoice occurrences",
    )
    grouping = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "each invoice identity",
    )
    meaning = ParsedSemanticQuestionMeaning(
        decision_basis="Return one result for each candidate invoice.",
        answer_requests=(
            AnswerRequestMeaning(
                requested_fact_id="fact_1",
                result_kind="grouped_results",
                candidate_set_origin=candidate,
                grouping_origins=(grouping,),
                grouping_kinds=("qualifying_row_identity",),
                row_identity_origin=None,
                ordering_origins=(),
                output_origins=(grouping,),
                selection_kind="all_results",
                selection_limit_input_ref=None,
                universal_shape="none",
            ),
        ),
        inputs=(),
        input_denotations=(),
    )

    schema = build_semantic_question_contract_schema_for_meaning(meaning)

    assert _grouping_kinds(schema) == {"candidate_instance_identity"}


def test_grouped_result_frame_owns_the_exact_grouping_shape():
    meaning_schema = build_semantic_question_frame_schema()
    complete = meaning_schema["properties"]["outcome"]["oneOf"][0]
    result_branches = complete["properties"]["answer_requests"]["items"]["oneOf"]
    branches_by_kind = {
        branch["properties"]["result_kind"]["enum"][0]: branch
        for branch in result_branches
    }
    assert (
        branches_by_kind["scalar"]["properties"]["grouping_meanings"]["maxItems"] == 0
    )
    assert (
        branches_by_kind["qualifying_instances"]["properties"]["grouping_meanings"][
            "maxItems"
        ]
        == 0
    )
    assert (
        branches_by_kind["grouped_results"]["properties"]["grouping_meanings"][
            "minItems"
        ]
        == 1
    )
    qualifying = branches_by_kind["qualifying_instances"]
    assert tuple(qualifying["properties"]) == (
        "result_kind",
        "qualifying_row_kind",
        "grouping_meanings",
        "returned_candidate_identity",
        "return_request_basis",
        "returned_result",
        "answer_values",
        "returned_value_refs",
        "ordering_value_refs",
        "selection",
        "universal_shape",
    )
    scalar_values = branches_by_kind["scalar"]["properties"]["answer_values"]
    assert scalar_values["minItems"] == scalar_values["maxItems"] == 1
    assert tuple(scalar_values["items"]["properties"]) == (
        "value_ref",
        "meaning",
        "origin",
    )

    contract_schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "grouped_results",
                ("qualifying_row_identity", "non_identity_value"),
                0,
                2,
                "all_results",
                None,
                "none",
            ),
        ),
        input_refs=(),
    )
    grouping = _answer_request_schema(contract_schema)["properties"]["grouping"]
    assert grouping["minItems"] == 2
    assert grouping["maxItems"] == 2
    distinct_by = _answer_request_schema(contract_schema)["properties"]["distinct_by"]
    assert distinct_by["maxItems"] == 0


def test_qualifying_instance_frame_requires_its_returned_identity():
    schema = build_semantic_question_frame_schema()
    complete = schema["properties"]["outcome"]["oneOf"][0]
    branches = complete["properties"]["answer_requests"]["items"]["oneOf"]
    qualifying_branches = [
        branch
        for branch in branches
        if branch["properties"]["result_kind"]["enum"]
        == ["qualifying_instances"]
    ]

    assert qualifying_branches
    assert all(
        "returned_candidate_identity" in branch["required"]
        for branch in qualifying_branches
    )


def test_ordered_question_frame_requires_a_declared_ordering_value():
    schema = build_semantic_question_frame_schema()
    complete = schema["properties"]["outcome"]["oneOf"][0]
    branches = complete["properties"]["answer_requests"]["items"]["oneOf"]
    ordered_branches = [
        branch
        for branch in branches
        if branch["properties"]["selection"]["properties"]["kind"]["enum"][0]
        in {"first_rank_with_ties", "take_with_boundary_ties"}
    ]

    assert ordered_branches
    assert all(
        branch["properties"]["answer_values"]["minItems"] >= 1
        for branch in ordered_branches
    )


def test_grouped_result_schema_rejects_row_grain_ordering_values():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "grouped_results",
                ("related_entity_identity",),
                1,
                1,
                "first_rank_with_ties",
                None,
                "none",
            ),
        ),
        input_refs=(),
    )
    payload = _grouped_ranking_contract()

    assert not tuple(Draft7Validator(schema).iter_errors(payload))
    payload["outcome"]["answer_requests"][0]["ordering"][0]["expression"] = {
        "kind": "fact",
        "observed_for_ref": "s1",
        "value_type": {"kind": "decimal", "measure": {"kind": "unitless"}},
        "origin": payload["outcome"]["answer_requests"][0]["origin"],
    }

    assert tuple(Draft7Validator(schema).iter_errors(payload))


def test_semantic_question_frame_exposes_semantic_string_value_kinds():
    schema = build_semantic_question_frame_schema()
    identity = _question_frame_payload(
        operand="Nairobi",
        denotation_kind="identity_reference",
        instance_kind="area",
    )
    property_value = _question_frame_payload(
        operand="approved",
        kind="property_value",
    )
    ambiguous_text = _question_frame_payload(operand="approved", kind="text")

    assert not tuple(Draft7Validator(schema).iter_errors(identity))
    assert not tuple(Draft7Validator(schema).iter_errors(property_value))
    assert tuple(Draft7Validator(schema).iter_errors(ambiguous_text))


def test_semantic_question_frame_couples_boolean_type_to_boolean_operand():
    schema = build_semantic_question_frame_schema()
    valid = _question_frame_payload(operand="true", kind="boolean")
    adjective = _question_frame_payload(operand="approved", kind="boolean")
    property_value = _question_frame_payload(operand="approved", kind="property_value")

    assert not tuple(Draft7Validator(schema).iter_errors(valid))
    assert tuple(Draft7Validator(schema).iter_errors(adjective))
    assert not tuple(Draft7Validator(schema).iter_errors(property_value))


def test_semantic_question_frame_couples_numeric_types_to_numeric_operands():
    schema = build_semantic_question_frame_schema()
    valid = _question_frame_payload(
        operand="125.50",
        value_type={
            "kind": "decimal",
            "measure": {
                "kind": "money",
                "currency": {"kind": "contextual"},
            },
        },
    )
    metric_name = _question_frame_payload(
        operand="revenue",
        value_type={
            "kind": "decimal",
            "measure": {
                "kind": "money",
                "currency": {"kind": "contextual"},
            },
        },
    )

    assert not tuple(Draft7Validator(schema).iter_errors(valid))
    assert tuple(Draft7Validator(schema).iter_errors(metric_name))


def _answer_request_schema(schema: dict[str, object]) -> dict[str, object]:
    complete = schema["properties"]["outcome"]["oneOf"][0]
    ref = complete["properties"]["answer_requests"]["items"]["oneOf"][0]["$ref"]
    return schema["$defs"][ref.rsplit("/", 1)[-1]]


def _question_frame_payload(
    *,
    operand: str,
    kind: str | None = None,
    value_type: dict[str, object] | None = None,
    denotation_kind: str = "scalar",
    instance_kind: str | None = None,
) -> dict[str, object]:
    supplied_value: dict[str, object] = {
        "meaning": "the supplied value",
        "denotation_basis": "This states what the supplied value denotes.",
    }
    if denotation_kind == "identity_reference":
        supplied_value["entity_reference"] = {
            "instance_kind": instance_kind,
            "value": {
                "operands": [operand],
                "origin": {"kind": "question"},
            },
        }
    else:
        supplied_value["non_entity_value"] = {
            "value": {
                "operands": [operand],
                "value_type": value_type or {"kind": kind},
                "origin": {"kind": "question"},
            }
        }
    return {
        "decision_basis": "The question asks for one count.",
        "outcome": {
            "kind": "question_meaning",
            "answer_requests": [
                {
                    "result_kind": "scalar",
                    "qualifying_row_kind": {
                        "meaning": "observations",
                        "origin": {"kind": "question"},
                    },
                    "grouping_meanings": [],
                    "return_request_basis": "The answer asks for one count.",
                    "returned_result": {"kind": "values"},
                    "answer_values": [
                        {
                            "value_ref": "v1",
                            "meaning": "observation count",
                            "origin": {"kind": "question"},
                        }
                    ],
                    "returned_value_refs": ["v1"],
                    "ordering_value_refs": [],
                    "selection": {"kind": "all_results"},
                    "universal_shape": "none",
                }
            ],
            "supplied_values": [supplied_value],
        },
    }


def _identity_count_contract() -> dict[str, object]:
    origin = {
        "source": "question_context",
        "meaning": "observations related to the supplied place",
        "resolved_input_ref": None,
    }
    return {
        "decision_basis": "The supplied identity qualifies related observations.",
        "outcome": {
            "kind": "question_contract",
            "answer_requests": [
                {
                    "requested_fact_ref": "fact_1",
                    "origin": origin,
                    "other_sets": [
                        {
                            "id": "s2",
                            "origin": {
                                "source": "question_context",
                                "meaning": "places",
                                "resolved_input_ref": None,
                            },
                        }
                    ],
                    "other_associations": [
                        {
                            "id": "a1",
                            "from_set_ref": "s1",
                            "to_set_ref": "s2",
                            "origin": origin,
                        }
                    ],
                    "candidate_set": {
                        "instance_interpretation": "normal_business_instance"
                    },
                    "qualification": {
                        "kind": "input_comparison",
                        "input": {"kind": "input_ref", "input_ref": "i1"},
                        "operator": "equals",
                        "fact": {
                            "kind": "fact",
                            "observed_for_ref": "a1",
                            "value_type": {"kind": "identifier", "set_ref": "s2"},
                            "origin": origin,
                        },
                    },
                    "grouping": [],
                    "ordering": [],
                    "selection": None,
                    "distinct_by": [],
                    "outputs": [
                        {
                            "expression": {
                                "kind": "aggregate",
                                "function": "count",
                                "argument": {
                                    "kind": "set_ref",
                                    "set_ref": "s1",
                                },
                                "distinct_argument": False,
                            }
                        }
                    ],
                }
            ],
        },
    }


def _grouped_ranking_contract() -> dict[str, object]:
    origin = {
        "source": "question_context",
        "meaning": "sales grouped by staff",
        "resolved_input_ref": None,
    }
    aggregate = {
        "kind": "aggregate",
        "function": "sum",
        "argument": {
            "kind": "fact",
            "observed_for_ref": "s1",
            "value_type": {"kind": "decimal", "measure": {"kind": "unitless"}},
            "origin": origin,
        },
        "distinct_argument": False,
    }
    return {
        "decision_basis": "Rank staff by aggregate sales.",
        "outcome": {
            "kind": "question_contract",
            "answer_requests": [
                {
                    "requested_fact_ref": "fact_1",
                    "origin": origin,
                    "candidate_set": {
                        "instance_interpretation": "normal_business_instance"
                    },
                    "grouping": [
                        {
                            "id": "g1",
                            "grouping_basis": "Group each sale by its staff identity.",
                            "kind": "related_instance_identity",
                            "identified_set": {"id": "s2"},
                            "association": {"id": "a1", "origin": origin},
                        }
                    ],
                    "other_sets": [],
                    "other_associations": [],
                    "qualification": None,
                    "ordering": [
                        {
                            "expression": aggregate,
                            "direction": "descending",
                        }
                    ],
                    "selection": {"kind": "first_rank_with_ties"},
                    "distinct_by": [],
                    "outputs": [{"expression": {"kind": "group_ref", "ref": "g1"}}],
                }
            ],
        },
    }


def _condition_kinds(schema: dict[str, object]) -> set[str]:
    return {
        _resolved_schema(schema, branch)["properties"]["kind"]["enum"][0]
        for branch in schema["$defs"]["row_condition_expression_3"]["oneOf"]
    }


def _comparison_operators(schema: dict[str, object]) -> set[str]:
    return {
        operator
        for branch in schema["$defs"]["row_condition_expression_3"]["oneOf"]
        if _resolved_schema(schema, branch)["properties"]["kind"]["enum"][0]
        in {"input_comparison", "value_comparison"}
        for operator in _resolved_schema(schema, branch)["properties"]["operator"][
            "enum"
        ]
    }


def _resolved_schema(
    schema: dict[str, object], value: dict[str, object]
) -> dict[str, object]:
    ref = value.get("$ref")
    if not isinstance(ref, str):
        return value
    return schema["$defs"][ref.rsplit("/", 1)[-1]]


def _selection_kinds(schema: dict[str, object]) -> set[str]:
    selection = _answer_request_schema(schema)["properties"]["selection"]
    if selection.get("type") != "object":
        return set()
    return {selection["properties"]["kind"]["enum"][0]}


def _grouping_kinds(schema: dict[str, object]) -> set[str]:
    grouping = _answer_request_schema(schema)["properties"]["grouping"]["items"]
    return {
        branch["properties"]["kind"]["enum"][0] for branch in grouping["oneOf"]
    }
