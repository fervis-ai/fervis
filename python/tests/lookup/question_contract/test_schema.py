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
    IdentifierType,
    SourceOrigin,
    SourceOriginKind,
    TemporalScopeType,
    TextType,
)


def test_semantic_question_contract_schema_is_valid_json_schema():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
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
        "set_graph",
        "grouping",
        "qualification",
        "ordering",
        "selection",
        "distinct_by",
        "outputs",
    )
    assert answer_request["properties"]["ordering"]["maxItems"] == 0
    assert answer_request["properties"]["distinct_by"]["maxItems"] == 0
    assert answer_request["properties"]["grouping"]["items"] == {"type": "null"}


def test_question_frame_row_projection_has_disjoint_identity_and_value_ownership():
    schema = build_semantic_question_frame_schema()
    complete = schema["properties"]["outcome"]["oneOf"][0]
    supplied_values = complete["properties"]["supplied_values"]
    assert tuple(supplied_values["properties"]) == (
        "operands",
        "selection_limits",
    )
    request = complete["properties"]["answer_requests"]["items"]["properties"][
        "request"
    ]["oneOf"][1]
    branches = request["properties"]["result"]["oneOf"]
    row_branch = next(
        branch
        for branch in branches
        if branch["properties"]["kind"]["enum"]
        == ["one_result_per_qualifying_row"]
    )
    assert "returned_meanings" not in row_branch["properties"]
    projection = row_branch["properties"]["projection"]
    assert tuple(projection["properties"]) == (
        "projection_basis",
        "candidate_identity",
        "explicitly_requested_values",
    )
    ordering = row_branch["properties"]["result_order"]["properties"]["ordering"]
    [ordered_by] = [
        variant
        for variant in ordering["oneOf"]
        if variant["properties"]["kind"]["enum"] == ["ordered_by"]
    ]
    ordering_kinds = {
        variant["properties"]["kind"]["enum"][0]
        for variant in ordered_by["properties"]["values"]["items"]["oneOf"]
    }
    assert ordering_kinds == {
        "requested_value_ref",
        "unreturned_ordering_meaning",
    }


def test_semantic_contract_exposes_only_operations_with_declared_input_operands():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
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
                (None,),
                1,
                (),
                "take_with_boundary_ties",
                "i3",
                "ordinary",
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


def test_quantified_row_identity_comparison_can_follow_its_own_relationship():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )

    row_condition = schema["$defs"]["row_condition_expression_3"]
    [identity_comparison] = [
        resolved
        for branch in row_condition["oneOf"]
        for resolved in (_resolved_schema(schema, branch),)
        if resolved["properties"]["kind"]["enum"] == ["input_comparison"]
        and resolved["properties"]["input"]["properties"]["input_ref"]["$ref"]
        == "#/$defs/identity_input_ref_value"
    ]
    identity_fact_ref = identity_comparison["properties"]["fact"]["$ref"]
    identity_fact = schema["$defs"][identity_fact_ref.rsplit("/", 1)[-1]]
    path_kinds = {
        branch["properties"]["kind"]["enum"][0]
        for branch in identity_fact["properties"]["identity_path"]["oneOf"]
    }

    assert path_kinds == {"candidate_instance", "related_instance"}


def test_semantic_contract_exposes_coverage_only_for_declared_coverage_shape():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "qualifying_instances",
                (),
                (),
                1,
                (),
                "all_results",
                None,
                "every_required_member_has_observation",
            ),
        ),
        input_refs=(),
    )

    qualification = _answer_request_schema(schema)["properties"]["qualification"]
    assert qualification["properties"]["kind"]["enum"] == ["coverage"]


def test_related_grouping_references_one_reserved_graph_set():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "grouped_results",
                ("related_entity_identity",),
                (None,),
                1,
                (),
                "first_rank_with_ties",
                None,
                "ordinary",
            ),
        ),
        input_refs=(),
    )

    grouping = _answer_request_schema(schema)["properties"]["grouping"]["items"]
    [related_identity] = grouping["oneOf"]
    properties = related_identity["properties"]
    assert properties["set_ref"] == {
        "type": "string",
        "pattern": r"^s(?:[2-9]|[1-9][0-9]+)$",
    }
    assert "identified_set_ref" not in properties
    assert "association_ref" not in properties


def test_collection_and_temporal_inputs_are_absent_from_generic_value_leaves():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
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
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
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
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )
    valid = _identity_count_contract()

    assert not tuple(Draft7Validator(schema).iter_errors(valid))
    assert "equals" in _comparison_operators(schema)
    assert schema["$defs"]["identity_input_ref_value"]["enum"] == ["i1"]


def test_identity_fact_is_a_valid_direct_output():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "qualifying_instances",
                (),
                (),
                1,
                (),
                "all_results",
                None,
                "ordinary",
            ),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )
    payload = _identity_count_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["set_graph"] = {
        "identity_input_relations": {"i1": None},
        "requested_output_relations": {},
        "other_related_sets": [],
    }
    identifier_fact = {
        "kind": "fact",
        "identity_path": {"kind": "candidate_instance"},
        "origin": answer_request["origin"],
    }
    answer_request["qualification"]["fact"] = identifier_fact
    answer_request["outputs"] = {
        "result_key_outputs": [{"expression": identifier_fact}],
        "requested_value_outputs": [],
    }

    assert not tuple(Draft7Validator(schema).iter_errors(payload))


def test_identity_fact_is_not_an_ordering_value():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "qualifying_instances",
                (),
                (None,),
                1,
                (),
                "all_results",
                None,
                "ordinary",
            ),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )
    payload = _identity_count_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["set_graph"] = {
        "identity_input_relations": {"i1": None},
        "requested_output_relations": {},
        "other_related_sets": [],
    }
    identifier_fact = {
        "kind": "fact",
        "identity_path": {"kind": "candidate_instance"},
        "origin": answer_request["origin"],
    }
    answer_request["qualification"]["fact"] = identifier_fact
    answer_request["ordering"] = [
        {
            "ordering_basis": "Compare candidate identifiers.",
            "expression": identifier_fact,
            "direction": "ascending",
        }
    ]
    answer_request["outputs"] = {
        "result_key_outputs": [{"expression": identifier_fact}],
        "requested_value_outputs": [],
    }

    assert tuple(Draft7Validator(schema).iter_errors(payload))


def test_mixed_result_schema_rejects_value_expression_in_result_key_output():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "grouped_results",
                ("related_entity_identity",),
                (),
                1,
                (("v1", "value"),),
                "all_results",
                None,
                "ordinary",
            ),
        ),
        input_refs=(),
    )
    payload = _grouped_ranking_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["ordering"] = []
    answer_request["selection"] = None
    answer_request["outputs"] = {
        "result_key_outputs": [
            {
                "expression": {
                    "kind": "fact",
                    "observed_for_ref": "s2",
                    "origin": answer_request["origin"],
                }
            }
        ],
        "requested_value_outputs": [
            {
                "output_ref": "v1",
                "expression": {
                    "kind": "aggregate",
                    "function": "count",
                    "argument": {"kind": "set_ref", "set_ref": "s1"},
                    "distinct_argument": False,
                }
            }
        ],
    }

    assert tuple(Draft7Validator(schema).iter_errors(payload))


def test_candidate_set_ref_cannot_be_redeclared_as_an_additional_set():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            ("fact_1", "scalar", (), (), 0, (("r1", "value"),), "all_results", None, "ordinary"),
        ),
        input_refs=("i1",),
        identity_input_refs=("i1",),
    )
    payload = _identity_count_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["set_graph"]["identity_input_relations"]["i1"]["set"][
        "id"
    ] = "s1"

    assert tuple(Draft7Validator(schema).iter_errors(payload))


def test_equal_origins_do_not_collapse_distinct_authored_sets() -> None:
    payload = _identity_count_contract()
    answer_request = payload["outcome"]["answer_requests"][0]
    answer_request["set_graph"]["identity_input_relations"]["i1"]["set"][
        "origin"
    ] = (
        answer_request["origin"]
    )
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
                    return_request_basis="Return the place count.",
                    relational_shape_basis="Count related place instances.",
                    result_grain_basis="One count for the population.",
                    ordering_request_basis="No ordering is requested.",
                    result_kind="scalar",
                candidate_set_origin=origin,
                grouping_refs=(),
                grouping_origins=(),
                grouping_kinds=(),
                grouping_value_shapes=(),
                ordering_group_refs=(),
            ordering_value_refs=(),
                ordering_origins=(),
                output_origins=(origin,),
                    output_kinds=("value",),
                    result_key_count=0,
                    requested_value_refs=("r1",),
                selection_kind="all_results",
                selection_limit_input_ref=None,
                relational_shape="ordinary",
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
    [identifier_fact] = fact.facts
    assert identifier_fact.owner_ref == "a1"
    assert identifier_fact.value_type == IdentifierType("s2")
    assert tuple(
        (item.id, item.from_set_ref, item.to_set_ref)
        for item in fact.associations
    ) == (("a1", "s1", "s2"),)


def test_equal_fact_origins_do_not_collapse_incompatible_relational_values() -> None:
    population_origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "in-person sales occurrences that happened this month",
    )
    time_origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "this month")
    meaning = ParsedSemanticQuestionMeaning(
        decision_basis="The question asks for one count over a supplied month.",
        answer_requests=(
                AnswerRequestMeaning(
                    requested_fact_id="fact_1",
                    return_request_basis="Return the sales count.",
                    relational_shape_basis="Count sales in the supplied month.",
                    result_grain_basis="One count for the population.",
                    ordering_request_basis="No ordering is requested.",
                    result_kind="scalar",
                candidate_set_origin=population_origin,
                grouping_refs=(),
                grouping_origins=(),
                grouping_kinds=(),
                grouping_value_shapes=(),
                ordering_group_refs=(),
            ordering_value_refs=(),
                ordering_origins=(),
                output_origins=(population_origin,),
                    output_kinds=("value",),
                    result_key_count=0,
                    requested_value_refs=("r1",),
                selection_kind="all_results",
                selection_limit_input_ref=None,
                relational_shape="ordinary",
            ),
        ),
        inputs=(InputTerm("i1", time_origin, "this month", TemporalScopeType()),),
        input_denotations=(
            InputDenotation(
                "input_denotation_1",
                "i1",
                "this month",
                "This supplies the time period.",
                None,
                InputDenotationKind.NON_IDENTITY_SCALAR,
            ),
        ),
    )
    origin = {
        "source": "question_context",
        "meaning": population_origin.meaning,
        "resolved_input_ref": None,
    }
    payload = {
        "decision_basis": "Count the qualifying sales occurrences.",
        "outcome": {
            "kind": "question_contract",
            "answer_requests": [
                {
                    "requested_fact_ref": "fact_1",
                    "origin": origin,
                    "candidate_set": {
                        "instance_kind": population_origin.meaning,
                        "instance_interpretation": "normal_business_instance"
                    },
                    "set_graph": {
                        "identity_input_relations": {},
                        "requested_output_relations": {},
                        "other_related_sets": [],
                    },
                    "grouping": [],
                    "qualification": {
                        "kind": "and",
                        "arguments": [
                            {
                                "kind": "fact",
                                "observed_for_ref": "s1",
                                "origin": origin,
                            },
                            {
                                "kind": "within",
                                "value": {
                                    "kind": "fact",
                                    "observed_for_ref": "s1",
                                    "origin": origin,
                                },
                                "scope": {"kind": "input_ref", "input_ref": "i1"},
                            },
                        ],
                    },
                    "ordering": [],
                    "selection": None,
                    "distinct_by": [],
                    "outputs": {
                        "result_key_outputs": [],
                        "requested_value_outputs": [
                            {
                                "output_ref": "r1",
                                "expression": {
                                    "kind": "aggregate",
                                    "function": "count",
                                    "argument": {"kind": "set_ref", "set_ref": "s1"},
                                    "distinct_argument": False,
                                }
                            }
                        ],
                    },
                }
            ],
        },
    }

    parsed = parse_semantic_question_contract(
        payload,
        meaning=meaning,
        question_context_texts=(
            "How many in-person sales happened this month?",
        ),
    )

    assert isinstance(parsed, ParsedSemanticQuestionContract)
    [requested_fact] = parsed.contract.requested_facts
    assert len(requested_fact.facts) == 2


def test_semantic_question_frame_schema_is_valid_and_basis_first():
    schema = build_semantic_question_frame_schema()
    Draft7Validator.check_schema(schema)
    complete = schema["properties"]["outcome"]["oneOf"][0]
    result = complete["properties"]["answer_requests"]["items"]
    assert tuple(complete["properties"]) == (
        "kind",
        "answer_requests",
        "supplied_values",
        "question_input_inventory_check",
    )
    assert tuple(result["properties"]) == (
        "return_request_basis",
        "relational_shape_basis",
        "request",
    )
    supplied = complete["properties"]["supplied_values"]
    branches_by_choice = supplied["properties"]
    assert set(branches_by_choice) == {
        "operands",
        "selection_limits",
    }
    operand_choices = branches_by_choice["operands"]["items"]["oneOf"]
    assert all(
        tuple(option["properties"])[:2] == ("meaning", "denotation_basis")
        for option in operand_choices
    )
    identity = operand_choices[0]
    assert tuple(identity["properties"]) == (
        "meaning",
        "denotation_basis",
        "entity_reference",
    )
    reference = identity["properties"]["entity_reference"]
    assert tuple(reference["properties"]) == ("instance_kind", "value")
    identity_values = reference["properties"]["value"]["oneOf"]
    assert tuple(identity_values[0]["properties"]) == (
        "kind",
        "identity_value",
        "origin",
    )
    assert tuple(identity_values[1]["properties"]) == (
        "kind",
        "identity_values",
        "origin",
    )


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
                    return_request_basis="Return each invoice identity.",
                    relational_shape_basis="Group qualifying invoices by identity.",
                    result_grain_basis="One row for each invoice.",
                    ordering_request_basis="No ordering is requested.",
                    result_kind="grouped_results",
                candidate_set_origin=candidate,
                grouping_refs=("g1",),
                grouping_origins=(grouping,),
                grouping_kinds=("qualifying_row_identity",),
                grouping_value_shapes=(None,),
                ordering_group_refs=(),
            ordering_value_refs=(),
                ordering_origins=(),
                output_origins=(grouping,),
                    output_kinds=("identity",),
                    result_key_count=1,
                    requested_value_refs=(),
                selection_kind="all_results",
                selection_limit_input_ref=None,
                relational_shape="ordinary",
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
    standard_request = complete["properties"]["answer_requests"]["items"][
        "properties"
    ]["request"]["oneOf"][1]
    result_branches = standard_request["properties"]["result"]["oneOf"]
    branches_by_kind = {
        branch["properties"]["kind"]["enum"][0]: branch
        for branch in result_branches
    }
    assert "grouping_meanings" not in branches_by_kind[
        "one_value_for_population"
    ]["properties"]
    assert "grouping_meanings" not in branches_by_kind[
        "one_result_per_qualifying_row"
    ]["properties"]
    assert (
        branches_by_kind["one_result_per_group"]["properties"][
            "grouping_meanings"
        ]["minItems"]
        == 1
    )
    grouped_projection = branches_by_kind["one_result_per_group"]["properties"][
        "projection"
    ]
    assert tuple(grouped_projection["properties"]) == (
        "projection_basis",
        "returned_grouping_keys",
        "explicitly_requested_values",
    )
    assert grouped_projection["properties"]["returned_grouping_keys"] == {
        "enum": ["all"]
    }
    grouping_branches = meaning_schema["$defs"]["grouping_origin"]["oneOf"]
    assert [
        branch["properties"]["grouping_kind"]["enum"][0]
        for branch in grouping_branches
    ] == ["qualifying_row_identity", "related_entity_identity", "non_identity_value"]
    non_identity = grouping_branches[2]["properties"]["grouping_value"]["oneOf"]
    assert [branch["properties"]["kind"]["enum"][0] for branch in non_identity] == [
        "observed_value",
        "temporal_bucket",
    ]
    qualifying = branches_by_kind["one_result_per_qualifying_row"]
    assert tuple(qualifying["properties"]) == (
        "kind",
        "result_candidates",
        "projection",
        "result_order",
    )
    scalar_values = branches_by_kind["one_value_for_population"]["properties"][
        "returned_meanings"
    ]
    assert scalar_values["minItems"] == scalar_values["maxItems"] == 1
    assert tuple(scalar_values["items"]["properties"]) == (
        "meaning",
        "origin",
        "meaning_ref",
    )

    contract_schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "grouped_results",
                ("qualifying_row_identity", "non_identity_value"),
                (),
                2,
                (),
                "all_results",
                None,
                "ordinary",
            ),
        ),
        input_refs=(),
    )
    grouping = _answer_request_schema(contract_schema)["properties"]["grouping"]
    assert grouping["minItems"] == 2
    assert grouping["maxItems"] == 2
    distinct_by = _answer_request_schema(contract_schema)["properties"]["distinct_by"]
    assert distinct_by["maxItems"] == 0


def test_candidate_result_frame_requires_a_projection():
    schema = build_semantic_question_frame_schema()
    complete = schema["properties"]["outcome"]["oneOf"][0]
    standard_request = complete["properties"]["answer_requests"]["items"][
        "properties"
    ]["request"]["oneOf"][1]
    branches = standard_request["properties"]["result"]["oneOf"]
    qualifying_branches = [
        branch
        for branch in branches
        if branch["properties"]["kind"]["enum"]
        == ["one_result_per_qualifying_row"]
    ]

    assert qualifying_branches
    assert all(
        set(branch["properties"]["projection"]["properties"])
        == {
            "projection_basis",
            "candidate_identity",
            "explicitly_requested_values",
        }
        for branch in qualifying_branches
    )


def test_ordered_question_frame_requires_a_declared_ordering_value():
    schema = build_semantic_question_frame_schema()
    complete = schema["properties"]["outcome"]["oneOf"][0]
    standard_request = complete["properties"]["answer_requests"]["items"][
        "properties"
    ]["request"]["oneOf"][1]
    branches = standard_request["properties"]["result"]["oneOf"]
    ordered_values = [
        branch
        for branch in branches
        if branch["properties"]["kind"]["enum"][0]
        != "one_value_for_population"
    ]

    assert ordered_values
    assert all(
        next(
            option
            for option in branch["properties"]["result_order"]["properties"][
                "ordering"
            ]["oneOf"]
            if option["properties"]["kind"]["enum"] == ["ordered_by"]
        )["properties"]["values"]["minItems"]
        >= 1
        for branch in ordered_values
    )


def test_grouped_result_schema_rejects_row_grain_ordering_values():
    schema = build_semantic_question_contract_schema(
        answer_request_specs=(
            (
                "fact_1",
                "grouped_results",
                ("related_entity_identity",),
                (None,),
                1,
                (),
                "first_rank_with_ties",
                None,
                "ordinary",
            ),
        ),
        input_refs=(),
    )
    payload = _grouped_ranking_contract()

    assert not tuple(Draft7Validator(schema).iter_errors(payload))
    payload["outcome"]["answer_requests"][0]["ordering"][0]["expression"] = {
        "kind": "fact",
        "observed_for_ref": "s1",
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
        kind="categorical_value",
    )
    ambiguous_text = _question_frame_payload(operand="approved", kind="text")

    assert not tuple(Draft7Validator(schema).iter_errors(identity))
    assert not tuple(Draft7Validator(schema).iter_errors(property_value))
    assert tuple(Draft7Validator(schema).iter_errors(ambiguous_text))


def test_semantic_question_frame_couples_boolean_type_to_boolean_operand():
    schema = build_semantic_question_frame_schema()
    valid = _question_frame_payload(operand="true", kind="boolean")
    adjective = _question_frame_payload(operand="approved", kind="boolean")
    property_value = _question_frame_payload(
        operand="approved", kind="categorical_value"
    )

    assert not tuple(Draft7Validator(schema).iter_errors(valid))
    assert tuple(Draft7Validator(schema).iter_errors(adjective))
    assert not tuple(Draft7Validator(schema).iter_errors(property_value))


def test_semantic_question_frame_couples_numeric_types_to_numeric_operands():
    schema = build_semantic_question_frame_schema()
    valid = _question_frame_payload(operand="125.50", kind="number")
    metric_name = _question_frame_payload(operand="revenue", kind="number")

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
                "kind": "single_identity",
                "identity_value": operand,
                "origin": {"kind": "question"},
            },
        }
        supplied_values = {
            "operands": [supplied_value],
            "selection_limits": [],
        }
    else:
        supplied_value["non_entity_value"] = {
            "kind": kind,
            "value": {
                "operands": [operand],
                "origin": {"kind": "question"},
            },
        }
        supplied_values = {
            "operands": [supplied_value],
            "selection_limits": [],
        }
    return {
        "decision_basis": "The question asks for one count.",
        "outcome": {
            "kind": "question_meaning",
            "answer_requests": [
                {
                    "return_request_basis": "The answer asks for one count.",
                    "relational_shape_basis": "Count ordinary observations.",
                    "request": {
                        "relational_shape": "ordinary",
                        "result_grain_basis": "One count over all observations.",
                        "result": {
                            "kind": "one_value_for_population",
                            "population_rows": {
                                "instance_kind": "observations",
                                "origin": {"kind": "question"},
                            },
                            "returned_meanings": [
                                {
                                    "meaning_ref": "r1",
                                    "meaning": "observation count",
                                    "origin": {"kind": "question"},
                                }
                            ],
                            "result_order": {
                                "ordering_request_basis": "No ordering is requested.",
                                "ordering": {"kind": "no_ordering_requested"},
                                "selection": {"kind": "all_results"},
                            },
                        },
                    },
                }
            ],
            "supplied_values": supplied_values,
            "question_input_inventory_check": {
                "all_input_like_phrases_declared": True,
            },
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
                    "candidate_set": {
                        "instance_kind": "observations related to the supplied place",
                        "instance_interpretation": "normal_business_instance"
                    },
                    "set_graph": {
                        "identity_input_relations": {
                            "i1": {
                                "association": {"id": "a1", "origin": origin},
                                "set": {
                                    "id": "s2",
                                    "instance_kind": "place",
                                    "origin": {
                                        "source": "question_context",
                                        "meaning": "places",
                                        "resolved_input_ref": None,
                                    },
                                },
                                "related_sets": [],
                            }
                        },
                        "requested_output_relations": {},
                        "other_related_sets": [],
                    },
                    "qualification": {
                        "kind": "input_comparison",
                        "input": {
                            "kind": "input_ref",
                            "input_ref": "i1",
                            "operand_meaning": "the supplied place",
                            "instance_kind": "place",
                        },
                        "operator": "equals",
                        "fact": {
                            "kind": "fact",
                            "identity_path": {
                                "kind": "related_instance",
                                "association_ref": "a1",
                            },
                            "origin": origin,
                        },
                    },
                    "grouping": [],
                    "ordering": [],
                    "selection": None,
                    "distinct_by": [],
                    "outputs": {
                        "result_key_outputs": [],
                        "requested_value_outputs": [
                            {
                                "output_ref": "r1",
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
                    },
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
                        "instance_kind": "sales",
                        "instance_interpretation": "normal_business_instance"
                    },
                    "set_graph": {
                        "identity_input_relations": {},
                        "requested_output_relations": {},
                        "other_related_sets": [
                            {
                                "association": {"id": "a1", "origin": origin},
                                "set": {
                                    "id": "s2",
                                    "instance_kind": "staff",
                                    "origin": origin,
                                },
                                "related_sets": [],
                            }
                        ]
                    },
                    "grouping": [
                        {
                            "id": "g1",
                            "grouping_basis": "Group each sale by its staff identity.",
                            "kind": "related_instance_identity",
                            "set_ref": "s2",
                        }
                    ],
                    "qualification": None,
                    "ordering": [
                        {
                            "ordering_basis": "Compare groups by summed sales.",
                            "expression": aggregate,
                            "direction": "descending",
                        }
                    ],
                    "selection": {"kind": "first_rank_with_ties"},
                    "distinct_by": [],
                    "outputs": {
                        "result_key_outputs": [
                            {"expression": {"kind": "group_ref", "ref": "g1"}}
                        ],
                        "requested_value_outputs": [],
                    },
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


def test_temporal_scope_is_one_complete_operand_not_endpoint_alternatives():
    schema = build_semantic_question_frame_schema()
    valid = _question_frame_payload(operand="March 1 through March 31, 2026", kind="temporal_scope")
    assert Draft7Validator(schema).is_valid(valid)
    from copy import deepcopy
    invalid = deepcopy(valid)
    invalid["outcome"]["supplied_values"]["operands"][0]["non_entity_value"]["value"]["operands"] = ["March 1, 2026", "March 31, 2026"]
    assert not Draft7Validator(schema).is_valid(invalid)
