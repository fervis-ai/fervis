from __future__ import annotations

from tests.lookup.source_binding._plan_member_targets_fixtures import (
    PlanSelectionSet,
    SourceBindingTurnPrompt,
    ValidationError,
    _binding_targets,
    _closed_key_grouped_staff_sales_request,
    _minimal_source_invocation,
    _set_difference_request,
    _source_binding_outcome_schema,
    _source_binding_plan_payload,
    _source_invocation_variants_by_target,
    _test_fact_binding,
    _test_plan_families,
    build_source_binding_schema,
    pytest,
    replace,
    source_binding_candidates_xml,
    validate,
)


def test_source_binding_schema_uses_compact_role_target_handles():
    request = _set_difference_request()
    prompt = SourceBindingTurnPrompt(request)
    targets = _binding_targets(prompt)
    invocation_schemas = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )

    target_ids = {target["binding_target_id"] for target in targets}

    assert set(invocation_schemas) == target_ids
    for invocation_schema in invocation_schemas.values():
        assert "source_candidate_id" not in invocation_schema["properties"]
    candidate_target_schema = invocation_schemas[
        "target.fact_1.set_difference.source_1.candidate_set"
    ]
    fulfillment_schema = candidate_target_schema["properties"]["fulfillment_decisions"]
    assert "answer_1" in fulfillment_schema["properties"]
    assert "anyOf" not in fulfillment_schema


def test_source_binding_projects_fact_shape_role_plan_families():
    prompt = SourceBindingTurnPrompt(_set_difference_request())

    payload = prompt.binding_plan_families_payload()

    fact = payload["bindings_by_requested_fact"]["fact_1"]
    shape = fact["plan_shapes"]["set_difference"]
    assert shape["member_constraint"] == "DISTINCT_SOURCE_CANDIDATES"
    assert set(shape["role_targets"]) == {"candidate_set", "observed_set"}
    assert all(
        target["requirement_id"] == role_id
        for role_id, targets in shape["role_targets"].items()
        for target in targets
    )


def test_source_binding_schema_is_fact_shape_role_keyed():
    schema = (
        SourceBindingTurnPrompt(_set_difference_request())
        .response_contract()
        .provider_schema
    )

    outcome = next(
        variant
        for variant in schema["properties"]["outcome"]["oneOf"]
        if variant["properties"]["kind"].get("enum") == ["source_bindings"]
    )
    fact = outcome["properties"]["bindings_for_fact_1"]

    assert "source_invocations" not in outcome["properties"]
    assert fact["properties"]["plan_shape"]["enum"] == ["set_difference"]
    assert set(fact["properties"]) - {"plan_shape"} == {
        "candidate_set",
        "observed_set",
    }


def test_source_binding_schema_scopes_param_surfaces_to_binding_target():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {
                "start_date": ("param_decision.source_1.start_date.bind.month",),
            },
            "target.source_2": {
                "status": ("param_decision.source_2.status.use_default.completed",),
                "group_by": ("param_decision.source_2.group_by.bind.location",),
            },
        },
        target_required_param_decision_ids={
            "target.source_1": ("start_date",),
            "target.source_2": ("group_by",),
        },
        target_finite_choice_values={
            "target.source_1": {"status": ("DRAFT", "COMPLETED")},
            "target.source_2": {"sale_type": ("STORE", "ONLINE")},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_test_ids={
            "target.source_1": {"status": ("subject_identity",)},
            "target.source_2": {"sale_type": ("subject_identity",)},
        },
        target_finite_choice_normal_instance_test_ids={
            "target.source_1": {"status": ()},
            "target.source_2": {"sale_type": ()},
        },
        target_row_predicate_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_population_roles={
            "target.source_1": ({"role_id": "role.source_1.rows"},),
            "target.source_2": ({"role_id": "role.source_2.rows"},),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={"fact_1": ("source_2.data.amount",)},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {"answer_1": ("support.source_1.answer_1",)},
            "target.source_2": {"answer_1": ("support.source_2.answer_1",)},
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": ("answer_1",),
            "target.source_2": ("answer_1",),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
            "target.source_2": ("pop.source_2.candidate_population",),
        },
        plan_families=_test_plan_families("target.source_1", "target.source_2"),
    )

    variants = _source_invocation_variants_by_target(schema)
    source_1 = variants["target.source_1"]
    source_2 = variants["target.source_2"]

    assert set(source_1["properties"]["param_decisions"]["properties"]) == {
        "start_date"
    }
    assert set(source_1["properties"]["finite_choice_param_reviews"]["properties"]) == {
        "status"
    }
    assert set(source_2["properties"]["param_decisions"]["properties"]) == {
        "group_by",
        "status",
    }
    assert source_1["properties"]["param_decisions"]["required"] == ["start_date"]
    assert source_2["properties"]["param_decisions"]["required"] == ["group_by"]
    assert set(source_2["properties"]["finite_choice_param_reviews"]["properties"]) == {
        "sale_type"
    }


def test_source_binding_schema_requires_only_selectable_fulfillment_outputs():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={"target.source_1": {}},
        target_required_param_decision_ids={"target.source_1": ()},
        target_finite_choice_values={"target.source_1": {}},
        target_row_predicate_values={"target.source_1": {}},
        target_finite_choice_test_ids={"target.source_1": {}},
        target_finite_choice_normal_instance_test_ids={"target.source_1": {}},
        target_row_predicate_test_ids={"target.source_1": {}},
        target_population_roles={"target.source_1": ()},
        target_requested_fact_ids={"target.source_1": "fact_1"},
        metric_evidence_ids_by_requested_fact={
            "fact_1": ("source_1.data.total",),
        },
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {
                "answer_group": ("support.source_1.answer_group",),
            },
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": ("answer_group", "answer_metric"),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
        },
        plan_families=_test_plan_families("target.source_1"),
    )
    invocation_schema = _source_invocation_variants_by_target(schema)["target.source_1"]
    fulfillment_schema = invocation_schema["properties"]["fulfillment_decisions"]

    assert fulfillment_schema["required"] == ["answer_group"]
    assert set(fulfillment_schema["properties"]) == {"answer_group"}
    validate(
        instance={
            "outcome": {
                "kind": "source_bindings",
                "metric_fit_bases": {
                    "fact_1": {
                        "source_1.data.total": {
                            "metric_meaning": "sales total",
                            "fit_basis": "The selected metric answers the fact.",
                        }
                    }
                },
                "fit_basis_interpretations": {
                    "fact_1": {
                        "source_1.data.total": {
                            "interpretation": "FITS_REQUESTED_ANSWER",
                        }
                    }
                },
                "bindings_for_fact_1": {
                    "plan_shape": "test_shape",
                    "primary": {
                        **_minimal_source_invocation(
                            "target.source_1",
                            "pop.source_1.candidate_population",
                        ),
                        "fulfillment_decisions": {
                            "answer_group": {
                                "fulfillment_choice_id": (
                                    "support.source_1.answer_group"
                                ),
                                "match_basis_explanation": "The selected group key.",
                            },
                        },
                    },
                },
            },
        },
        schema=schema,
    )


def test_source_binding_schema_requires_exposed_row_predicate_reviews():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={"target.source_1": {}},
        target_required_param_decision_ids={"target.source_1": ()},
        target_finite_choice_values={"target.source_1": {}},
        target_row_predicate_values={
            "target.source_1": {
                "rp.source_1.row.data.is_deposited": ("true", "false"),
            },
        },
        target_finite_choice_test_ids={"target.source_1": {}},
        target_finite_choice_normal_instance_test_ids={"target.source_1": {}},
        target_row_predicate_test_ids={
            "target.source_1": {
                "rp.source_1.row.data.is_deposited": ("membership_test_1",),
            },
        },
        target_population_roles={"target.source_1": ()},
        target_requested_fact_ids={"target.source_1": "fact_1"},
        metric_evidence_ids_by_requested_fact={},
        target_fulfillment_support_set_ids_by_answer_output={"target.source_1": {}},
        target_required_fulfillment_answer_output_ids={"target.source_1": ()},
        target_population_binding_ids={
            "target.source_1": ("pop.source_1.candidate_population",),
        },
        plan_families=_test_plan_families("target.source_1"),
    )
    invocation_schema = _source_invocation_variants_by_target(schema)["target.source_1"]
    row_reviews_schema = invocation_schema["properties"]["row_predicate_reviews"]

    assert row_reviews_schema["required"] == ["rp.source_1.row.data.is_deposited"]
    with pytest.raises(ValidationError):
        validate(
            instance={
                "outcome": {
                    "kind": "source_bindings",
                    "metric_fit_bases": {},
                    "fit_basis_interpretations": {},
                    **_test_fact_binding(
                        requested_fact_id="fact_1",
                        plan_shape="test_shape",
                        requirement_id="primary",
                        invocation=_minimal_source_invocation(
                            "target.source_1",
                            "pop.source_1.candidate_population",
                        ),
                    ),
                },
            },
            schema=schema,
        )


def test_source_binding_schema_accepts_known_target_arrays_without_enumeration():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_required_param_decision_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_finite_choice_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_normal_instance_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_population_roles={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1",),
            "target.source_2": ("pop.source_2",),
        },
        plan_families=_test_plan_families("target.source_1", "target.source_2"),
    )
    source_1 = _minimal_source_invocation("target.source_1", "pop.source_1")
    source_2 = _minimal_source_invocation("target.source_2", "pop.source_2")

    validate(
        instance=_source_binding_plan_payload(source_1),
        schema=schema,
    )
    validate(instance=_source_binding_plan_payload(source_2), schema=schema)


def test_source_binding_schema_exposes_alternatives_as_one_role_choice():
    schema = build_source_binding_schema(
        target_param_decision_ids_by_param={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_required_param_decision_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_finite_choice_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_values={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_finite_choice_normal_instance_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_row_predicate_test_ids={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_population_roles={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_requested_fact_ids={
            "target.source_1": "fact_1",
            "target.source_2": "fact_1",
        },
        metric_evidence_ids_by_requested_fact={},
        target_fulfillment_support_set_ids_by_answer_output={
            "target.source_1": {},
            "target.source_2": {},
        },
        target_required_fulfillment_answer_output_ids={
            "target.source_1": (),
            "target.source_2": (),
        },
        target_population_binding_ids={
            "target.source_1": ("pop.source_1",),
            "target.source_2": ("pop.source_2",),
        },
        plan_families=_test_plan_families("target.source_1", "target.source_2"),
    )
    source_1 = _minimal_source_invocation("target.source_1", "pop.source_1")
    source_2 = _minimal_source_invocation("target.source_2", "pop.source_2")

    validate(instance=_source_binding_plan_payload(source_1), schema=schema)
    validate(instance=_source_binding_plan_payload(source_2), schema=schema)


def test_source_binding_prompt_requires_each_shape_role_once():
    schema = (
        SourceBindingTurnPrompt(_set_difference_request())
        .response_contract()
        .provider_schema
    )
    outcome = _source_binding_outcome_schema(schema)
    fact = outcome["properties"]["bindings_for_fact_1"]

    assert fact["required"] == ["plan_shape", "candidate_set", "observed_set"]


def test_source_binding_prompt_does_not_forward_plan_selection_basis():
    request = _closed_key_grouped_staff_sales_request()
    note = "Uses row-level sales records; grouping and counting still happen later."
    plan = request.plan_selection.plan_selections[0]
    request = replace(
        request,
        plan_selection=PlanSelectionSet(plan_selections=(replace(plan, basis=note),)),
    )
    prompt = SourceBindingTurnPrompt(request)
    xml = source_binding_candidates_xml(prompt.source_invocation_candidate_payload())

    assert note not in xml
    assert "<selection_note>" not in xml


def test_source_binding_requested_fact_has_one_authoritative_description():
    prompt = SourceBindingTurnPrompt(_closed_key_grouped_staff_sales_request())
    fact = prompt.requested_facts_payload()["requested_facts"][0]

    assert "description" not in fact
    assert fact["answer_request"]["answer_fact"]
