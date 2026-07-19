from __future__ import annotations

from tests.lookup.source_binding._plan_member_targets_fixtures import (
    FulfillmentDecisionOutput,
    ProviderObject,
    SourceBindingPlan,
    SourceBindingTarget,
    SourceBindingTurnPrompt,
    SourceCandidate,
    _closed_key_grouped_staff_sales_request,
    _closed_key_grouped_staff_sales_today_request,
    _closed_key_model_output_with_single_staff_param,
    _grain_safe_fulfillment_supports,
    _only_binding_target,
    _only_metric_evidence_id,
    _param_proofs_by_invocation,
    _population_binding_id,
    _prompt_candidates_by_id,
    _source_invocation_variants_by_target,
    _test_fact_binding,
    build_turn_prompt_context,
    compile_pattern_answer_program,
    compiler_input_context,
    parse_evidence_item,
    parse_fulfillment_support_set,
    parse_pattern_answer,
    parse_source_binding,
    parse_source_fulfillments,
    pytest,
    replace,
    source_binding_candidates_xml,
    satisfying_source_population_test_results,
    source_fulfills_by_row_population_for_candidate,
    source_fulfills_fields_for_candidate,
    source_fulfills_keys_for_candidate,
)


def test_closed_key_grouped_identity_param_is_backend_owned_not_model_authored():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )

    assert "backend_owned_param_bindings" in target
    assert "staff_id" not in param_properties


def test_closed_key_grouped_identity_param_is_backend_owned_without_grounding_uses():
    request = replace(
        _closed_key_grouped_staff_sales_request(),
        available_value_uses=(),
    )
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )
    candidate_prompt = source_binding_candidates_xml(
        prompt.source_invocation_candidate_payload()
    )

    assert "backend_owned_param_bindings" in str(target)
    assert "staff_id" not in param_properties
    assert '<param param_id="staff_id"' not in candidate_prompt


def test_closed_key_grouped_identity_param_can_use_group_key_field_id():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )

    assert "backend_owned_param_bindings" in str(target)
    assert "staff_id" not in param_properties


def test_resolved_endpoint_params_use_input_applications_not_param_decisions():
    request = _closed_key_grouped_staff_sales_today_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]

    param_properties = invocation_schema["properties"]["param_decisions"].get(
        "properties",
        {},
    )
    backend_bindings = target["backend_owned_param_bindings"]
    direct_param_ids = {
        binding["param_id"] for binding in backend_bindings if "param_id" in binding
    }
    key_param_ids = {
        param_id
        for binding in backend_bindings
        for param_id in binding.get("params_by_component_id", {}).values()
    }
    backend_param_ids = direct_param_ids | key_param_ids
    application_target_ids = {
        target_id
        for target_ids in target["resolved_input_application"][
            "targets_by_kind"
        ].values()
        for target_id in target_ids
    }

    assert param_properties == {}
    assert application_target_ids == {"start_date", "end_date"}
    assert "staff_id" not in param_properties
    assert backend_param_ids == {"staff_id"}


def test_closed_key_grouped_identity_param_scopes_group_key_fulfillment_choices():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    invocation_schema = _source_invocation_variants_by_target(
        prompt.response_contract().provider_schema
    )[target["binding_target_id"]]
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    choice_ids = invocation_schema["properties"]["fulfillment_decisions"]["properties"][
        "answer_staff"
    ]["properties"]["fulfillment_choice_id"]["enum"]
    staff_id_choice = source_fulfills_keys_for_candidate(
        candidate,
        key_ids_by_answer_output={"answer_staff": "staff_key"},
    )["answer_staff"]["fulfillment_choice_id"]

    assert choice_ids == ["source_1.data.reference.sale_staff"]
    assert staff_id_choice == "source_1.data.reference.sale_staff"
    assert "source_1.data.staff_name" not in choice_ids


def test_source_binding_prompt_marks_canonical_group_entity_choice():
    request = _closed_key_grouped_staff_sales_request()
    prompt = SourceBindingTurnPrompt(request)
    prompt_text = prompt.to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
        )
    ).prompt_text

    candidate_prompt = source_binding_candidates_xml(
        prompt.source_invocation_candidate_payload()
    )

    assert (
        "Entity outputs use a declared source candidate key or entity reference"
        in prompt_text
    )
    assert "Context labels are not selectable computation evidence" in prompt_text
    assert '<choice id="source_1.data.reference.sale_staff"' in candidate_prompt
    assert '<choice id="fulfillment_' not in candidate_prompt


def test_parse_source_binding_rejects_closed_key_param_with_mismatched_group_key():
    request = _closed_key_grouped_staff_sales_request()
    row_population_evidence_id = _only_metric_evidence_id(request)
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    with pytest.raises(
        ValueError, match="source fulfillment references unknown choice"
    ):
        parse_source_binding(
            {
                "outcome": {
                    "kind": "source_bindings",
                    "metric_fit_bases": {
                        "fact_1": {
                            row_population_evidence_id: {
                                "metric_meaning": "count of sales rows",
                                "fit_basis": (
                                    "The requested sales count is row cardinality."
                                ),
                            }
                        }
                    },
                    "fit_basis_interpretations": {
                        "fact_1": {
                            row_population_evidence_id: {
                                "interpretation": "FITS_REQUESTED_ANSWER",
                            }
                        }
                    },
                    **_test_fact_binding(
                        requested_fact_id=target["requested_fact_id"],
                        plan_shape=target["plan_shape"],
                        requirement_id=target["requirement_id"],
                        invocation={
                            "binding_target_id": target["binding_target_id"],
                            "answer_population": {
                                "population_binding_id": _population_binding_id(
                                    candidate
                                ),
                                "intent_text": "sales by specified staff member",
                                "match_basis_explanation": (
                                    "Use the sales row population for the grouped count."
                                ),
                                "population_test_results": (
                                    satisfying_source_population_test_results(target)
                                ),
                            },
                            "fulfillment_decisions": {
                                "answer_staff": {
                                    "fulfillment_choice_id": "source_1.data.staff_name",
                                    "match_basis_explanation": (
                                        "Stale non-canonical display field choice."
                                    ),
                                },
                                **source_fulfills_by_row_population_for_candidate(
                                    candidate,
                                    answer_output_ids=("answer_count",),
                                    row_path_id="data",
                                ),
                            },
                            "param_decisions": {},
                            "resolved_input_applications": [],
                            "row_predicate_reviews": {},
                            "finite_choice_param_reviews": {},
                        },
                    ),
                },
            },
            request=request,
        )


def test_parse_source_binding_expands_backend_owned_closed_key_param_bindings():
    request = _closed_key_grouped_staff_sales_request()
    row_population_evidence_id = _only_metric_evidence_id(request)
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    result = parse_source_binding(
        {
            "outcome": {
                "kind": "source_bindings",
                "metric_fit_bases": {
                    "fact_1": {
                        row_population_evidence_id: {
                            "metric_meaning": "count of sales rows",
                            "fit_basis": "The requested sales count is row cardinality.",
                        }
                    }
                },
                "fit_basis_interpretations": {
                    "fact_1": {
                        row_population_evidence_id: {
                            "interpretation": "FITS_REQUESTED_ANSWER",
                        }
                    }
                },
                **_test_fact_binding(
                    requested_fact_id=target["requested_fact_id"],
                    plan_shape=target["plan_shape"],
                    requirement_id=target["requirement_id"],
                    invocation={
                        "binding_target_id": target["binding_target_id"],
                        "answer_population": {
                            "population_binding_id": _population_binding_id(candidate),
                            "intent_text": "sales by specified staff member",
                            "match_basis_explanation": (
                                "Use the sales row population for the grouped count."
                            ),
                            "population_test_results": (
                                satisfying_source_population_test_results(target)
                            ),
                        },
                        "fulfillment_decisions": {
                            **source_fulfills_fields_for_candidate(
                                candidate,
                                field_ids_by_answer_output={
                                    "answer_staff": ("staff_id",),
                                },
                            ),
                            **source_fulfills_by_row_population_for_candidate(
                                candidate,
                                answer_output_ids=("answer_count",),
                                row_path_id="data",
                            ),
                        },
                        "param_decisions": {},
                        "resolved_input_applications": [],
                        "row_predicate_reviews": {},
                        "finite_choice_param_reviews": {},
                    },
                ),
            },
        },
        request=request,
    )

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_source = result.outcome.bound_sources[0]
    assert tuple(
        binding.value
        for invocation in bound_source.source_invocations
        for binding in invocation.param_bindings
        if binding.param_id == "staff_id"
    ) == (
        "51515151-0000-0000-0002-000000000001",
        "51515151-0000-0000-0002-000000000002",
    )


def test_parse_source_fulfillment_derives_required_row_count_without_selectable_choice():
    evidence_items = tuple(
        parse_evidence_item(item)
        for item in (
            {
                "evidence_id": "source_1.data.staff_id",
                "field_id": "staff_id",
                "type": "string",
            },
            {
                "evidence_id": "source_1.data.key.staff_key",
                "type": "candidate_key",
                "key_id": "staff_key",
                "entity_kind": "staff",
                "components": [
                    {
                        "component_id": "staff_id",
                        "field_evidence_id": "source_1.data.staff_id",
                        "field_id": "staff_id",
                    }
                ],
                "row_source_id": "read.sales.data",
                "row_path_id": "data",
            },
            {
                "evidence_id": "row_population.data",
                "type": "row_population",
                "row_source_id": "read.sales.data",
                "row_path_id": "data",
                "row_cardinality": "many",
            },
        )
    )
    support_set = parse_fulfillment_support_set(
        {
            "answer_output_id": "answer_staff",
            "fulfillment_choice_id": "fulfillment_staff",
            "fulfillment_support_set_id": "support_staff",
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot_staff",
                    "entity_evidence": [evidence_items[1].payload()],
                }
            ],
        }
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        evidence_items=evidence_items,
        fulfillment_support_sets=(support_set,),
    )

    fulfillments = parse_source_fulfillments(
        {
            "answer_staff": FulfillmentDecisionOutput(
                fulfillment_choice_id="fulfillment_staff",
                match_basis_explanation="Staff id is the grouped result key.",
            )
        },
        requested_fact_id="fact_1",
        answer_output_ids={"answer_staff", "answer_count"},
        required_answer_output_ids={"answer_staff", "answer_count"},
        metric_answer_output_ids={"answer_count"},
        candidate=candidate,
        plan_shape="aggregate_by_group",
        metric_fit_reviews_by_requested_output={
            "fact_1": {
                "row_population.data": {
                    "interpretation": "FITS_REQUESTED_ANSWER",
                    "metric_meaning": "count of sales rows",
                    "fit_basis": "The requested count is row cardinality.",
                }
            }
        },
    )
    count_fulfillment = next(
        fulfillment
        for fulfillment in fulfillments
        if fulfillment.answer_output_id == "answer_count"
    )
    assert count_fulfillment.row_count_basis_evidence_ids == ("row_population.data",)


def test_ranked_rows_rejects_repeating_entity_reference_fulfillment():
    evidence_items = tuple(
        parse_evidence_item(item)
        for item in (
            {
                "evidence_id": "source_1.data.key.primary_key",
                "type": "candidate_key",
                "key_id": "primary_key",
                "entity_kind": "shift_compensation",
                "components": [
                    {
                        "component_id": "shift_compensation_id",
                        "field_evidence_id": "source_1.data.shift_compensation_id",
                        "field_id": "shift_compensation_id",
                    }
                ],
                "row_source_id": "read.shift_compensation.data",
                "row_path_id": "data",
            },
            {
                "evidence_id": "source_1.data.reference.staff",
                "type": "entity_reference",
                "reference_id": "staff",
                "target_entity_kind": "staff",
                "target_key_id": "primary_key",
                "components": [
                    {
                        "component_id": "staff_id",
                        "field_evidence_id": "source_1.data.staff_id",
                        "field_id": "staff_id",
                    }
                ],
                "row_source_id": "read.shift_compensation.data",
                "row_path_id": "data",
            },
        )
    )
    support_set = parse_fulfillment_support_set(
        {
            "answer_output_id": "answer_staff",
            "fulfillment_choice_id": "fulfillment_staff",
            "fulfillment_support_set_id": "support_staff",
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot_staff",
                    "entity_evidence": [evidence_items[1].payload()],
                }
            ],
        }
    )
    candidate = SourceCandidate(
        id="source_1",
        applies_to_requested_fact_ids=("fact_1",),
        kind="read",
        evidence_items=evidence_items,
        fulfillment_support_sets=(support_set,),
    )

    visible_supports = _grain_safe_fulfillment_supports(
        candidate,
        target=SourceBindingTarget(
            binding_target_id="target.fact_1.ranked_rows.source_1.primary",
            requested_fact_id="fact_1",
            plan_shape="ranked_rows",
            source_candidate_id="source_1",
            requirement_id="primary",
            answer_output_ids=("answer_staff",),
            required_answer_output_ids=("answer_staff",),
        ),
        fulfillment_supports={"answer_staff": ("fulfillment_staff",)},
    )

    assert visible_supports == {"answer_staff": ()}

    with pytest.raises(
        ValueError,
        match="ranked_rows entity fulfillment does not identify source row grain",
    ):
        parse_source_fulfillments(
            {
                "answer_staff": FulfillmentDecisionOutput(
                    fulfillment_choice_id="fulfillment_staff",
                    match_basis_explanation="Return the staff on the highest row.",
                )
            },
            requested_fact_id="fact_1",
            answer_output_ids={"answer_staff"},
            required_answer_output_ids={"answer_staff"},
            metric_answer_output_ids=set(),
            candidate=candidate,
            plan_shape="ranked_rows",
            metric_fit_reviews_by_requested_output={},
        )


def test_closed_key_source_binding_retains_input_proofs_through_grouped_count_plan():
    request = _closed_key_grouped_staff_sales_today_request()
    prompt = SourceBindingTurnPrompt(request)
    target = _only_binding_target(prompt)
    candidate = _prompt_candidates_by_id(prompt.source_invocation_candidate_payload())[
        target["source_candidate_id"]
    ]

    result = parse_source_binding(
        _closed_key_model_output_with_single_staff_param(
            binding_target=target,
            candidate=candidate,
            row_population_evidence_id=_only_metric_evidence_id(request),
        ),
        request=request,
    )

    assert isinstance(result.outcome, SourceBindingPlan)
    bound_source = result.outcome.bound_sources[0]
    assert _param_proofs_by_invocation(bound_source, "staff_id") == (
        (
            "51515151-0000-0000-0002-000000000001",
            ("known_input:staff_id_1",),
        ),
        (
            "51515151-0000-0000-0002-000000000002",
            ("known_input:staff_id_2",),
        ),
    )
    assert _param_proofs_by_invocation(bound_source, "start_date") == (
        ("2026-07-06", ("known_input:today",)),
        ("2026-07-06", ("known_input:today",)),
    )
    assert _param_proofs_by_invocation(bound_source, "end_date") == (
        ("2026-07-06", ("known_input:today",)),
        ("2026-07-06", ("known_input:today",)),
    )
    assert bound_source.applied_filters == ()

    answer_plan, answer_bindings = compile_pattern_answer_program(
        (
            parse_pattern_answer(
                ProviderObject(
                    {
                        "requested_fact_id": "fact_1",
                        "pattern": "aggregate_by_group",
                        "source_binding_id": bound_source.id,
                        "metric": {
                            "selection_basis": "Count matching sales rows.",
                            "id": "metric_1",
                            "kind": "count_records",
                        },
                        "function": {
                            "selection_basis": "A row count uses count.",
                            "id": "function_count",
                            "value": "count",
                        },
                    }
                )
            ),
        ),
        bound_sources=result.outcome.bound_sources,
        source_binding_ids_by_requested_fact_id={"fact_1": (bound_source.id,)},
        source_binding_ids_by_requirement_by_requested_fact_id={
            "fact_1": {"operation": (bound_source.id,)}
        },
        input_context=compiler_input_context(
            values=request.available_values,
            question_contract=request.question_contract,
        ),
    )

    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.instantiation import (
        ExecutionEnvironment,
        instantiate_answer_program,
    )

    program, bindings = compile_answer_program(
        answer_plan,
        question_contract=request.question_contract,
        catalog=request.relation_catalog,
        bindings=answer_bindings,
    )
    instantiate_answer_program(
        program,
        bindings,
        ExecutionEnvironment(
            catalog=request.relation_catalog,
            catalog_selection=request.catalog_selection,
        ),
    )
