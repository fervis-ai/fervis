from fervis.lookup.turn_prompts.projections import source_alignment_reviews_xml
from fervis.lookup.relation_catalog import (
    CatalogField,
    EndpointRead,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerOutput,
)
from fervis.lookup.source_binding.candidates.plan_selection_filter import (
    filter_prompt_payload_by_plan_selection,
)
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.operation_families.plan_selection_registry import (
    plan_selection_shape_specs_for_family,
)
from fervis.lookup.plan_selection import (
    SelectedSourceStrategy,
    PlanSelectionSet,
    SourceStrategyMember,
    PlanSelectionRequest as _PlanSelectionRequest,
    PlanSelectionTurnPrompt,
    parse_plan_selection,
)
from fervis.lookup.plan_selection.source_strategies import source_strategies_by_fact
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFactBasis,
    PlanImpossible,
)


def PlanSelectionRequest(**kwargs) -> _PlanSelectionRequest:
    relation_catalog = kwargs.get("relation_catalog")
    source_candidate_payload = kwargs.get("source_candidate_payload")
    if (
        isinstance(relation_catalog, RelationCatalog)
        and not relation_catalog.reads
        and isinstance(source_candidate_payload, dict)
    ):
        kwargs["relation_catalog"] = _relation_catalog_for_source_candidate_payload(
            source_candidate_payload
        )
    return _PlanSelectionRequest(**kwargs)


def test_source_alignment_prompt_groups_sources_inside_requested_fact():
    request = _two_source_alignment_request()

    payload = PlanSelectionTurnPrompt(request).source_alignment_candidates_payload()
    xml = source_alignment_reviews_xml(payload)

    assert "requested_facts" not in payload
    fact_payload = payload["requested_fact_source_candidates"][0]
    assert fact_payload["requested_fact_id"] == "fact_1"
    assert fact_payload["fact_text"] == (
        "staff member with the highest compensation this month"
    )
    assert [item["source_candidate_id"] for item in fact_payload["source_candidates"]] == [
        "source_1",
        "source_2",
    ]
    assert "plan_shape" not in xml
    assert "<source_candidate id=\"source_1\"" in xml
    assert "<source_candidate id=\"source_2\"" in xml
    assert xml.index("<requested_fact id=\"fact_1\">") < xml.index(
        "<source_candidate id=\"source_1\""
    )


def test_source_alignment_reviews_forward_aligned_contributors_without_narrowing_support():
    request = _two_source_alignment_request()
    source_payload = request.source_candidate_payload["requested_fact_sources"][0][
        "source_contexts"
    ][0]["source_options"][0]
    original_support_set_ids = _support_set_ids(
        source_payload["binding_surface"]["fulfillment_support_sets"]
    )

    plan_selection = parse_plan_selection(
        {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": {
                    "fact_1": {
                        "source_1": {
                            "source_candidate_id": "source_1",
                            "basis": "Rows contain staff identity and compensation values for the requested fact.",
                            "source_alignment": "DIRECT",
                        },
                        "source_2": {
                            "source_candidate_id": "source_2",
                            "basis": "Summary rows are payroll-adjacent but do not directly represent earned compensation.",
                            "source_alignment": "PARTIAL",
                        },
                    }
                },
            }
        },
        request=request,
    ).outcome

    filtered = filter_prompt_payload_by_plan_selection(
        request.source_candidate_payload,
        SourceBindingRequest(
            question=request.question,
            question_contract=request.question_contract,
            requested_facts=request.requested_facts,
            relation_catalog=request.relation_catalog,
            catalog_selection=None,
            plan_selection=plan_selection,
        ),
    )

    filtered_sources = filtered["requested_fact_sources"][0]["source_contexts"][0][
        "source_options"
    ]
    assert [item["source_candidate_id"] for item in filtered_sources] == [
        "source_1",
        "source_2",
    ]
    assert _support_set_ids(
        filtered_sources[0]["binding_surface"]["fulfillment_support_sets"]
    ) == original_support_set_ids


def test_source_alignment_reviews_exclude_not_aligned_sources():
    request = _two_source_alignment_request()

    plan_selection = parse_plan_selection(
        {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": {
                    "fact_1": {
                        "source_1": {
                            "source_candidate_id": "source_1",
                            "basis": "Rows contain staff identity and compensation values for the requested fact.",
                            "source_alignment": "DIRECT",
                        },
                        "source_2": {
                            "source_candidate_id": "source_2",
                            "basis": "Summary rows are payroll-adjacent but do not contain evidence for earned compensation.",
                            "source_alignment": "NOT_ALIGNED",
                        },
                    }
                },
            }
        },
        request=request,
    ).outcome

    filtered = filter_prompt_payload_by_plan_selection(
        request.source_candidate_payload,
        SourceBindingRequest(
            question=request.question,
            question_contract=request.question_contract,
            requested_facts=request.requested_facts,
            relation_catalog=request.relation_catalog,
            catalog_selection=None,
            plan_selection=plan_selection,
        ),
    )

    filtered_sources = filtered["requested_fact_sources"][0]["source_contexts"][0][
        "source_options"
    ]
    assert [item["source_candidate_id"] for item in filtered_sources] == ["source_1"]


def test_source_alignment_reviews_derive_impossible_when_no_source_is_aligned():
    request = _two_source_alignment_request()

    outcome = parse_plan_selection(
        {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": {
                    "fact_1": {
                        "source_1": {
                            "source_candidate_id": "source_1",
                            "basis": "Shift rows do not contain the requested evidence.",
                            "source_alignment": "NOT_ALIGNED",
                        },
                        "source_2": {
                            "source_candidate_id": "source_2",
                            "basis": "Payroll summary rows do not contain the requested evidence.",
                            "source_alignment": "NOT_ALIGNED",
                        },
                    }
                },
            }
        },
        request=request,
    ).outcome

    assert isinstance(outcome, PlanImpossible)
    blocked = outcome.blocked_facts[0]
    assert blocked.requested_fact_id == "fact_1"
    assert blocked.basis == BlockedFactBasis.CATALOG_ACCESS
    assert blocked.reviewed_read_ids == (
        "list_shift_compensation_list",
        "list_payroll_summary",
    )
    assert blocked.nearest_fields == ()


def test_plan_selection_schema_exposes_only_source_alignment_reviews():
    request = _two_source_alignment_request()
    schema_text = str(PlanSelectionTurnPrompt(request).response_contract().provider_schema)

    assert "source_alignment_reviews" in schema_text
    assert "blocked_facts" not in schema_text
    assert "nearest_fields" not in schema_text


def test_source_alignment_parser_derives_aligned_source_strategy():
    request = _plan_selection_request()
    source_strategy = _source_strategy_payload(request, plan_shape="ranked_aggregate")

    result = parse_plan_selection(
        {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": {
                    "fact_1": {
                        "source_3": {
                            "source_candidate_id": "source_3",
                            "basis": "The sales summary source has row population, amount, and location identity support.",
                            "source_alignment": "DIRECT",
                        }
                    }
                },
            }
        },
        request=request,
    )

    plan = result.outcome.plan_selection_for("fact_1")

    assert plan.plan_shape == "ranked_aggregate"
    assert plan.source_strategy_id == source_strategy["source_strategy_id"]
    assert plan.required_answer_output_ids == ("answer_1",)
    assert tuple(member.source_candidate_id for member in plan.source_members) == (
        "source_3",
    )
    assert plan.source_members[0].fulfillment_support_set_ids == ()
    assert plan.source_members[0].source_interface["answer_output_ids"] == ["answer_1"]
    assert set(
        _source_interface_response_row_field_ids(plan.source_members[0].source_interface)
    ) >= {"location_id", "amount"}


def test_plan_selection_prompt_projects_source_strategies():
    payload = PlanSelectionTurnPrompt(
        _plan_selection_request()
    ).plan_selection_candidates_payload()

    fact_payload = payload["requested_fact_source_strategies"][0]
    source_strategy = next(
        item
        for item in fact_payload["source_strategies"]
        if item["plan_shape"] == "ranked_aggregate"
    )
    source_member = source_strategy["source_members"][0]

    assert fact_payload["requested_fact_id"] == "fact_1"
    assert source_strategy["plan_shape"] == "ranked_aggregate"
    assert source_member["source_candidate_id"] == "source_3"
    assert "fulfillment_support_set_ids" not in source_member
    assert set(source_strategy["required_answer_output_ids"]) == {"answer_1"}
    assert set(_response_row_field_ids(source_member)) >= {"location_id", "amount"}


def test_grouped_ranked_candidates_project_exact_canonical_operation_bundles():
    fact = RequestedFact(
        id="fact_1",
        description="store with the highest sales this month",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="store with the highest sales",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="Which store has the highest sales this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_sales_summary",
                                    "fields": [
                                        {"field_id": "location_id"},
                                        {"field_id": "label"},
                                        {"field_id": "amount"},
                                        {"field_id": "count"},
                                    ],
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.source_1.answer_1.location_id",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.location_id",
                                            evidence_id="source_1.data.location_id",
                                            field_id="location_id",
                                            roles=("identity",),
                                        ),
                                        _group_key_support_set(
                                            "support.source_1.answer_1.label",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.label",
                                            evidence_id="source_1.data.label",
                                            field_id="label",
                                        ),
                                        _metric_support_set(
                                            "support.source_1.answer_1.amount",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.amount",
                                            evidence_id="source_1.data.amount",
                                            field_id="amount",
                                        ),
                                        _row_count_support_set(
                                            "support.source_1.answer_1.count",
                                            slot_id="slot.source_1.answer_1.count",
                                            evidence_id="source_1.data.count",
                                            row_path_id="data",
                                            field_id="count",
                                        ),
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    source_strategies = payload["requested_fact_source_strategies"][0]["source_strategies"]

    source_members = [plan["source_members"][0] for plan in source_strategies]
    assert all("fulfillment_support_set_ids" not in member for member in source_members)
    assert {
        _operation_field_ids(member)
        for member in source_members
    } == {
        ("location_id", "amount"),
        ("location_id", "count"),
        ("label", "amount"),
        ("label", "count"),
    }
    assert [_response_row_field_ids(member) for member in source_members] == [
        ["location_id", "label", "amount", "count"],
        ["location_id", "label", "amount", "count"],
        ["location_id", "label", "amount", "count"],
        ["location_id", "label", "amount", "count"],
    ]


def test_aggregate_operation_plan_selection_surfaces_each_valid_metric_operation():
    fact = RequestedFact(
        id="fact_1",
        description="staff person with the highest sales total",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="staff person who made the most sales",
            ),
            RequestedFactAnswerOutput(
                id="answer_2",
                description="sales total for that staff person",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="Which staff person made the most sales?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_sale_list",
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.source_1.answer_1.staff_id",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.staff_id",
                                            evidence_id="source_1.data.staff_id",
                                            field_id="staff_id",
                                        ),
                                        _metric_support_set(
                                            "support.source_1.answer_2.count",
                                            answer_output_id="answer_2",
                                            slot_id="slot.source_1.answer_2.count",
                                            evidence_id="source_1.data.count",
                                            field_id="count",
                                        ),
                                        _metric_support_set(
                                            "support.source_1.answer_2.amount",
                                            answer_output_id="answer_2",
                                            slot_id="slot.source_1.answer_2.amount",
                                            evidence_id="source_1.data.amount",
                                            field_id="amount",
                                        ),
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    strategies = source_strategies_by_fact(
        request.source_candidate_payload,
        requested_facts=request.requested_facts,
        relation_catalog=request.relation_catalog,
        shape_specs_for_family=plan_selection_shape_specs_for_family,
    )["fact_1"]
    ranked_strategies = [
        item for item in strategies if item.plan_shape == "ranked_aggregate"
    ]

    assert {
        frozenset(item.source_members[0].fulfillment_support_set_ids)
        for item in ranked_strategies
    } == {
        frozenset(
            (
                "support.source_1.answer_1.staff_id",
                "support.source_1.answer_2.count",
            )
        ),
        frozenset(
            (
                "support.source_1.answer_1.staff_id",
                "support.source_1.answer_2.amount",
            )
        ),
    }


def test_plan_selection_projects_operation_evidence_without_operation_bundle_choices():
    request = _ranked_multi_metric_plan_selection_request()

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    fact_payload = payload["requested_fact_source_strategies"][0]
    source_strategies = fact_payload["source_strategies"]
    assert len(source_strategies) == 6
    assert {
        _operation_field_ids(strategy["source_members"][0])
        for strategy in source_strategies
    } == {
        ("location_id", "amount"),
        ("location_id", "calculated_pay"),
        ("location_id", "commission"),
        ("staff_id", "amount"),
        ("staff_id", "calculated_pay"),
        ("staff_id", "commission"),
    }
    source_members = [strategy["source_members"][0] for strategy in source_strategies]
    assert all(
        strategy["source_strategy_id"].startswith(
            "source_strategy.fact_1.ranked_aggregate."
        )
        for strategy in source_strategies
    )
    assert all(strategy["plan_shape"] == "ranked_aggregate" for strategy in source_strategies)
    assert all(member["source_candidate_id"] == "source_1" for member in source_members)
    assert all("fulfillment_support_sets" not in member for member in source_members)
    assert all({
        "staff_id",
        "amount",
        "calculated_pay",
        "commission",
    } <= set(_response_row_field_ids(member)) for member in source_members)


def test_source_alignment_filter_preserves_all_support_sets_for_aligned_source():
    request = _ranked_multi_metric_plan_selection_request()
    source_payload = request.source_candidate_payload["requested_fact_sources"][0][
        "source_contexts"
    ][0]["source_options"][0]
    original_support_set_ids = _support_set_ids(
        source_payload["binding_surface"]["fulfillment_support_sets"]
    )
    plan_selection = parse_plan_selection(
        {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": {
                    "fact_1": {
                        "source_1": {
                            "source_candidate_id": "source_1",
                            "basis": "Use the sales rows as the ranked aggregate source.",
                            "source_alignment": "DIRECT",
                        }
                    }
                },
            }
        },
        request=request,
    ).outcome

    filtered = filter_prompt_payload_by_plan_selection(
        request.source_candidate_payload,
        SourceBindingRequest(
            question=request.question,
            question_contract=request.question_contract,
            requested_facts=request.requested_facts,
            relation_catalog=request.relation_catalog,
            catalog_selection=None,
            plan_selection=plan_selection,
        ),
    )

    filtered_source = filtered["requested_fact_sources"][0]["source_contexts"][0][
        "source_options"
    ][0]
    assert _support_set_ids(
        filtered_source["binding_surface"]["fulfillment_support_sets"]
    ) == original_support_set_ids


def test_plan_selection_payload_preserves_relation_member_roles():
    fact = RequestedFact(
        id="fact_1",
        description="products not sold this month",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SET_DIFFERENCE,
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    request = PlanSelectionRequest(
        question="Which products were not sold this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_products",
                                    "kind": "new_api_read",
                                    "read_id": "list_products",
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.products.answer_1.name",
                                            answer_output_id="answer_1",
                                            slot_id="slot.products.name",
                                            evidence_id="source_products.data.name",
                                            field_id="name",
                                        )
                                    ],
                                    "fields": [
                                        {"field_id": "product_id"},
                                        {"field_id": "name"},
                                    ],
                                },
                                {
                                    "source_candidate_id": "source_sales",
                                    "kind": "new_api_read",
                                    "read_id": "list_sales",
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.sales.answer_1.product_id",
                                            answer_output_id="answer_1",
                                            slot_id="slot.sales.product_id",
                                            evidence_id=(
                                                "source_sales.data.product_id"
                                            ),
                                            field_id="product_id",
                                        )
                                    ],
                                    "fields": [{"field_id": "product_id"}],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    source_strategy = payload["requested_fact_source_strategies"][0]["source_strategies"][0]

    assert [
        (member["requirement_ids"], member["source_candidate_id"])
        for member in source_strategy["source_members"]
    ] == [
        (["candidate_set"], "source_products"),
        (["observed_set"], "source_sales"),
    ]


def test_plan_selection_candidates_are_limited_to_answer_expression_family():
    payload = PlanSelectionTurnPrompt(
        _plan_selection_request()
    ).plan_selection_candidates_payload()

    candidate_shapes = {
        item["plan_shape"]
        for item in payload["requested_fact_source_strategies"][0]["source_strategies"]
    }

    assert candidate_shapes == {"ranked_aggregate"}


def test_relation_source_strategies_reject_intrinsic_only_value_sources():
    fact = RequestedFact(
        id="fact_1",
        description="store rows",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.LIST_ROWS,
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    request = PlanSelectionRequest(
        question="Which stores are open?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [],
            "value_source_candidates": [
                {
                    "source_candidate_id": "value_1",
                    "kind": "value",
                    "value_id": "value_1",
                }
            ],
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    assert payload["requested_fact_source_strategies"][0]["source_strategies"] == []


def test_scalar_count_candidates_keep_one_row_population_grain_per_candidate():
    fact = RequestedFact(
        id="fact_1",
        description="count of shifts today",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="shift count",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="How many shifts do we have today?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_shift_record_list",
                                    "fulfillment_support_sets": [
                                        _row_count_support_set(
                                            "support.source_1.answer_1.row.data",
                                            slot_id="slot.source_1.answer_1.row.data",
                                            evidence_id=(
                                                "source_1.data.shift_record_id"
                                            ),
                                            row_path_id="data",
                                            field_id="shift_record_id",
                                        ),
                                        _row_count_support_set(
                                            (
                                                "support.source_1.answer_1."
                                                "row.data_deposits"
                                            ),
                                            slot_id=(
                                                "slot.source_1.answer_1."
                                                "row.data_deposits"
                                            ),
                                            evidence_id=(
                                                "source_1.data.deposits."
                                                "cash_deposit_id"
                                            ),
                                            row_path_id="data.deposits",
                                            field_id="cash_deposit_id",
                                        ),
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    row_count_summaries = [
        _response_row_field_ids(member)
        for source_strategy in payload["requested_fact_source_strategies"][0][
            "source_strategies"
        ]
        if source_strategy["plan_shape"] == "aggregate_scalar"
        for member in source_strategy["source_members"]
    ]

    assert row_count_summaries == [["shift_record_id", "cash_deposit_id"]]


def test_plan_selection_summary_explains_count_vs_measured_value_candidates():
    fact = RequestedFact(
        id="fact_1",
        description="cash deposited",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="Amount of cash deposited",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="How much cash was deposited this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_cash_deposit_list",
                                    "fulfillment_support_sets": [
                                        _row_count_support_set(
                                            "support.source_1.answer_1.row",
                                            slot_id="slot.source_1.answer_1.row",
                                            evidence_id=(
                                                "source_1.data.cash_deposit_id"
                                            ),
                                            row_path_id="data",
                                            field_id="cash_deposit_id",
                                        ),
                                        {
                                            "fulfillment_support_set_id": (
                                                "support.source_1.answer_1.metric"
                                            ),
                                            "answer_output_id": "answer_1",
                                            "fulfillment_slots": [
                                                {
                                                    "fulfillment_slot_id": (
                                                        "slot.source_1.answer_1."
                                                        "metric"
                                                    ),
                                                    "metric_measure_evidence": [
                                                        {
                                                            "evidence_id": (
                                                                "source_1.data."
                                                                "amount"
                                                            ),
                                                            "field_id": "amount",
                                                            "row_path_id": "data",
                                                            "type": "decimal",
                                                        },
                                                        {
                                                            "evidence_id": (
                                                                "source_1.data." "fees"
                                                            ),
                                                            "field_id": "fees",
                                                            "row_path_id": "data",
                                                            "type": "decimal",
                                                        },
                                                    ],
                                                }
                                            ],
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    members = [
        source_strategy["source_members"][0]
        for source_strategy in payload["requested_fact_source_strategies"][0][
            "source_strategies"
        ]
        if source_strategy["plan_shape"] == "aggregate_scalar"
    ]

    assert [_response_row_field_ids(member) for member in members] == [
        ["cash_deposit_id", "amount", "fees"]
    ]


def test_scalar_count_candidates_drop_non_executable_row_population_grains():
    fact = RequestedFact(
        id="fact_1",
        description="count of sales this month",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    request = PlanSelectionRequest(
        question="How many sales happened this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_sale_list",
                                    "fulfillment_support_sets": [
                                        _row_count_support_set(
                                            "support.source_1.answer_1.row.sale_id",
                                            slot_id="slot.source_1.answer_1.row",
                                            evidence_id="source_1.data.sale_id",
                                            row_path_id="data",
                                            field_id="sale_id",
                                        )
                                    ],
                                },
                                {
                                    "source_candidate_id": "source_2",
                                    "kind": "new_api_read",
                                    "read_id": "list_sales_summary",
                                    "fulfillment_support_sets": [
                                        _row_population_support_set(
                                            "support.source_2.answer_1.row.data",
                                            slot_id="slot.source_2.answer_1.row",
                                            row_path_id="data",
                                        )
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    aggregate_scalar_candidates = [
        source_strategy
        for source_strategy in payload["requested_fact_source_strategies"][0][
            "source_strategies"
        ]
        if source_strategy["plan_shape"] == "aggregate_scalar"
    ]

    assert [
        member["read_id"]
        for source_strategy in aggregate_scalar_candidates
        for member in source_strategy["source_members"]
    ] == ["list_sale_list"]
    source_member = aggregate_scalar_candidates[0]["source_members"][0]
    assert aggregate_scalar_candidates[0]["required_answer_output_ids"] == ["answer_1"]
    assert _response_row_field_ids(source_member) == ["sale_id"]


def test_scalar_count_candidates_keep_executable_row_population_grain():
    fact = RequestedFact(
        id="fact_1",
        description="count of summary detail rows",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    request = PlanSelectionRequest(
        question="How many detail rows exist?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_summary",
                                    "fulfillment_support_sets": [
                                        _row_population_support_set(
                                            "support.source_1.answer_1.row.details",
                                            slot_id="slot.source_1.answer_1.row",
                                            row_path_id="details",
                                            row_source_id="rs.list_summary.details",
                                        )
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    aggregate_scalar_candidates = [
        source_strategy
        for source_strategy in payload["requested_fact_source_strategies"][0][
            "source_strategies"
        ]
        if source_strategy["plan_shape"] == "aggregate_scalar"
    ]

    assert [
        member["read_id"]
        for source_strategy in aggregate_scalar_candidates
        for member in source_strategy["source_members"]
    ] == ["list_summary"]
    assert _response_row_field_ids(
        aggregate_scalar_candidates[0]["source_members"][0]
    ) == ["details"]


def test_list_rows_candidate_keeps_answer_value_support_with_row_context():
    fact = RequestedFact(
        id="fact_1",
        description="staff and sales amount",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.LIST_ROWS,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="staff name",
            ),
            RequestedFactAnswerOutput(
                id="answer_2",
                description="sales amount",
            ),
        ),
    )
    support_sets = [
        _row_count_support_set(
            "support.source_1.answer_1.row",
            slot_id="slot.source_1.answer_1.row",
            evidence_id="source_1.data.staff_id",
            row_path_id="data",
            field_id="staff_id",
        ),
        {
            "fulfillment_support_set_id": "support.source_1.answer_2.metric",
            "answer_output_id": "answer_2",
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot.source_1.answer_2.metric",
                    "metric_measure_evidence": [
                        {
                            "evidence_id": "source_1.data.amount",
                            "field_id": "amount",
                        }
                    ],
                }
            ],
        },
    ]
    request = PlanSelectionRequest(
        question="List staff and sales amount.",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_staff_sales",
                                    "fields": [
                                        {"field_id": "staff_id", "type": "uuid"},
                                        {"field_id": "staff_name", "type": "string"},
                                        {"field_id": "amount", "type": "decimal"},
                                    ],
                                    "fulfillment_support_sets": support_sets,
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    source_strategy = next(
        item
        for item in PlanSelectionTurnPrompt(
            request
        ).plan_selection_candidates_payload()["requested_fact_source_strategies"][0][
            "source_strategies"
        ]
        if item["plan_shape"] == "list_rows"
    )
    source_member = source_strategy["source_members"][0]

    assert source_strategy["required_answer_output_ids"] == [
        "answer_1",
        "answer_2",
    ]
    assert set(_response_row_field_ids(source_member)) >= {"staff_id", "amount"}
    assert "fulfillment_support_set_ids" not in source_member


def test_source_binding_preserves_support_sets_for_aligned_source_candidate():
    request = _plan_selection_request()
    plan_selection = parse_plan_selection(
        {
            "outcome": {
                "kind": "source_alignment_reviews",
                "reviews_by_requested_fact": {
                    "fact_1": {
                        "source_3": {
                            "source_candidate_id": "source_3",
                            "basis": "Select the complete ranked aggregate source candidate.",
                            "source_alignment": "DIRECT",
                        }
                    }
                },
            }
        },
        request=request,
    ).outcome

    filtered = filter_prompt_payload_by_plan_selection(
        request.source_candidate_payload,
        SourceBindingRequest(
            question=request.question,
            question_contract=request.question_contract,
            requested_facts=request.requested_facts,
            relation_catalog=request.relation_catalog,
            catalog_selection=None,
            plan_selection=plan_selection,
        ),
    )

    source_option = filtered["requested_fact_sources"][0]["source_contexts"][0][
        "source_options"
    ][0]
    support_set_ids = {
        support_set["fulfillment_support_set_id"]
        for support_set in source_option["binding_surface"]["fulfillment_support_sets"]
    }
    assert support_set_ids == {
        "support.source_3.fact_1.row",
        "support.source_3.answer_1.slot.metric",
        "support.source_3.answer_1.slot.group",
        "support.source_3.fact_1.scope",
    }


def test_source_binding_filters_top_level_value_candidate_support_sets():
    selected_support_set = {
        "fulfillment_support_set_id": "support.value_1.answer_1.metric",
        "answer_output_id": "answer_1",
        "fulfillment_slots": [
            {
                "fulfillment_slot_id": "slot.value_1.metric",
                "metric_measure_evidence": [
                    {
                        "evidence_id": "value_1.metric",
                        "field_id": "value",
                    }
                ],
            }
        ],
    }
    stale_support_set = {
        "fulfillment_support_set_id": "support.value_1.answer_2.metric",
        "answer_output_id": "answer_2",
        "fulfillment_slots": [
            {
                "fulfillment_slot_id": "slot.value_1.stale",
                "metric_measure_evidence": [
                    {
                        "evidence_id": "value_1.stale",
                        "field_id": "stale",
                    }
                ],
            }
        ],
    }
    payload = {
        "requested_fact_sources": [],
        "value_source_candidates": [
            {
                "source_candidate_id": "value_1",
                "kind": "value",
                "value_id": "value_1",
                "fulfillment_support_sets": [
                    selected_support_set,
                    stale_support_set,
                ],
            }
        ],
    }
    plan_selection = PlanSelectionSet(
        plan_selections=(
            SelectedSourceStrategy(
                plan_selection_id="plan.fact_1",
                requested_fact_id="fact_1",
                source_strategy_id="source_strategy.fact_1.direct_field_value.1",
                plan_shape="direct_field_value",
                required_answer_output_ids=("answer_1",),
                source_members=(
                    SourceStrategyMember(
                        source_candidate_id="value_1",
                        fulfillment_support_set_ids=(
                            "support.value_1.answer_1.metric",
                        ),
                        source_interface={"answer_output_ids": ["answer_1"]},
                    ),
                ),
                basis="Use the selected value source.",
            ),
        )
    )

    filtered = filter_prompt_payload_by_plan_selection(
        payload,
        SourceBindingRequest(
            question="What is the known value?",
            question_contract=_plan_selection_request().question_contract,
            requested_facts=_plan_selection_request().requested_facts,
            relation_catalog=RelationCatalog(reads=()),
            catalog_selection=None,
            plan_selection=plan_selection,
        ),
    )

    assert filtered["value_source_candidates"][0]["fulfillment_support_sets"] == [
        selected_support_set,
    ]


def test_plan_selection_omits_ranked_candidate_without_full_output_operation_bundle():
    payload = _source_candidate_payload()
    support_sets = payload["requested_fact_sources"][0]["source_contexts"][0][
        "source_options"
    ][0]["fulfillment_support_sets"]
    metric_support_set = next(
        support_set
        for support_set in support_sets
        if support_set["fulfillment_support_set_id"]
        == "support.source_3.answer_1.slot.metric"
    )
    metric_support_set["answer_output_id"] = "answer_unrelated"

    request = PlanSelectionRequest(
        question="Which store has the highest sales this month?",
        question_contract=_plan_selection_request().question_contract,
        requested_facts=_plan_selection_request().requested_facts,
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload=payload,
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    candidate_shapes = {
        item["plan_shape"]
        for item in payload["requested_fact_source_strategies"][0]["source_strategies"]
    }

    assert "ranked_aggregate" not in candidate_shapes


def test_grouped_ranked_candidates_reject_cross_row_grain_operation_bundle():
    fact = RequestedFact(
        id="fact_1",
        description="staff who earned the most compensation",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    request = PlanSelectionRequest(
        question="Which staff earned the most compensation this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_payroll_summary",
                                    "fields": [
                                        {"field_id": "staff_id", "type": "uuid"},
                                        {"field_id": "staff_name", "type": "string"},
                                        {"field_id": "total_paid", "type": "decimal"},
                                    ],
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.source_1.answer_1.staff_id",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.staff_id",
                                            evidence_id=("source_1.staffs.staff_id"),
                                            field_id="staff_id",
                                            row_path_id="staffs",
                                            roles=("identity",),
                                        ),
                                        _row_count_support_set(
                                            "support.source_1.answer_1.root_count",
                                            slot_id="slot.source_1.answer_1.root_count",
                                            evidence_id="row_population.root",
                                            row_path_id="root",
                                            field_id="root",
                                        ),
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    assert payload["requested_fact_source_strategies"][0]["source_strategies"] == []


def test_grouped_ranked_candidates_reject_collection_group_coordinates():
    fact = RequestedFact(
        id="fact_1",
        description="store with the highest sales this month",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
        answer_outputs=(RequestedFactAnswerOutput(id="answer_1"),),
    )
    request = PlanSelectionRequest(
        question="Which store has the highest sales this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_store_list",
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.source_1.answer_1.hours",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.hours",
                                            evidence_id="source_1.data.hours",
                                            field_id="hours",
                                            row_path_id="data",
                                            type="json",
                                        ),
                                        _metric_support_set(
                                            "support.source_1.answer_1.amount",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.amount",
                                            evidence_id="source_1.data.amount",
                                            field_id="amount",
                                        ),
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    assert payload["requested_fact_source_strategies"][0]["source_strategies"] == []


def test_grouped_operation_candidate_combines_multiple_identity_outputs_with_metric():
    fact = RequestedFact(
        id="fact_1",
        description="salespeople, products, shades, and total sales",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(id="answer_1", description="staff"),
            RequestedFactAnswerOutput(id="answer_2", description="product"),
            RequestedFactAnswerOutput(id="answer_3", description="shade"),
            RequestedFactAnswerOutput(id="answer_4", description="total sales"),
        ),
    )
    request = PlanSelectionRequest(
        question="List all salespeople, products, shades, and total sales per person.",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "sales_read",
                                    "fields": [
                                        {"field_id": "staff_name", "type": "string"},
                                        {
                                            "field_id": "snapshot_merch_name",
                                            "type": "string",
                                        },
                                        {
                                            "field_id": "snapshot_shade_name",
                                            "type": "string",
                                        },
                                        {"field_id": "amount", "type": "decimal"},
                                    ],
                                    "fulfillment_support_sets": [
                                        _group_key_support_set(
                                            "support.source_1.answer_1.staff",
                                            answer_output_id="answer_1",
                                            slot_id="slot.source_1.answer_1.staff",
                                            evidence_id="source_1.data.staff_name",
                                            field_id="staff_name",
                                        ),
                                        _group_key_support_set(
                                            "support.source_1.answer_2.product",
                                            answer_output_id="answer_2",
                                            slot_id="slot.source_1.answer_2.product",
                                            evidence_id=(
                                                "source_1.data.snapshot_merch_name"
                                            ),
                                            field_id="snapshot_merch_name",
                                        ),
                                        _group_key_support_set(
                                            "support.source_1.answer_3.shade",
                                            answer_output_id="answer_3",
                                            slot_id="slot.source_1.answer_3.shade",
                                            evidence_id=(
                                                "source_1.data.snapshot_shade_name"
                                            ),
                                            field_id="snapshot_shade_name",
                                        ),
                                        _metric_support_set(
                                            "support.source_1.answer_4.amount",
                                            answer_output_id="answer_4",
                                            slot_id="slot.source_1.answer_4.amount",
                                            evidence_id="source_1.data.amount",
                                            field_id="amount",
                                        ),
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()
    source_strategy = next(
        item
        for item in payload["requested_fact_source_strategies"][0]["source_strategies"]
        if item["plan_shape"] == "aggregate_by_group"
        and _response_row_field_ids(item["source_members"][0])
        == ["staff_name", "snapshot_merch_name", "snapshot_shade_name", "amount"]
    )
    source_member = source_strategy["source_members"][0]

    assert set(source_strategy["required_answer_output_ids"]) == {
        "answer_1",
        "answer_2",
        "answer_3",
        "answer_4",
    }
    assert _response_row_field_ids(source_member) == [
        "staff_name",
        "snapshot_merch_name",
        "snapshot_shade_name",
        "amount",
    ]


def test_plan_selection_prompt_projects_value_candidates_as_selectable_options():
    fact = RequestedFact(
        id="fact_1",
        description="difference between the two known totals",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="difference",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="What is the difference?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [],
            "value_source_candidates": [
                {
                    "source_candidate_id": "value_1",
                    "kind": "value",
                    "value_id": "value_1",
                    "type": "number",
                    "value": 42,
                    "applies_to_requested_facts": ["fact_1"],
                }
            ],
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    candidate_shapes = {
        item["plan_shape"]
        for item in payload["requested_fact_source_strategies"][0]["source_strategies"]
    }
    assert "computed_scalar" not in candidate_shapes


def test_plan_selection_projects_computed_scalar_from_two_distinct_values():
    fact = RequestedFact(
        id="fact_1",
        description="difference between the two known totals",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.COMPUTED_SCALAR,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="difference",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="What is the difference?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [],
            "value_source_candidates": [
                {
                    "source_candidate_id": "value_1",
                    "kind": "value",
                    "value_id": "value_1",
                    "type": "number",
                    "value": 42,
                    "applies_to_requested_facts": ["fact_1"],
                },
                {
                    "source_candidate_id": "value_2",
                    "kind": "value",
                    "value_id": "value_2",
                    "type": "number",
                    "value": 13,
                    "applies_to_requested_facts": ["fact_1"],
                },
            ],
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    source_strategy = next(
        item
        for item in payload["requested_fact_source_strategies"][0]["source_strategies"]
        if item["plan_shape"] == "computed_scalar"
    )
    assert {
        member["source_candidate_id"] for member in source_strategy["source_members"]
    } == {"value_1", "value_2"}


def test_plan_selection_keeps_value_source_role_when_value_has_fulfillment_support():
    fact = RequestedFact(
        id="fact_1",
        description="known total",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.SCALAR_AGGREGATE,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="known total",
            ),
        ),
    )
    request = PlanSelectionRequest(
        question="What is the difference?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [],
            "value_source_candidates": [
                {
                    "source_candidate_id": "value_1",
                    "kind": "value",
                    "value_id": "value_1",
                    "type": "number",
                    "value": 42,
                    "applies_to_requested_facts": ["fact_1"],
                    "fulfillment_support_sets": [
                        {
                            "fulfillment_support_set_id": "support.value_1.answer_1.metric",
                            "answer_output_id": "answer_1",
                            "fulfillment_slots": [
                                {
                                    "fulfillment_slot_id": "slot.value_1.metric",
                                    "metric_measure_evidence": [
                                        {
                                            "evidence_id": "value_1.metric",
                                            "field_id": "value",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        conversation_context={},
    )

    payload = PlanSelectionTurnPrompt(request).plan_selection_candidates_payload()

    source_strategy = payload["requested_fact_source_strategies"][0]["source_strategies"][0]
    assert source_strategy["source_members"][0]["source_candidate_id"] == "value_1"
    assert "row_path_ids" not in source_strategy["source_members"][0]


def _plan_selection_request() -> PlanSelectionRequest:
    fact = RequestedFact(
        id="fact_1",
        description="store with the highest sales this month",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="store with the highest sales",
            ),
        ),
    )
    return PlanSelectionRequest(
        question="Which store has the highest sales this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload=_source_candidate_payload(),
        conversation_context={},
    )


def _ranked_multi_metric_plan_selection_request() -> PlanSelectionRequest:
    fact = RequestedFact(
        id="fact_1",
        description="staff member with the highest compensation this month",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.RANKED_SELECTION,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                description="staff member with the highest compensation",
            ),
        ),
    )
    support_sets = [
        _group_key_support_set(
            "support.source_1.answer_1.location_id",
            answer_output_id="answer_1",
            slot_id="slot.source_1.answer_1.location_id",
            evidence_id="source_1.data.location.id",
            field_id="location_id",
            row_path_id="shift_compensation",
            type="uuid",
        ),
        _group_key_support_set(
            "support.source_1.answer_1.staff_id",
            answer_output_id="answer_1",
            slot_id="slot.source_1.answer_1.staff_id",
            evidence_id="source_1.data.staff.id",
            field_id="staff_id",
            row_path_id="shift_compensation",
            type="uuid",
        ),
        _metric_support_set(
            "support.source_1.answer_1.amount",
            answer_output_id="answer_1",
            slot_id="slot.source_1.answer_1.amount",
            evidence_id="source_1.data.amount",
            field_id="amount",
            row_path_id="shift_compensation",
            type="decimal",
        ),
        _metric_support_set(
            "support.source_1.answer_1.calculated_pay",
            answer_output_id="answer_1",
            slot_id="slot.source_1.answer_1.calculated_pay",
            evidence_id="source_1.data.calculated_pay",
            field_id="calculated_pay",
            row_path_id="shift_compensation",
            type="decimal",
        ),
        _metric_support_set(
            "support.source_1.answer_1.commission",
            answer_output_id="answer_1",
            slot_id="slot.source_1.answer_1.commission",
            evidence_id="source_1.data.commission",
            field_id="commission",
            row_path_id="shift_compensation",
            type="decimal",
        ),
    ]
    return PlanSelectionRequest(
        question="Which staff earned the most this month?",
        question_contract=QuestionContract(requested_facts=(fact,)),
        requested_facts=(fact,),
        relation_catalog=RelationCatalog(reads=()),
        source_candidate_payload={
            "requested_fact_sources": [
                {
                    "requested_fact_id": "fact_1",
                    "source_contexts": [
                        {
                            "context_id": "fact_1:sources",
                            "source_options": [
                                {
                                    "source_candidate_id": "source_1",
                                    "kind": "new_api_read",
                                    "read_id": "list_shift_compensation_list",
                                    "fields": [
                                        {"field_id": "location_id"},
                                        {"field_id": "staff_id"},
                                        {"field_id": "amount"},
                                        {"field_id": "calculated_pay"},
                                        {"field_id": "commission"},
                                    ],
                                    "fulfillment_support_sets": support_sets,
                                    "binding_surface": {
                                        "fields": [
                                            {"field_id": "location_id"},
                                            {"field_id": "staff_id"},
                                            {"field_id": "amount"},
                                            {"field_id": "calculated_pay"},
                                            {"field_id": "commission"},
                                        ],
                                        "fulfillment_support_sets": support_sets,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        conversation_context={},
    )


def _two_source_alignment_request() -> PlanSelectionRequest:
    request = _ranked_multi_metric_plan_selection_request()
    shift_compensation_source = request.source_candidate_payload[
        "requested_fact_sources"
    ][0]["source_contexts"][0]["source_options"][0]
    payroll_support_sets = [
        _group_key_support_set(
            "support.source_2.answer_1.staff_id",
            answer_output_id="answer_1",
            slot_id="slot.source_2.answer_1.staff_id",
            evidence_id="source_2.staffs.staff_id",
            field_id="staff_id",
            row_path_id="staffs",
            type="uuid",
        ),
        _metric_support_set(
            "support.source_2.answer_1.pending_compensation",
            answer_output_id="answer_1",
            slot_id="slot.source_2.answer_1.pending_compensation",
            evidence_id="source_2.staffs.pending_compensation",
            field_id="pending_compensation",
            row_path_id="staffs",
            type="decimal",
        ),
    ]
    source_candidate_payload = {
        "requested_fact_sources": [
            {
                "requested_fact_id": "fact_1",
                "source_contexts": [
                    {
                        "context_id": "fact_1:sources",
                        "source_options": [
                            shift_compensation_source,
                            {
                                "source_candidate_id": "source_2",
                                "kind": "new_api_read",
                                "read_id": "list_payroll_summary",
                                "fields": [
                                    {"field_id": "staff_id"},
                                    {"field_id": "staff_name"},
                                    {"field_id": "pending_compensation"},
                                ],
                                "fulfillment_support_sets": payroll_support_sets,
                                "binding_surface": {
                                    "fields": [
                                        {"field_id": "staff_id"},
                                        {"field_id": "staff_name"},
                                        {"field_id": "pending_compensation"},
                                    ],
                                    "fulfillment_support_sets": payroll_support_sets,
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    }
    return PlanSelectionRequest(
        question=request.question,
        question_contract=request.question_contract,
        requested_facts=request.requested_facts,
        relation_catalog=_relation_catalog_for_source_candidate_payload(
            source_candidate_payload
        ),
        source_candidate_payload=source_candidate_payload,
        conversation_context={},
    )


def _source_candidate_payload() -> dict[str, object]:
    fulfillment_support_sets = [
        {
            "fulfillment_support_set_id": ("support.source_3.fact_1.row"),
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot.row",
                    "row_count_basis_evidence": [
                        {
                            "type": "row_population",
                            "row_path_id": "data",
                            "field_id": "data",
                        }
                    ],
                }
            ],
        },
        {
            "fulfillment_support_set_id": ("support.source_3.answer_1.slot.metric"),
            "answer_output_id": "answer_1",
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot.source_3.answer_1.metric",
                    "metric_measure_evidence": [
                        {
                            "evidence_id": ("source_3.data.amount"),
                            "field_id": "amount",
                            "row_path_id": "data",
                        }
                    ],
                }
            ],
        },
        {
            "fulfillment_support_set_id": ("support.source_3.answer_1.slot.group"),
            "answer_output_id": "answer_1",
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot.source_3.answer_1.group",
                    "group_key_evidence": [
                        {
                            "evidence_id": ("source_3.data.location_id"),
                            "field_id": "location_id",
                            "row_path_id": "data",
                        }
                    ],
                }
            ],
        },
        {
            "fulfillment_support_set_id": ("support.source_3.fact_1.scope"),
            "fulfillment_slots": [
                {
                    "fulfillment_slot_id": "slot.scope",
                    "scope_evidence": [
                        {
                            "evidence_id": ("source_3.data.sold_at"),
                            "field_id": "sold_at",
                            "row_path_id": "data",
                        }
                    ],
                }
            ],
        },
    ]
    return {
        "requested_fact_sources": [
            {
                "requested_fact_id": "fact_1",
                "source_contexts": [
                    {
                        "context_id": "fact_1:sources",
                        "source_options": [
                            {
                                "source_candidate_id": "source_3",
                                "kind": "new_api_read",
                                "read_id": "list_sales_summary",
                                "fulfillment_support_sets": fulfillment_support_sets,
                                "binding_surface": {
                                    "fulfillment_support_sets": fulfillment_support_sets
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _row_count_support_set(
    support_set_id: str,
    *,
    slot_id: str,
    evidence_id: str,
    row_path_id: str,
    field_id: str,
) -> dict[str, object]:
    return {
        "fulfillment_support_set_id": support_set_id,
        "answer_output_id": "answer_1",
        "fulfillment_slots": [
            {
                "fulfillment_slot_id": slot_id,
                "row_count_basis_evidence": [
                    {
                        "evidence_id": evidence_id,
                        "row_path_id": row_path_id,
                        "field_id": field_id,
                    }
                ],
            }
        ],
    }


def _row_population_support_set(
    support_set_id: str,
    *,
    slot_id: str,
    row_path_id: str,
    row_source_id: str = "",
) -> dict[str, object]:
    evidence = {
        "evidence_id": f"row_population.{row_path_id}",
        "row_path_id": row_path_id,
        "field_id": row_path_id,
        "type": "row_population",
        "row_cardinality": "many",
    }
    if row_source_id:
        evidence["row_source_id"] = row_source_id
    return {
        "fulfillment_support_set_id": support_set_id,
        "answer_output_id": "answer_1",
        "fulfillment_slots": [
            {
                "fulfillment_slot_id": slot_id,
                "row_count_basis_evidence": [evidence],
            }
        ],
    }


def _metric_support_set(
    support_set_id: str,
    *,
    answer_output_id: str,
    slot_id: str,
    evidence_id: str,
    field_id: str,
    row_path_id: str = "data",
    type: str = "",
) -> dict[str, object]:
    evidence = {
        "evidence_id": evidence_id,
        "field_id": field_id,
    }
    if row_path_id:
        evidence["row_path_id"] = row_path_id
    if type:
        evidence["type"] = type
    return {
        "fulfillment_support_set_id": support_set_id,
        "answer_output_id": answer_output_id,
        "fulfillment_slots": [
            {
                "fulfillment_slot_id": slot_id,
                "metric_measure_evidence": [evidence],
            }
        ],
    }


def _group_key_support_set(
    support_set_id: str,
    *,
    answer_output_id: str,
    slot_id: str,
    evidence_id: str,
    field_id: str,
    row_path_id: str = "data",
    roles: tuple[str, ...] = (),
    type: str = "",
) -> dict[str, object]:
    evidence = {
        "evidence_id": evidence_id,
        "field_id": field_id,
    }
    if row_path_id:
        evidence["row_path_id"] = row_path_id
    if roles:
        evidence["roles"] = list(roles)
    if type:
        evidence["type"] = type
    return {
        "fulfillment_support_set_id": support_set_id,
        "answer_output_id": answer_output_id,
        "fulfillment_slots": [
            {
                "fulfillment_slot_id": slot_id,
                "group_key_evidence": [evidence],
            }
        ],
    }


def _source_strategy_payload(
    request: PlanSelectionRequest,
    *,
    plan_shape: str,
) -> dict[str, object]:
    return next(
        item
        for item in PlanSelectionTurnPrompt(
            request
        ).plan_selection_candidates_payload()["requested_fact_source_strategies"][0][
            "source_strategies"
        ]
        if item["plan_shape"] == plan_shape
    )


def _relation_catalog_for_source_candidate_payload(
    payload: dict[str, object],
) -> RelationCatalog:
    reads: dict[str, EndpointRead] = {}
    for candidate in _api_read_candidates(payload):
        read_id = str(candidate.get("read_id") or "")
        if not read_id or read_id in reads:
            continue
        row_paths = _candidate_row_paths(candidate)
        reads[read_id] = EndpointRead(
            id=read_id,
            endpoint_name=read_id,
            row_paths=row_paths,
            fields=_candidate_catalog_fields(candidate, row_paths=row_paths),
        )
    return RelationCatalog(reads=tuple(reads.values()))


def _api_read_candidates(
    payload: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        candidate
        for fact_sources in payload.get("requested_fact_sources") or ()
        if isinstance(fact_sources, dict)
        for context in fact_sources.get("source_contexts") or ()
        if isinstance(context, dict)
        for candidate in context.get("source_options") or ()
        if isinstance(candidate, dict)
        and candidate.get("kind") in {"new_api_read", "same_scope_api_read"}
    )


def _candidate_row_paths(
    candidate: dict[str, object],
) -> tuple[RowPath, ...]:
    ids: list[str] = []
    cardinality_by_id: dict[str, RowCardinality] = {}
    for grain in candidate.get("result_grains") or ():
        if not isinstance(grain, dict):
            continue
        row_path_id = str(grain.get("row_path_id") or "")
        if not row_path_id:
            continue
        if row_path_id not in ids:
            ids.append(row_path_id)
        cardinality_by_id[row_path_id] = (
            RowCardinality.MANY
            if str(grain.get("cardinality") or "") == RowCardinality.MANY.value
            else RowCardinality.ONE
        )
    for evidence in _candidate_support_evidence(candidate):
        row_path_id = str(evidence.get("row_path_id") or "")
        if row_path_id and row_path_id not in ids:
            ids.append(row_path_id)
            cardinality_by_id[row_path_id] = RowCardinality.MANY
    if not ids:
        ids.append("root")
        cardinality_by_id["root"] = RowCardinality.ONE
    return tuple(
        RowPath(
            id=row_path_id,
            path=row_path_id,
            cardinality=cardinality_by_id.get(row_path_id, RowCardinality.ONE),
        )
        for row_path_id in ids
    )


def _candidate_catalog_fields(
    candidate: dict[str, object],
    *,
    row_paths: tuple[RowPath, ...],
) -> tuple[CatalogField, ...]:
    fields: list[CatalogField] = []
    seen: set[str] = set()
    read_id = str(candidate.get("read_id") or "")
    default_row_path_id = row_paths[0].id if row_paths else "root"
    for evidence in _candidate_support_evidence(candidate):
        field_id = str(evidence.get("field_id") or "")
        if not field_id:
            continue
        path = _evidence_path(
            candidate,
            evidence,
            default_row_path_id=default_row_path_id,
        )
        if path in seen:
            continue
        seen.add(path)
        fields.append(
            CatalogField(
                ref=f"{read_id}.{path}",
                path=path,
                type=str(evidence.get("type") or "string"),
                row_path_id=str(evidence.get("row_path_id") or default_row_path_id),
            )
        )
    for field in candidate.get("fields") or ():
        if not isinstance(field, dict):
            continue
        field_id = str(field.get("field_id") or field.get("id") or "")
        if not field_id:
            continue
        path = f"{default_row_path_id}.{field_id}"
        if path in seen:
            continue
        seen.add(path)
        fields.append(
            CatalogField(
                ref=f"{read_id}.{path}",
                path=path,
                type=str(field.get("type") or "string"),
                row_path_id=default_row_path_id,
            )
        )
    return tuple(fields)


def _candidate_support_evidence(
    candidate: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        evidence
        for support_set in candidate.get("fulfillment_support_sets") or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for key in (
            "metric_measure_evidence",
            "scope_evidence",
            "group_key_evidence",
            "row_count_basis_evidence",
        )
        for evidence in slot.get(key) or ()
        if isinstance(evidence, dict) and evidence.get("field_id")
    )


def _evidence_path(
    candidate: dict[str, object],
    evidence: dict[str, object],
    *,
    default_row_path_id: str,
) -> str:
    explicit_path = str(
        evidence.get("field_path") or evidence.get("response_path") or evidence.get("path") or ""
    )
    if explicit_path:
        return explicit_path
    row_path_id = str(evidence.get("row_path_id") or default_row_path_id)
    return f"{row_path_id}.{evidence['field_id']}"


def _response_row_field_ids(source_member: dict[str, object]) -> list[str]:
    return [
        str(field.get("field_id") or "")
        for row in source_member.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    ]


def _operation_field_ids(source_member: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(item.get("field_id") or "")
        for item in source_member.get("operation_evidence", ())
        if isinstance(item, dict) and str(item.get("field_id") or "")
    )


def _source_interface_response_row_field_ids(
    source_interface: dict[str, object],
) -> list[str]:
    return [
        str(field.get("field_id") or "")
        for row in source_interface.get("response_rows") or ()
        if isinstance(row, dict)
        for field in row.get("fields") or ()
        if isinstance(field, dict)
    ]


def _support_set_ids(support_sets: object) -> list[str]:
    return [
        str(item.get("fulfillment_support_set_id") or "")
        for item in support_sets or []
        if isinstance(item, dict)
    ]
