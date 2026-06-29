from ._helpers import *  # noqa: F403

from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import RelationEngineInput
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessStatus,
    RelationRows,
)


def _ranked_payload(*, group_id="group_1", group_field_id="location_name", metric_id="metric_1", metric_field_id="metric_total", function_id="function_sum", function_value="sum"):
    return {
        "answers": [
            {
                "requested_fact_id": "rf_answer",
                "pattern": "ranked_aggregate",
                "source_binding_id": "sb_1",
                "group": {
                    "selection_basis": "The requested answer is grouped by location.",
                    "id": group_id,
                    "field_id": group_field_id,
                },
                "metric": {
                    "selection_basis": "The requested measure is metric_total.",
                    "id": metric_id,
                    "kind": "aggregate_field",
                    "field_id": metric_field_id,
                },
                "function": {
                    "selection_basis": "The requested measure is a total.",
                    "id": function_id,
                    "value": function_value,
                },
                "rank": {
                    "selection_basis": "The question asks for the highest group.",
                    "id": "rank_top_1_desc",
                    "sort": "desc",
                    "limit": 1,
                },
            }
        ]
    }


def test_aggregate_by_group_fulfillment_maps_answer_outputs_by_selected_parts():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "aggregate_by_group",
                    "source_binding_id": "sb_1",
                    "group": {
                        "selection_basis": "Group by the location answer.",
                        "id": "group_1",
                        "field_id": "location_name",
                    },
                    "metric": {
                        "selection_basis": "Metric total is the measured value.",
                        "id": "metric_1",
                        "kind": "aggregate_field",
                        "field_id": "metric_total",
                    },
                    "function": {
                        "selection_basis": "Total requires sum.",
                        "id": "function_sum",
                        "value": "sum",
                    },
                }
            ]
        },
        bound_sources=(_two_output_aggregate_bound_source(),),
    )

    assert {
        item.answer_output_id: item.render_output_id for item in plan.fulfillment
    } == {
        "answer_1": "location_name",
        "answer_2": "metric_total",
    }


def test_pattern_compiler_rejects_unknown_pattern():
    with pytest.raises(ValueError, match="unsupported fact plan pattern"):
        compile_pattern_answer_plan(
            {
                "answers": [
                    {
                        "requested_fact_id": "rf_answer",
                        "pattern": "not_a_pattern",
                        "answer_output_ids": ["answer_1"],
                    }
                ]
            },
            bound_sources=(),
        )


def test_pattern_compiler_rejects_multi_relation_source_outside_selected_plan():
    with pytest.raises(
        ValueError,
        match="fact plan references source outside selected plan shape",
    ):
        compile_pattern_answer_plan(
            {
                "answers": [
                    {
                        "requested_fact_id": "rf_answer",
                        "pattern": "set_difference",
                        "answer_output_ids": ["answer_1"],
                        "candidate": {
                            "source_binding_id": "sb_1",
                            "identity_fields": ["location_name"],
                            "output_fields": [{"field_id": "location_name"}],
                        },
                        "observed": {
                            "source_binding_id": "sb_2",
                            "identity_fields": ["location_name"],
                        },
                    }
                ]
            },
            bound_sources=(_two_output_aggregate_bound_source(),),
            source_binding_ids_by_requested_fact_id={"rf_answer": ("sb_1",)},
        )


def test_pattern_compiler_rejects_multi_relation_source_outside_selected_role():
    with pytest.raises(
        ValueError,
        match="fact plan references source outside selected operand role",
    ):
        compile_pattern_answer_plan(
            {
                "answers": [
                    {
                        "requested_fact_id": "rf_answer",
                        "pattern": "set_difference",
                        "answer_output_ids": ["answer_1"],
                        "candidate": {
                            "source_binding_id": "sb_2",
                            "identity_fields": ["location_name"],
                            "output_fields": [{"field_id": "location_name"}],
                        },
                        "observed": {
                            "source_binding_id": "sb_1",
                            "identity_fields": ["location_name"],
                        },
                    }
                ]
            },
            bound_sources=_two_output_aggregate_bound_source_pair(),
            source_binding_ids_by_requested_fact_id={"rf_answer": ("sb_1", "sb_2")},
            source_binding_ids_by_requirement_by_requested_fact_id={
                "rf_answer": {
                    "candidate_set": ("sb_1",),
                    "observed_set": ("sb_2",),
                }
            },
        )


def test_ranked_aggregate_metric_answer_output_is_rendered_as_answer_value():
    plan = compile_pattern_answer_plan(
        _ranked_payload(),
        bound_sources=(_two_output_aggregate_bound_source(),),
    )

    fulfillment_by_output = {
        item.answer_output_id: item.render_output_id for item in plan.fulfillment
    }
    render_by_id = {
        item.id: item for item in plan.render_spec.relation_outputs  # type: ignore[union-attr]
    }

    assert fulfillment_by_output["answer_1"] == "answer_1"
    assert fulfillment_by_output["answer_2"] == "answer_2"
    assert render_by_id["answer_2"].field_id == "metric_total"
    assert render_by_id["answer_2"].role == "answer_value"


def test_ranked_aggregate_prompt_exposes_compact_linear_choice_surface():
    request = FactPlanRequest(
        question="Which store had the highest sales this month?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="rf_answer",
                    description="store with the highest sales this month",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            description="store name",
                        ),
                        RequestedFactAnswerOutput(
                            id="answer_2",
                            description="sales amount",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(_two_output_aggregate_bound_source(),),
    )
    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="rf_answer",
                    plan_selection_id="rf_answer.ranked_aggregate.sb_1",
                    source_strategy_id="source_strategy.rf_answer.ranked_aggregate.1",
                    plan_shape="ranked_aggregate",
                    required_answer_output_ids=("answer_1", "answer_2"),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    assert "Grouped/ranked operation choices:" in prompt
    assert '<group id="group_1" field="location_name" type="string" />' in prompt
    assert '<metric id="metric_1" kind="aggregate_field" field="metric_total" type="decimal" allowed_functions="sum min max avg" />' in prompt
    assert '<function id="function_sum" value="sum" meaning="total across matching rows" />' in prompt
    assert prompt.count("<metric ") == 1


def test_ranked_aggregate_parser_validates_candidate_id_matches_echoed_field():
    with pytest.raises(ValueError, match="group selection mismatches candidate"):
        compile_pattern_answer_plan(
            _ranked_payload(group_id="group_1", group_field_id="metric_total"),
            bound_sources=(_two_output_aggregate_bound_source(),),
        )


def test_ranked_aggregate_choice_keeps_canonical_group_key_for_render_contract():
    plan = compile_pattern_answer_plan(
        _ranked_payload(group_field_id="location_id"),
        bound_sources=(_ranked_group_key_with_display_bound_source(),),
    )

    fulfillment_by_output = {
        item.answer_output_id: item.render_output_id for item in plan.fulfillment
    }
    render_by_id = {
        item.id: item for item in plan.render_spec.relation_outputs  # type: ignore[union-attr]
    }

    assert fulfillment_by_output == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }
    assert render_by_id["answer_1"].field_id == "location_id"
    assert render_by_id["answer_2"].field_id == "metric_total"
    assert "location_name" not in {
        item.field_id for item in plan.render_spec.relation_outputs  # type: ignore[union-attr]
    }


def test_ranked_aggregate_choice_compiles_count_metric():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "ranked_aggregate",
                    "source_binding_id": "sb_1",
                    "group": {
                        "selection_basis": "The requested answer is a store.",
                        "id": "group_1",
                        "field_id": "store_id",
                    },
                    "metric": {
                        "selection_basis": "Orders are counted as records.",
                        "id": "metric_1",
                        "kind": "count_records",
                    },
                    "function": {
                        "selection_basis": "Counting records uses count.",
                        "id": "function_count",
                        "value": "count",
                    },
                    "rank": {
                        "selection_basis": "The question asks for the most orders.",
                        "id": "rank_top_1_desc",
                        "sort": "desc",
                        "limit": 1,
                    },
                }
            ]
        },
        bound_sources=(_ranked_count_by_store_bound_source(),),
    )

    fulfillment_by_output = {
        item.answer_output_id: item.render_output_id for item in plan.fulfillment
    }
    render_by_id = {
        item.id: item for item in plan.render_spec.relation_outputs  # type: ignore[union-attr]
    }

    assert fulfillment_by_output == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }
    assert render_by_id["answer_1"].field_id == "store_id"
    assert render_by_id["answer_2"].field_id == "count"


def test_ranked_aggregate_excludes_null_answer_group_keys_before_ranking():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "ranked_aggregate",
                    "source_binding_id": "sb_1",
                    "group": {
                        "selection_basis": "The requested answer is a store.",
                        "id": "group_1",
                        "field_id": "store_id",
                    },
                    "metric": {
                        "selection_basis": "Orders are counted as records.",
                        "id": "metric_1",
                        "kind": "count_records",
                    },
                    "function": {
                        "selection_basis": "Counting records uses count.",
                        "id": "function_count",
                        "value": "count",
                    },
                    "rank": {
                        "selection_basis": "The question asks for the most orders.",
                        "id": "rank_top_1_desc",
                        "sort": "desc",
                        "limit": 1,
                    },
                }
            ]
        },
        bound_sources=(_ranked_count_by_store_bound_source(),),
    )

    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    id="answer_1_source",
                    rows=(
                        {"store_id": None, "order_id": "order_1"},
                        {"store_id": None, "order_id": "order_2"},
                        {"store_id": None, "order_id": "order_3"},
                        {"store_id": "store_a", "order_id": "order_4"},
                        {"store_id": "store_a", "order_id": "order_5"},
                        {"store_id": "store_a", "order_id": "order_6"},
                    ),
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=plan.operations,
        )
    )

    assert result.relation("answer_1_rows").rows == (
        {"store_id": "store_a", "count": 3},
    )


def test_ranked_aggregate_keeps_metric_render_artifact_role_scoped_for_single_output():
    plan = compile_pattern_answer_plan(
        _ranked_payload(group_field_id="location_id"),
        bound_sources=(_single_output_ranked_aggregate_bound_source(),),
    )

    relation_outputs = plan.render_spec.relation_outputs  # type: ignore[union-attr]
    render_output_ids = tuple(item.id for item in relation_outputs)
    fulfillment_by_output = {
        item.answer_output_id: item.render_output_id for item in plan.fulfillment
    }
    render_by_id = {item.id: item for item in relation_outputs}
    metric_outputs = tuple(
        item for item in relation_outputs if item.field_id == "metric_total"
    )

    assert len(render_output_ids) == len(set(render_output_ids))
    assert fulfillment_by_output["answer_1"] == "answer_1"
    assert render_by_id["answer_1"].field_id == "location_id"
    assert len(metric_outputs) == 1
    assert metric_outputs[0].id != "answer_1"
    assert metric_outputs[0].role == "ranking_metric"
