from ._helpers import *  # noqa: F403

from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.fact_planning.grouped_aggregate_choices import (
    selected_grouped_aggregate_operation,
)
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactGroupKey,
    RequestedFactOrderingDirection,
    ResultSelectionKind,
)


def _ordered_grouped_payload(
    *,
    metric_id="metric_1",
    metric_field_id="metric_total",
    function_id="function_sum",
    function_value="sum",
):
    return {
        "answers": [
            {
                "requested_fact_id": "rf_answer",
                "pattern": "aggregate_by_group",
                "source_binding_id": "sb_1",
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
            }
        ]
    }


def _ordered_grouped_fact() -> RequestedFact:
    return RequestedFact(
        id="rf_answer",
        description="group with the highest total",
        answer_expression=RequestedFactAnswerExpression(
            family=RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            group_key=RequestedFactGroupKey(
                id="answer_1",
                description="result group",
                domain=GroupKeyDomainKind.SOURCE_RESULT_VALUES,
            ),
            ordering_basis="aggregate value",
            ordering_direction=RequestedFactOrderingDirection.DESCENDING,
            selection_kind=ResultSelectionKind.TAKE_ONE,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_2",
                role="ANSWER_VALUE",
                description="aggregate value",
            ),
        ),
    )


def test_aggregate_by_group_fulfillment_maps_answer_outputs_by_selected_parts():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "aggregate_by_group",
                    "source_binding_id": "sb_1",
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
        item.answer_output_id: item.result_output_id for item in plan.fulfillment
    } == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }


def test_grouped_aggregate_group_is_backend_owned_not_model_selected():
    selection = selected_grouped_aggregate_operation(
        pattern_answer(
            {
                "requested_fact_id": "rf_answer",
                "pattern": "aggregate_by_group",
                "source_binding_id": "sb_1",
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
        ),
        bound_sources={"sb_1": _two_output_aggregate_bound_source()},
    )

    assert selection.group_field_ids == ("location_id",)


def test_aggregate_by_group_fulfillment_maps_answer_outputs_by_evidence_not_order():
    source = _two_output_aggregate_bound_source()
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "aggregate_by_group",
                    "source_binding_id": "sb_1",
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
        bound_sources=(
            replace(source, fulfillments=tuple(reversed(source.fulfillments))),
        ),
    )

    assert {
        item.answer_output_id: item.result_output_id for item in plan.fulfillment
    } == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }


def test_grouped_aggregate_selection_roles_outputs_by_selected_evidence_kind():
    selection = selected_grouped_aggregate_operation(
        pattern_answer(
            {
                "requested_fact_id": "rf_answer",
                "pattern": "aggregate_by_group",
                "source_binding_id": "sb_1",
                "metric": {
                    "selection_basis": "Orders are counted as records.",
                    "id": "metric_1",
                    "kind": "count_records",
                },
                "function": {
                    "selection_basis": "Count the matching rows.",
                    "id": "function_count",
                    "value": "count",
                },
            }
        ),
        bound_sources={"sb_1": _ranked_count_by_store_bound_source()},
    )

    assert {item.answer_output_id: item.role for item in selection.answer_outputs} == {
        "answer_1": "GROUP_KEY",
        "answer_2": "ROW_COUNT",
    }


def test_grouped_aggregate_count_metric_keeps_answer_output_identity():
    selection = selected_grouped_aggregate_operation(
        pattern_answer(
            {
                "requested_fact_id": "rf_answer",
                "pattern": "aggregate_by_group",
                "source_binding_id": "sb_1",
                "metric": {
                    "selection_basis": "Orders are counted as records.",
                    "id": "metric_1",
                    "kind": "count_records",
                },
                "function": {
                    "selection_basis": "Count the matching rows.",
                    "id": "function_count",
                    "value": "count",
                },
            }
        ),
        bound_sources={"sb_1": _ranked_count_by_store_bound_source()},
    )

    assert selection.metric.answer_output_id == "answer_2"


def test_aggregate_by_group_count_plan_keeps_count_output_through_validation():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "aggregate_by_group",
                    "source_binding_id": "sb_1",
                    "metric": {
                        "selection_basis": "Orders are counted as records.",
                        "id": "metric_1",
                        "kind": "count_records",
                    },
                    "function": {
                        "selection_basis": "Count the matching rows.",
                        "id": "function_count",
                        "value": "count",
                    },
                }
            ]
        },
        bound_sources=(_ranked_count_by_store_bound_source(),),
    )

    assert {
        item.answer_output_id: item.result_output_id for item in plan.fulfillment
    } == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }


def test_grouped_aggregate_measured_metric_keeps_answer_output_identity():
    selection = selected_grouped_aggregate_operation(
        pattern_answer(
            {
                "requested_fact_id": "rf_answer",
                "pattern": "aggregate_by_group",
                "source_binding_id": "sb_1",
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
        ),
        bound_sources={"sb_1": _two_output_aggregate_bound_source()},
    )

    assert selection.metric.answer_output_id == "answer_2"


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
                            "identity_fields": ["location_id"],
                        },
                        "observed": {
                            "source_binding_id": "sb_2",
                            "identity_fields": ["location_id"],
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
                            "identity_fields": ["location_id"],
                        },
                        "observed": {
                            "source_binding_id": "sb_1",
                            "identity_fields": ["location_id"],
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


def test_ordered_grouped_aggregate_metric_is_rendered_as_answer_value():
    plan = compile_pattern_answer_plan(
        _ordered_grouped_payload(),
        bound_sources=(_two_output_aggregate_bound_source(),),
        requested_facts=(_ordered_grouped_fact(),),
    )

    fulfillment_by_output = {
        item.answer_output_id: item.result_output_id for item in plan.fulfillment
    }
    render_by_id = {
        item.id: item
        for item in plan.result_projection.relation_outputs  # type: ignore[union-attr]
    }

    assert fulfillment_by_output["answer_1"] == "answer_1"
    assert fulfillment_by_output["answer_2"] == "answer_2"
    assert render_by_id["answer_2"].field_id == "metric_total"
    assert render_by_id["answer_2"].role == "answer_value"


def test_grouped_aggregate_prompt_exposes_compact_linear_choice_surface():
    request = FactPlanRequest(
        question="Which store had the highest sales this month?",
        question_contract=QuestionContract(
            requested_facts=(
                _ordered_grouped_fact(),
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
                    plan_selection_id="rf_answer.aggregate_by_group.sb_1",
                    source_strategy_id="source_strategy.rf_answer.aggregate_by_group.1",
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_1", "answer_2"),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    assert "Grouped aggregate operation choices:" in prompt
    assert (
        '<group fields="location_id" key_id="location_key" entity_kind="location" source="source_binding" />'
        in prompt
    )
    assert "<group_candidates>" not in prompt
    assert "choose group" not in prompt
    assert (
        '<metric id="metric_1" kind="aggregate_field" field="metric_total" type="decimal" allowed_functions="sum min max avg" />'
        in prompt
    )
    assert (
        '<function id="function_sum" value="sum" meaning="total across matching rows" />'
        in prompt
    )
    assert prompt.count("<metric ") == 1


def test_grouped_aggregate_schema_does_not_request_model_group_selection():
    request = FactPlanRequest(
        question="Which store had the highest sales this month?",
        question_contract=QuestionContract(
            requested_facts=(
                _ordered_grouped_fact(),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(_two_output_aggregate_bound_source(),),
    )
    prompt = PatternFactPlanTurnPrompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="rf_answer",
                    plan_selection_id="rf_answer.aggregate_by_group.sb_1",
                    source_strategy_id="source_strategy.rf_answer.aggregate_by_group.1",
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_1", "answer_2"),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )
    schema_text = json.dumps(prompt.response_contract().provider_schema)

    assert '"group"' not in schema_text


def test_ordered_grouped_aggregate_keeps_canonical_group_key_for_render_contract():
    plan = compile_pattern_answer_plan(
        _ordered_grouped_payload(),
        bound_sources=(_ranked_group_key_with_display_bound_source(),),
        requested_facts=(_ordered_grouped_fact(),),
    )

    fulfillment_by_output = {
        item.answer_output_id: item.result_output_id for item in plan.fulfillment
    }
    render_by_id = {
        item.id: item
        for item in plan.result_projection.relation_outputs  # type: ignore[union-attr]
    }

    assert fulfillment_by_output == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }
    assert render_by_id["answer_1"].entity_key is not None
    assert render_by_id["answer_1"].entity_key.entity_kind == "location"
    assert render_by_id["answer_1"].entity_key.key_id == "location_key"
    assert tuple(
        component.field_id
        for component in render_by_id["answer_1"].entity_key.components
    ) == ("location_id",)
    assert render_by_id["answer_2"].field_id == "metric_total"
    assert "location_id" not in {
        item.field_id
        for item in plan.result_projection.relation_outputs  # type: ignore[union-attr]
    }


def test_ordered_grouped_aggregate_compiles_count_metric():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "aggregate_by_group",
                    "source_binding_id": "sb_1",
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
                }
            ]
        },
        bound_sources=(_ranked_count_by_store_bound_source(),),
        requested_facts=(_ordered_grouped_fact(),),
    )

    fulfillment_by_output = {
        item.answer_output_id: item.result_output_id for item in plan.fulfillment
    }
    render_by_id = {
        item.id: item
        for item in plan.result_projection.relation_outputs  # type: ignore[union-attr]
    }

    assert fulfillment_by_output == {
        "answer_1": "answer_1",
        "answer_2": "answer_2",
    }
    assert render_by_id["answer_1"].entity_key is not None
    assert render_by_id["answer_1"].entity_key.entity_kind == "store"
    assert render_by_id["answer_1"].entity_key.key_id == "store_key"
    assert tuple(
        component.field_id
        for component in render_by_id["answer_1"].entity_key.components
    ) == ("store_id",)
    assert render_by_id["answer_2"].field_id == "count"


def test_ordered_grouped_aggregate_excludes_null_group_keys_before_ordering():
    plan = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "requested_fact_id": "rf_answer",
                    "pattern": "aggregate_by_group",
                    "source_binding_id": "sb_1",
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
                }
            ]
        },
        bound_sources=(_ranked_count_by_store_bound_source(),),
        requested_facts=(_ordered_grouped_fact(),),
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
            operations=tuple(
                ExecutableOperation(
                    id=operation.id,
                    spec=operation.spec,
                    output_relation=operation.output_relation,
                )
                for operation in plan.operations
            ),
            scalar_inputs=(
                ScalarInput(
                    id="constant:selection.take-one@selection@1",
                    value=1,
                ),
            ),
        )
    )

    assert result.relation("answer_1_rows").rows == (
        {"store_id": "store_a", "count": 3},
    )
