from fervis.lookup.plan_selection import (
    PlanSelectionSet,
    SelectedSourceStrategy,
    SourceStrategyMember,
)
from fervis.lookup.plan_selection.model import OperationEvidence
from fervis.lookup.question_contract import (
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    ResultSelectionKind,
    RequestedFactGroupKey,
    RequestedFactAnswerOutput,
)
from fervis.lookup.source_binding.model import (
    AnswerPopulation,
    BoundSource,
    SourceBindingPlan,
    SourceFulfillment,
)
from fervis.lookup.source_binding.role_selection import (
    bound_plan_selection_for_source_binding,
)


def test_role_binding_preserves_every_plan_supported_by_admitted_evidence():
    fact = RequestedFact(
        id="fact_1",
        description="top staff compensation",
        answer_expression=RequestedFactAnswerExpression(
            RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE,
            group_key=RequestedFactGroupKey(
                id="answer_1",
                description="staff",
                domain=GroupKeyDomainKind.SOURCE_RESULT_VALUES,
            ),
            selection_kind=ResultSelectionKind.ALL_RESULTS,
        ),
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                role="ANSWER_VALUE",
                description="staff",
            ),
        ),
    )
    plans = PlanSelectionSet(
        plan_selections=(
            _ranked_plan("plan_calculated_pay", metric_evidence_id="calculated_pay"),
            _ranked_plan("plan_amount_paid", metric_evidence_id="amount_paid"),
        )
    )
    source_binding = SourceBindingPlan(
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                binding_target_id=("target.fact_1.aggregate_by_group.source_1.operation"),
                requirement_id="operation",
                answer_population=AnswerPopulation(
                    population_binding_id="population_1",
                    intent_text="compensation rows",
                    match_basis_explanation="The source contains compensation rows.",
                ),
                value_id="value_1",
                source_candidate_id="source_1",
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="The source contains the group and metrics.",
                        value_evidence_ids=("staff",),
                        metric_measure_evidence_ids=(
                            "calculated_pay",
                            "amount_paid",
                        ),
                    ),
                ),
            ),
        )
    )

    bound = bound_plan_selection_for_source_binding(
        plans,
        source_binding,
        requested_facts=(fact,),
    )

    assert bound is not None
    assert tuple(plan.plan_selection_id for plan in bound.plan_selections) == (
        "plan_calculated_pay",
        "plan_amount_paid",
    )


def test_role_binding_accepts_one_selected_fulfillment_alternative():
    fact = RequestedFact(
        id="fact_1",
        description="staff identifier",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="answer_1",
                role="ANSWER_VALUE",
                description="staff identifier",
            ),
        ),
    )
    member = SourceStrategyMember(
        source_candidate_id="source_1",
        requirement_ids=("primary",),
        fulfillment_support_set_ids=("support_staff_id", "support_phone"),
        answer_output_ids=("answer_1",),
        operation_evidence=(
            OperationEvidence(
                kind="candidate_key",
                evidence_id="staff_id",
                field_id="staff_id",
            ),
            OperationEvidence(
                kind="candidate_key",
                evidence_id="phone",
                field_id="phone",
            ),
        ),
    )
    plans = PlanSelectionSet(
        plan_selections=(
            SelectedSourceStrategy(
                plan_selection_id="plan_staff_value",
                requested_fact_id="fact_1",
                source_strategy_id="strategy_staff_value",
                plan_shape="direct_field_value",
                required_answer_output_ids=("answer_1",),
                source_members=(member,),
                basis="The source exposes alternative staff keys.",
            ),
        )
    )
    source_binding = SourceBindingPlan(
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                binding_target_id=("target.fact_1.direct_field_value.source_1.primary"),
                requirement_id="primary",
                answer_population=AnswerPopulation(
                    population_binding_id="population_1",
                    intent_text="the selected staff row",
                    match_basis_explanation="The source contains staff rows.",
                ),
                value_id="value_1",
                source_candidate_id="source_1",
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="The selected key is the staff ID.",
                        fulfillment_support_set_id="support_staff_id",
                        value_evidence_ids=("staff_id",),
                    ),
                ),
            ),
        )
    )

    bound = bound_plan_selection_for_source_binding(
        plans,
        source_binding,
        requested_facts=(fact,),
    )

    assert bound is not None
    assert tuple(plan.plan_selection_id for plan in bound.plan_selections) == (
        "plan_staff_value",
    )


def _ranked_plan(
    plan_id: str,
    *,
    metric_evidence_id: str,
) -> SelectedSourceStrategy:
    operation_evidence = (
        OperationEvidence(
            kind="group_key",
            evidence_id="staff",
            field_id="staff_id",
        ),
        OperationEvidence(
            kind="metric_measure",
            evidence_id=metric_evidence_id,
            field_id=metric_evidence_id,
        ),
    )
    member = SourceStrategyMember(
        source_candidate_id="source_1",
        requirement_ids=("operation",),
        answer_output_ids=("answer_1",),
        operation_evidence=operation_evidence,
    )
    return SelectedSourceStrategy(
        plan_selection_id=plan_id,
        requested_fact_id="fact_1",
        source_strategy_id=f"strategy_{plan_id}",
        plan_shape="aggregate_by_group",
        required_answer_output_ids=("answer_1",),
        source_members=(member,),
        basis="The source supports the ranked aggregate.",
    )
