from ._helpers import *  # noqa: F403

from fervis.lookup.answer_program.expressions import NodeOutputRef
from fervis.lookup.answer_program.operations import AggregateSpec, ComputeSpec
from fervis.lookup.answer_program.values import LiteralType
from fervis.lookup.fact_planning.scalar_values import (
    source_derived_scalar_values_by_fact,
)


def test_pattern_prompt_projects_scalar_aggregate_choices_for_numeric_summary_evidence():
    request = FactPlanRequest(
        question="How many in-person sales happened this month?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="in-person sales count this month",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="count of in-person sales this month",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            _bound_source_fixture(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="fact_1",
                    answer_population=_answer_population(),
                    source=DraftRelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_sales_summary",
                    ),
                    cardinality="many",
                    available_field_ids=("location_id", "count"),
                    available_fields=(
                        SourceField(
                            field_id="location_id", type="uuid", roles=("identity",)
                        ),
                        SourceField(field_id="count", type="integer"),
                    ),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="source_1_evidence_1",
                            field_id="location_id",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1_evidence_2",
                            field_id="count",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_1",
                            match_basis_explanation=(
                                "answer_1 is fulfilled by count because count is "
                                "the numeric answer value."
                            ),
                            metric_measure_evidence_ids=("source_1_evidence_2",),
                        ),
                    ),
                )
            ),
        ),
    )

    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_scalar.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                    plan_shape="aggregate_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    choices = _text_prompt_section(
        prompt,
        label="Scalar aggregate operation choices",
        next_label="Decision Scope",
    )
    schema_text = json.dumps(
        PatternFactPlanTurnPrompt(
            request,
            plan_selection=BoundPlanSelectionSet(
                plan_selections=(
                    BoundSelectedSourceStrategy(
                        requested_fact_id="fact_1",
                        plan_selection_id="fact_1.aggregate_scalar.sb_1",
                        source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                        plan_shape="aggregate_scalar",
                        required_answer_output_ids=("answer_1",),
                        source_members=(
                            _bound_plan_member(request, source_binding_ids=("sb_1",)),
                        ),
                    ),
                )
            ),
        )
        .response_contract()
        .provider_schema
    )

    assert '<metric id="metric_1" kind="aggregate_field" field="count"' in choices
    assert 'allowed_functions="sum min max avg"' in choices
    assert '<function id="function_sum" value="sum"' in choices
    assert '<function id="function_avg" value="avg"' in choices
    assert '"metric"' in schema_text
    assert '"function"' in schema_text


def test_computed_scalar_uses_one_catalog_for_source_and_literal_values():
    source = BoundSource(
        id="sb_1",
        requested_fact_id="fact_1",
        answer_population=_answer_population(),
        source=DraftRelationSource(
            kind=SourceKind.API_READ,
            read_id="list_measurements",
        ),
        cardinality="many",
        available_field_ids=("amount",),
        available_fields=(SourceField(field_id="amount", type="decimal"),),
        evidence_items=(
            SourceEvidenceItem(
                evidence_id="source.amount",
                field_id="amount",
                row_cardinality="many",
            ),
        ),
        fulfillments=(
            SourceFulfillment(
                requested_fact_id="fact_1",
                answer_output_id="answer_1",
                match_basis_explanation="amount is the requested measure",
                metric_measure_evidence_ids=("source.amount",),
            ),
        ),
    )
    selected = BoundPlanSelectionSet(
        plan_selections=(
            BoundSelectedSourceStrategy(
                requested_fact_id="fact_1",
                plan_selection_id="fact_1.computed_scalar.sb_1",
                source_strategy_id="source_strategy.fact_1.computed_scalar.1",
                plan_shape="computed_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        BoundSourceStrategyMember(
                            source_candidate_id="candidate_1",
                            role_targets=(
                                BoundRoleTarget(
                                    requirement_id="answer_1",
                                    source_candidate_id="candidate_1",
                                    source_binding_ids=("sb_1",),
                                ),
                            ),
                        ),
                    ),
            ),
        )
    )
    request = FactPlanRequest(
        question="What is one tenth of the total measure?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="one tenth of the total measure",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(source,),
        available_values=(
            FactValue.literal(
                id="fraction",
                literal_type=LiteralType.NUMBER,
                value="0.1",
                applies_to_requested_fact_ids=("fact_1",),
            ),
        ),
    )
    prompt = PatternFactPlanTurnPrompt(request, plan_selection=selected)
    source_values = prompt.source_derived_scalar_values()
    sum_value = next(
        value
        for value in source_values
        if value.metric.function.value == "sum"
    )

    assert sum_value.value_id in _text_prompt_section(
        _pattern_fact_plan_prompt(request, plan_selection=selected),
        label="Operation input values",
        next_label="Bound sources",
    )

    program = compile_pattern_answer_plan(
        {
            "answers": [
                {
                    "pattern": "computed_scalar",
                    "requested_fact_id": "fact_1",
                    "answer_output_ids": ["answer_1"],
                    "scalar_inputs": [
                        {"input_id": "total", "value_id": sum_value.value_id},
                        {"input_id": "fraction", "value_id": "fraction"},
                    ],
                    "expression": [
                        {"input_id": "total"},
                        {"input_id": "fraction"},
                        {"operator": "multiply"},
                    ],
                    "output": {"scalar_id": "result"},
                }
            ]
        },
        bound_sources=(source,),
        requested_facts=request.question_contract.requested_facts,
        available_values=request.available_values,
        question_contract=request.question_contract,
        source_derived_scalar_values=source_values,
    )

    aggregate = next(
        operation for operation in program.operations if isinstance(operation.spec, AggregateSpec)
    )
    compute = next(
        operation for operation in program.operations if isinstance(operation.spec, ComputeSpec)
    )
    assert NodeOutputRef(
        node_id=aggregate.id,
        output_id=sum_value.metric.output_field_id,
    ) in (compute.spec.expression.left, compute.spec.expression.right)


def test_pattern_prompt_projects_scalar_aggregate_choices_for_one_row_summary_evidence():
    request = FactPlanRequest(
        question="How much revenue did we make this week?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="revenue made this week",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="total revenue for the week",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ,
                    read_id="list_sales_summary",
                ),
                cardinality="many",
                available_field_ids=("total_amount",),
                available_fields=(
                    SourceField(
                        field_id="total_amount",
                        type="decimal",
                        row_cardinality="one",
                    ),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.summary.total_amount",
                        field_id="total_amount",
                        row_cardinality="one",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation=(
                            "The one-row summary total_amount is the total revenue."
                        ),
                        metric_measure_evidence_ids=("source_1.summary.total_amount",),
                    ),
                ),
            ),
        ),
    )

    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_scalar.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                    plan_shape="aggregate_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    choices = _text_prompt_section(
        prompt,
        label="Scalar aggregate operation choices",
        next_label="Decision Scope",
    )

    assert (
        '<metric id="metric_1" kind="aggregate_field" field="total_amount"' in choices
    )
    assert 'allowed_functions="sum min max avg"' in choices


def test_pattern_prompt_uses_metric_measure_evidence_not_generic_scope_for_metrics():
    request = FactPlanRequest(
        question="Which staff earned the most compensation this month?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="staff earned the most compensation this month",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="staff member",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            _bound_source_fixture(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="fact_1",
                    answer_population=_answer_population(),
                    source=DraftRelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_shift_compensation",
                    ),
                    cardinality="many",
                    available_field_ids=(
                        "staff_id",
                        "calculated_pay",
                        "amount_paid",
                        "payment_status",
                    ),
                    available_fields=(
                        SourceField(field_id="staff_id", type="uuid"),
                        SourceField(field_id="calculated_pay", type="decimal"),
                        SourceField(field_id="amount_paid", type="decimal"),
                        SourceField(field_id="payment_status", type="choice"),
                    ),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="source_1.data.staff_id",
                            field_id="staff_id",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.data.calculated_pay",
                            field_id="calculated_pay",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.data.amount_paid",
                            field_id="amount_paid",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.data.payment_status",
                            field_id="payment_status",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_1",
                            match_basis_explanation=(
                                "staff_id identifies the answer and calculated_pay "
                                "is the measure for ranking."
                            ),
                            entity_evidence=candidate_key_evidence(
                                "staff_id",
                                entity_kind="staff",
                                key_id="staff_key",
                                field_evidence_ids=("source_1.data.staff_id",),
                            ),
                            metric_measure_evidence_ids=(
                                "source_1.data.calculated_pay",
                            ),
                        ),
                    ),
                )
            ),
        ),
    )
    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_by_group.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_by_group.1",
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    assert "Grouped aggregate operation choices:" in prompt
    assert 'field="calculated_pay"' in prompt
    assert 'field="amount_paid"' not in prompt


def test_pattern_prompt_requires_metric_evidence_not_count_basis_for_aggregate_metric():
    request = FactPlanRequest(
        question="How much cash was deposited this month?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="total cash deposited this month",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="total cash deposited",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            _bound_source_fixture(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="fact_1",
                    answer_population=_answer_population(),
                    source=DraftRelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_cash_deposit_list",
                    ),
                    cardinality="many",
                    available_field_ids=("cash_deposit_id", "amount"),
                    available_fields=(
                        SourceField(
                            field_id="cash_deposit_id",
                            type="uuid",
                            roles=("identity",),
                        ),
                        SourceField(field_id="amount", type="decimal"),
                    ),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="source_1.data.cash_deposit_id",
                            field_id="cash_deposit_id",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.data.amount",
                            field_id="amount",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_1",
                            match_basis_explanation=(
                                "amount is the measured quantity for total deposits; "
                                "cash_deposit_id only anchors deposit rows."
                            ),
                            metric_measure_evidence_ids=("source_1.data.amount",),
                            row_count_basis_evidence_ids=(
                                "source_1.data.cash_deposit_id",
                            ),
                        ),
                    ),
                )
            ),
        ),
    )
    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_scalar.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_scalar.1",
                    plan_shape="aggregate_scalar",
                    required_answer_output_ids=("answer_1",),
                    source_members=(
                        _bound_plan_member(request, source_binding_ids=("sb_1",)),
                    ),
                ),
            )
        ),
    )

    payload = _json_prompt_section(
        prompt,
        label="Required fulfillment evidence",
        next_label="Decision Scope",
    )

    assert payload == {
        "required_fulfillment_evidence": [
            {
                "requested_fact_id": "fact_1",
                "answer_output_id": "answer_1",
                "source_binding_id": "sb_1",
                "must_use_evidence": [
                    {
                        "evidence_id": "source_1.data.amount",
                        "field_id": "amount",
                    }
                ],
            }
        ]
    }


def test_pattern_prompt_requires_metric_evidence_even_when_answer_value_exists():
    request = FactPlanRequest(
        question="Which location had the highest total payroll spend this month?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="location with highest total payroll spend",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="location",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            _bound_source_fixture(
                BoundSource(
                    id="sb_1",
                    requested_fact_id="fact_1",
                    answer_population=_answer_population(),
                    source=DraftRelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_shift_compensation",
                    ),
                    cardinality="many",
                    available_field_ids=("location_id", "calculated_pay"),
                    available_fields=(
                        SourceField(field_id="location_id", type="uuid"),
                        SourceField(field_id="calculated_pay", type="decimal"),
                    ),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="source_1.data.location_id",
                            field_id="location_id",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_1.data.calculated_pay",
                            field_id="calculated_pay",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_1",
                            match_basis_explanation=(
                                "location_id identifies the answer and calculated_pay "
                                "is the measured quantity for ranking."
                            ),
                            entity_evidence=candidate_key_evidence(
                                "location_id",
                                entity_kind="location",
                                key_id="location_key",
                                field_evidence_ids=("source_1.data.location_id",),
                            ),
                            metric_measure_evidence_ids=(
                                "source_1.data.calculated_pay",
                            ),
                        ),
                    ),
                )
            ),
        ),
    )
    prompt = _pattern_fact_plan_prompt(
        request,
        plan_selection=BoundPlanSelectionSet(
            plan_selections=(
                BoundSelectedSourceStrategy(
                    requested_fact_id="fact_1",
                    plan_selection_id="fact_1.aggregate_by_group.sb_1",
                    source_strategy_id="source_strategy.fact_1.aggregate_by_group.1",
                    plan_shape="aggregate_by_group",
                    required_answer_output_ids=("answer_1",),
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
    assert (
        '<metric id="metric_1" kind="aggregate_field" field="calculated_pay" '
        'type="decimal" allowed_functions="sum min max avg" />'
    ) in prompt


def test_pattern_prompt_does_not_offer_identity_source_numeric_fields_as_metrics():
    request = FactPlanRequest(
        question="Which staff earned the most compensation this month?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="staff member earned the most compensation this month",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            role="ANSWER_VALUE",
                            description="the staff member",
                        ),
                        RequestedFactAnswerOutput(
                            id="answer_2",
                            role="ANSWER_VALUE",
                            description="the amount paid",
                        ),
                    ),
                ),
            )
        ),
        relation_catalog=RelationCatalog(reads=()),
        bound_sources=(
            BoundSource(
                id="sb_1",
                requested_fact_id="fact_1",
                answer_population=_answer_population(),
                source=DraftRelationSource(
                    kind=SourceKind.API_READ, read_id="list_staff"
                ),
                cardinality="many",
                available_field_ids=("staff_id", "daily_base_pay"),
                available_fields=(
                    SourceField(field_id="staff_id", type="uuid", roles=("identity",)),
                    SourceField(field_id="daily_base_pay", type="decimal"),
                ),
                evidence_items=(
                    SourceEvidenceItem(
                        evidence_id="source_1.data.staff_id",
                        field_id="staff_id",
                        row_cardinality="many",
                    ),
                ),
                fulfillments=(
                    SourceFulfillment(
                        requested_fact_id="fact_1",
                        answer_output_id="answer_1",
                        match_basis_explanation="staff_id identifies the staff member.",
                        entity_evidence=candidate_key_evidence(
                            "staff_id",
                            entity_kind="staff",
                            key_id="staff_key",
                            field_evidence_ids=("source_1.data.staff_id",),
                        ),
                    ),
                ),
            ),
            _bound_source_fixture(
                BoundSource(
                    id="sb_2",
                    requested_fact_id="fact_1",
                    answer_population=_answer_population(),
                    source=DraftRelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_shift_compensation",
                    ),
                    cardinality="many",
                    available_field_ids=("staff_id", "amount_paid"),
                    available_fields=(
                        SourceField(
                            field_id="staff_id", type="uuid", roles=("identity",)
                        ),
                        SourceField(field_id="amount_paid", type="decimal"),
                    ),
                    evidence_items=(
                        SourceEvidenceItem(
                            evidence_id="source_2.data.staff_id",
                            field_id="staff_id",
                            row_cardinality="many",
                        ),
                        SourceEvidenceItem(
                            evidence_id="source_2.data.amount_paid",
                            field_id="amount_paid",
                            row_cardinality="many",
                        ),
                    ),
                    fulfillments=(
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_1",
                            match_basis_explanation="staff_id identifies the staff member.",
                            entity_evidence=candidate_key_evidence(
                                "staff_id",
                                entity_kind="staff",
                                key_id="staff_key",
                                field_evidence_ids=("source_2.data.staff_id",),
                            ),
                        ),
                        SourceFulfillment(
                            requested_fact_id="fact_1",
                            answer_output_id="answer_2",
                            match_basis_explanation="amount_paid is the paid amount.",
                            metric_measure_evidence_ids=("source_2.data.amount_paid",),
                        ),
                    ),
                )
            ),
        ),
    )
    plan_selection = BoundPlanSelectionSet(
        plan_selections=(
            BoundSelectedSourceStrategy(
                requested_fact_id="fact_1",
                plan_selection_id="fact_1.aggregate_by_group.relations",
                source_strategy_id="source_strategy.fact_1.aggregate_by_group.1",
                plan_shape="aggregate_by_group",
                required_answer_output_ids=("answer_1",),
                source_members=(
                    _bound_plan_member(
                        request,
                        source_binding_ids=("sb_1", "sb_2"),
                    ),
                ),
            ),
        )
    )

    prompt = _pattern_fact_plan_prompt(request, plan_selection=plan_selection)
    assert "Grouped aggregate operation choices:" in prompt
    assert '<source_binding id="sb_2" read="list_shift_compensation">' in prompt
    assert 'field="amount_paid"' in prompt
    assert "daily_base_pay" not in prompt
