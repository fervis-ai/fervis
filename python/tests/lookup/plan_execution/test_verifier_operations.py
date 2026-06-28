import pytest

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.verification import (
    verify_fact_plan as verify_fact_plan_impl,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.fact_plan.fact_plan import (
    AnswerPlan,
    FactFulfillment,
    FactPlan,
)
from fervis.lookup.fact_plan.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    AntiJoinSpec,
    ComputeSpec,
    FilterSpec,
    JoinKey,
    Operation,
    Predicate,
    PredicateOperator,
    ProjectField,
    ProjectSpec,
    RankSpec,
    RelationRole,
    RelationRoleRef,
    SortDirection,
    SortKey,
    TiePolicy,
    UniversalConditionSpec,
)
from fervis.lookup.fact_plan.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import memory_row_source_id
from fervis.lookup.fact_plan.values import (
    FactValue,
    LiteralType,
    RankLimitUse,
    ScalarInputUse,
    ValueUse,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
)
from fervis.lookup.fact_plan.render_spec import (
    RenderRelationOutput,
    RenderScalarOutput,
    RenderSpec,
)


def _plan_with(operation: Operation) -> FactPlan:
    relation_ids = _input_relation_ids(operation)
    render_field = _render_field(operation)
    fulfillment = _fulfillment(operation, render_field)
    return FactPlan(
        outcome=AnswerPlan(
            fulfillment=fulfillment,
            relations=tuple(_relation(item) for item in sorted(relation_ids)),
            operations=(operation,),
            render_spec=RenderSpec(
                relation_outputs=(
                    ()
                    if isinstance(operation.spec, ComputeSpec)
                    else (
                        RenderRelationOutput(
                            id="answer",
                            relation_id="result",
                            field_id=render_field,
                        ),
                    )
                )
            ),
        )
    )


def _question_contract(
    description: str = "answer",
    *,
    binding_target_ids: tuple[str, ...] = ("answer",),
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description=description,
                answer_outputs=tuple(
                    RequestedFactAnswerOutput(id=binding_target_id)
                    for binding_target_id in binding_target_ids
                ),
            ),
        )
    )


def verify_fact_plan(plan: FactPlan, **kwargs):
    catalog = kwargs.pop("catalog", RelationCatalog())
    return verify_fact_plan_impl(
        plan,
        question_contract=kwargs.pop(
            "question_contract",
            _question_contract(
                _default_description(plan),
                binding_target_ids=_render_output_ids(plan),
            ),
        ),
        catalog=catalog,
        memory_relations=kwargs.pop("memory_relations", _memory_relations(plan)),
        **kwargs,
    )


def _default_description(plan: FactPlan) -> str:
    outcome = plan.outcome
    render_spec = outcome.render_spec
    if render_spec is not None and render_spec.scalar_outputs:
        return render_spec.scalar_outputs[0].scalar_id
    if render_spec is not None and render_spec.relation_outputs:
        return render_spec.relation_outputs[0].field_id
    return "name"


def _render_output_ids(plan: FactPlan) -> tuple[str, ...]:
    outcome = plan.outcome
    if outcome.render_spec is None:
        return ("answer",)
    binding_target_ids = tuple(
        dict.fromkeys(
            slot.id
            for slot in (
                *outcome.render_spec.relation_outputs,
                *outcome.render_spec.scalar_outputs,
            )
        )
    )
    return binding_target_ids or ("answer",)


def _fulfillment(
    operation: Operation,
    render_field: str,
) -> tuple[FactFulfillment, ...]:
    if isinstance(operation.spec, ComputeSpec):
        return (
            FactFulfillment(
                requested_fact_id="rf_answer",
                answer_output_id="answer",
                render_output_id="answer",
            ),
        )
    return (
        FactFulfillment(
            requested_fact_id="rf_answer",
            answer_output_id="answer",
            render_output_id="answer",
        ),
    )


def _render_field(operation: Operation) -> str:
    spec = operation.spec
    if isinstance(spec, AntiJoinSpec):
        return spec.output_fields[0].output or spec.output_fields[0].source
    if isinstance(spec, UniversalConditionSpec):
        return spec.output_fields[0].output or spec.output_fields[0].source
    if isinstance(spec, ProjectSpec):
        return spec.fields[0].output or spec.fields[0].source
    if isinstance(spec, AggregateSpec):
        return spec.group_by[0] if spec.group_by else spec.aggregations[0].output_field
    return "name"


def _input_relation_ids(operation: Operation) -> set[str]:
    spec = operation.spec
    if isinstance(spec, (AntiJoinSpec,)):
        return {spec.candidate.relation_id, spec.observed.relation_id}
    if isinstance(spec, UniversalConditionSpec):
        return {
            spec.candidate_subject.relation_id,
            spec.required_dimension.relation_id,
            spec.observation.relation_id,
        }
    if isinstance(spec, RankSpec):
        return {spec.input_relation}
    if isinstance(spec, ProjectSpec):
        return {spec.input_relation}
    if isinstance(spec, FilterSpec):
        return {spec.input_relation}
    if isinstance(spec, AggregateSpec):
        return {spec.input_relation}
    return set()


def _memory_relations(plan: FactPlan) -> tuple[RelationRows, ...]:
    outcome = plan.outcome
    if not isinstance(outcome, AnswerPlan):
        return ()
    return tuple(
        RelationRows(
            id=relation.id,
            rows=({field.field_id: field.field_id for field in relation.fields},),
            grain_keys=relation.grain_keys,
        )
        for relation in outcome.relations
    )


def _relation(relation_id: str) -> Relation:
    grain_keys = {
        "candidate_rows": ("candidate.id",),
        "observed_rows": ("observed.id",),
        "candidate_subjects": ("subject_id",),
        "required_dimensions": ("dimension_id",),
        "observations": ("obs_subject_id", "obs_dimension_id"),
    }.get(relation_id, ("id",))

    def roles_for(field_id: str) -> tuple[FieldBindingRole, ...]:
        if field_id in grain_keys:
            return (FieldBindingRole.IDENTITY,)
        if field_id in {"field.quantity", "field.value", "field.other", "field.amount"}:
            return (FieldBindingRole.PREDICATE,)
        return (FieldBindingRole.OUTPUT,)

    def field(field_id: str) -> RelationField:
        return RelationField(field_id=field_id, roles=roles_for(field_id))

    return Relation(
        id=relation_id,
        source=RelationSource(
            kind=SourceKind.MEMORY_READ,
            memory_relation_id=relation_id,
        ),
        fields=(
            field("id"),
            field("candidate.id"),
            field("observed.id"),
            field("candidate.name"),
            field("subject_id"),
            field("obs_subject_id"),
            field("dimension_id"),
            field("obs_dimension_id"),
            field("field.quantity"),
            field("field.value"),
            field("field.other"),
            field("field.first"),
            field("field.second"),
            field("field.group"),
            field("field.amount"),
            field("subject_name"),
            field("name"),
            field("total"),
        ),
    )


def test_anti_join_requires_role_refs_join_keys_and_output_fields():
    invalid = Operation(
        id="missing_items",
        spec=AntiJoinSpec(
            candidate=RelationRoleRef(
                relation_id="candidate_rows",
                role=RelationRole.ANTI_JOIN_CANDIDATE,
                required_identity_fields=("candidate.id",),
            ),
            observed=RelationRoleRef(
                relation_id="observed_rows",
                role=RelationRole.ANTI_JOIN_OBSERVED,
                required_identity_fields=("observed.id",),
            ),
            join_keys=(),
            output_fields=(ProjectField(source="candidate.name"),),
        ),
        output_relation="result",
    )
    valid = Operation(
        id="missing_items",
        spec=AntiJoinSpec(
            candidate=RelationRoleRef(
                relation_id="candidate_rows",
                role=RelationRole.ANTI_JOIN_CANDIDATE,
                required_identity_fields=("candidate.id",),
            ),
            observed=RelationRoleRef(
                relation_id="observed_rows",
                role=RelationRole.ANTI_JOIN_OBSERVED,
                required_identity_fields=("observed.id",),
            ),
            join_keys=(JoinKey(left="candidate.id", right="observed.id"),),
            output_fields=(ProjectField(source="candidate.name"),),
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="join keys"):
        verify_fact_plan(_plan_with(invalid))
    verify_fact_plan(_plan_with(valid))


def test_anti_join_rejects_identity_binding_as_output_field():
    plan = _plan_with(
        Operation(
            id="missing_items",
            spec=AntiJoinSpec(
                candidate=RelationRoleRef(
                    relation_id="candidate_rows",
                    role=RelationRole.ANTI_JOIN_CANDIDATE,
                    required_identity_fields=("candidate.id",),
                ),
                observed=RelationRoleRef(
                    relation_id="observed_rows",
                    role=RelationRole.ANTI_JOIN_OBSERVED,
                    required_identity_fields=("observed.id",),
                ),
                join_keys=(JoinKey(left="candidate.id", right="observed.id"),),
                output_fields=(ProjectField(source="candidate.id"),),
            ),
            output_relation="result",
        )
    )

    with pytest.raises(VerificationError, match="wrong binding role"):
        verify_fact_plan(plan)


def test_universal_condition_requires_subject_dimension_and_predicate():
    invalid = Operation(
        id="all_rows_match",
        spec=UniversalConditionSpec(
            candidate_subject=RelationRoleRef(
                relation_id="candidate_subjects",
                role=RelationRole.UNIVERSAL_CANDIDATE_SUBJECT,
                required_identity_fields=("subject_id",),
            ),
            required_dimension=RelationRoleRef(
                relation_id="required_dimensions",
                role=RelationRole.UNIVERSAL_REQUIRED_DIMENSION,
                required_identity_fields=("dimension_id",),
            ),
            observation=RelationRoleRef(
                relation_id="observations",
                role=RelationRole.UNIVERSAL_OBSERVATION,
                required_identity_fields=("obs_subject_id", "obs_dimension_id"),
            ),
            subject_keys=(JoinKey(left="subject_id", right="obs_subject_id"),),
            dimension_keys=(JoinKey(left="dimension_id", right="obs_dimension_id"),),
            predicate=Predicate(left="", operator=""),
            output_fields=(ProjectField(source="subject_name"),),
        ),
        output_relation="result",
    )
    valid = Operation(
        id="all_rows_match",
        spec=UniversalConditionSpec(
            candidate_subject=RelationRoleRef(
                relation_id="candidate_subjects",
                role=RelationRole.UNIVERSAL_CANDIDATE_SUBJECT,
                required_identity_fields=("subject_id",),
            ),
            required_dimension=RelationRoleRef(
                relation_id="required_dimensions",
                role=RelationRole.UNIVERSAL_REQUIRED_DIMENSION,
                required_identity_fields=("dimension_id",),
            ),
            observation=RelationRoleRef(
                relation_id="observations",
                role=RelationRole.UNIVERSAL_OBSERVATION,
                required_identity_fields=("obs_subject_id", "obs_dimension_id"),
            ),
            subject_keys=(JoinKey(left="subject_id", right="obs_subject_id"),),
            dimension_keys=(JoinKey(left="dimension_id", right="obs_dimension_id"),),
            predicate=Predicate(
                left="field.quantity",
                operator=PredicateOperator.LTE,
                right="field.other",
            ),
            output_fields=(ProjectField(source="subject_name"),),
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="predicate"):
        verify_fact_plan(_plan_with(invalid))
    verify_fact_plan(_plan_with(valid))


def test_rank_requires_ordering_and_deterministic_tie_policy():
    invalid = Operation(
        id="ranked",
        spec=RankSpec(input_relation="totals", order_by=(), tie_policy="", limit=1),
        output_relation="result",
    )
    valid = Operation(
        id="ranked",
        spec=RankSpec(
            input_relation="totals",
            order_by=(SortKey(field="total", direction=SortDirection.DESC),),
            tie_policy=TiePolicy.FIELD,
            tie_breakers=(SortKey(field="name", direction=SortDirection.ASC),),
            limit=1,
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="tie policy"):
        verify_fact_plan(_plan_with(invalid))
    valid_plan = _plan_with(valid)
    valid_plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=valid_plan.outcome.fulfillment,
            value_uses=(
                ValueUse(
                    id="use_rank_limit",
                    value_id="rank_limit",
                    target=RankLimitUse(operation_id="ranked"),
                ),
            ),
            relations=valid_plan.outcome.relations,
            operations=valid_plan.outcome.operations,
            render_spec=valid_plan.outcome.render_spec,
        )
    )
    verify_fact_plan(
        valid_plan,
        available_values=(
            FactValue.literal(
                id="rank_limit",
                literal_type=LiteralType.NUMBER,
                value="1",
            ),
        ),
    )


def test_compute_references_scalar_inputs_only():
    invalid = Operation(
        id="remaining",
        spec=ComputeSpec(
            expression="target - total", scalar_inputs=(), output_scalar="x"
        ),
    )
    valid = Operation(
        id="remaining",
        spec=ComputeSpec(
            expression="target - total",
            scalar_inputs=("target", "total"),
            output_scalar="remaining",
        ),
    )

    with pytest.raises(VerificationError, match="scalar inputs"):
        verify_fact_plan(_plan_with(invalid))
    verify_fact_plan(
        FactPlan(
            outcome=AnswerPlan(
                fulfillment=(
                    FactFulfillment(
                        requested_fact_id="rf_answer",
                        answer_output_id="answer",
                        render_output_id="answer",
                    ),
                    FactFulfillment(
                        requested_fact_id="rf_answer",
                        answer_output_id="remaining",
                        render_output_id="remaining",
                    ),
                ),
                relations=(_relation("rows"),),
                value_uses=(
                    ValueUse(
                        id="use_target",
                        value_id="target_value",
                        target=ScalarInputUse(
                            operation_id="remaining",
                            input_id="target",
                        ),
                    ),
                    ValueUse(
                        id="use_total",
                        value_id="total_value",
                        target=ScalarInputUse(
                            operation_id="remaining",
                            input_id="total",
                        ),
                    ),
                ),
                operations=(
                    Operation(
                        id="project_answer",
                        spec=ProjectSpec(
                            input_relation="rows",
                            fields=(ProjectField(source="name", output="name"),),
                        ),
                        output_relation="result",
                    ),
                    valid,
                ),
                render_spec=RenderSpec(
                    relation_outputs=(
                        RenderRelationOutput(
                            id="answer",
                            relation_id="result",
                            field_id="name",
                        ),
                    ),
                    scalar_outputs=(
                        RenderScalarOutput(
                            id="remaining",
                            scalar_id="remaining",
                        ),
                    ),
                ),
            )
        ),
        available_values=(
            FactValue.literal(
                id="target_value",
                literal_type=LiteralType.NUMBER,
                value="100",
                proof_refs=("known_input:target_value",),
            ),
            FactValue.literal(
                id="total_value",
                literal_type=LiteralType.NUMBER,
                value="40",
                proof_refs=("prior:total_value",),
            ),
        ),
    )


def test_one_answer_output_can_be_fulfilled_by_multiple_distinct_render_outputs():
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="day_1_amount",
                ),
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="day_2_amount",
                ),
            ),
            relations=(
                Relation(
                    id="day_1_rows",
                    source=RelationSource(
                        kind=SourceKind.MEMORY_READ,
                        memory_relation_id="day_1_rows",
                    ),
                    fields=(
                        RelationField(
                            field_id="amount",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
                Relation(
                    id="day_2_rows",
                    source=RelationSource(
                        kind=SourceKind.MEMORY_READ,
                        memory_relation_id="day_2_rows",
                    ),
                    fields=(
                        RelationField(
                            field_id="amount",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_day_1",
                    spec=ProjectSpec(
                        input_relation="day_1_rows",
                        fields=(ProjectField(source="amount", output="amount"),),
                    ),
                    output_relation="day_1_result",
                ),
                Operation(
                    id="project_day_2",
                    spec=ProjectSpec(
                        input_relation="day_2_rows",
                        fields=(ProjectField(source="amount", output="amount"),),
                    ),
                    output_relation="day_2_result",
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="day_1_amount",
                        relation_id="day_1_result",
                        field_id="amount",
                    ),
                    RenderRelationOutput(
                        id="day_2_amount",
                        relation_id="day_2_result",
                        field_id="amount",
                    ),
                ),
            ),
        )
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_answer",
                description="Amounts for each requested day.",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer",
                        description="Amounts shown separately by day.",
                    ),
                ),
            ),
        )
    )

    assert verify_fact_plan(plan, question_contract=question_contract) is plan


def test_compute_expression_allows_numeric_constants():
    operation = Operation(
        id="remaining",
        spec=ComputeSpec(
            expression="target - 1",
            scalar_inputs=("target",),
            output_scalar="remaining",
        ),
    )
    plan = _plan_with(operation)
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=plan.outcome.fulfillment,
            value_uses=(
                ValueUse(
                    id="use_target",
                    value_id="target_value",
                    target=ScalarInputUse(
                        operation_id="remaining",
                        input_id="target",
                    ),
                ),
            ),
            relations=plan.outcome.relations,
            operations=plan.outcome.operations,
            render_spec=RenderSpec(
                relation_outputs=(),
                scalar_outputs=(
                    RenderScalarOutput(id="answer", scalar_id="remaining"),
                ),
            ),
        )
    )

    assert verify_fact_plan(
        plan,
        available_values=(
            FactValue.literal(
                id="target_value",
                literal_type=LiteralType.NUMBER,
                value="100",
                proof_refs=("known_input:target",),
            ),
        ),
    )


def test_compute_expression_rejects_unsupported_ast_forms():
    operation = Operation(
        id="remaining",
        spec=ComputeSpec(
            expression="max(target, total)",
            scalar_inputs=("target", "total"),
            output_scalar="remaining",
        ),
    )

    with pytest.raises(VerificationError, match="unsupported compute expression"):
        verify_fact_plan(_plan_with(operation))


def test_compute_only_literal_answer_requires_evidence_proof():
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="answer",
                ),
            ),
            value_uses=(
                ValueUse(
                    id="use_target",
                    value_id="target_value",
                    target=ScalarInputUse(
                        operation_id="remaining",
                        input_id="target",
                    ),
                ),
                ValueUse(
                    id="use_total",
                    value_id="total_value",
                    target=ScalarInputUse(
                        operation_id="remaining",
                        input_id="total",
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="remaining",
                    spec=ComputeSpec(
                        expression="target - total",
                        scalar_inputs=("target", "total"),
                        output_scalar="remaining",
                    ),
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(),
                scalar_outputs=(
                    RenderScalarOutput(id="answer", scalar_id="remaining"),
                ),
            ),
        )
    )

    with pytest.raises(VerificationError, match="evidence proof"):
        verify_fact_plan(
            plan,
            available_values=(
                FactValue.literal(
                    id="target_value",
                    literal_type=LiteralType.NUMBER,
                    value="100",
                ),
                FactValue.literal(
                    id="total_value",
                    literal_type=LiteralType.NUMBER,
                    value="40",
                ),
            ),
        )


def test_scalar_only_terminal_answer_cannot_launder_unrelated_relation_evidence():
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="answer",
                ),
            ),
            relations=(_relation("rows"),),
            value_uses=(
                ValueUse(
                    id="use_target",
                    value_id="target_value",
                    target=ScalarInputUse(
                        operation_id="remaining",
                        input_id="target",
                    ),
                ),
                ValueUse(
                    id="use_total",
                    value_id="total_value",
                    target=ScalarInputUse(
                        operation_id="remaining",
                        input_id="total",
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_unrelated_evidence",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="unrelated_evidence",
                ),
                Operation(
                    id="remaining",
                    spec=ComputeSpec(
                        expression="target - total",
                        scalar_inputs=("target", "total"),
                        output_scalar="remaining",
                    ),
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(),
                scalar_outputs=(
                    RenderScalarOutput(id="answer", scalar_id="remaining"),
                ),
            ),
        )
    )

    with pytest.raises(VerificationError, match="terminal relation output"):
        verify_fact_plan(
            plan,
            available_values=(
                FactValue.literal(
                    id="target_value",
                    literal_type=LiteralType.NUMBER,
                    value="100",
                ),
                FactValue.literal(
                    id="total_value",
                    literal_type=LiteralType.NUMBER,
                    value="40",
                ),
            ),
        )


def test_unrendered_compute_outputs_are_not_legal_answer_work():
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="answer",
                ),
            ),
            relations=(_relation("rows"),),
            value_uses=(
                ValueUse(
                    id="use_target",
                    value_id="target_value",
                    target=ScalarInputUse(
                        operation_id="unused_compute",
                        input_id="target",
                    ),
                ),
            ),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(ProjectField(source="name"),),
                    ),
                    output_relation="result",
                ),
                Operation(
                    id="unused_compute",
                    spec=ComputeSpec(
                        expression="target",
                        scalar_inputs=("target",),
                        output_scalar="unused_total",
                    ),
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unrendered scalar output"):
        verify_fact_plan(
            plan,
            available_values=(
                FactValue.literal(
                    id="target_value",
                    literal_type=LiteralType.NUMBER,
                    value="100",
                ),
            ),
        )


def test_render_output_ids_must_be_unique():
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="answer",
                ),
            ),
            relations=(_relation("rows"),),
            operations=(
                Operation(
                    id="project_answer",
                    spec=ProjectSpec(
                        input_relation="rows",
                        fields=(
                            ProjectField(source="name"),
                            ProjectField(source="total"),
                        ),
                    ),
                    output_relation="result",
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                    RenderRelationOutput(
                        id="answer", relation_id="result", field_id="total"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate render output"):
        verify_fact_plan(plan)


def test_relation_answer_requires_render_outputs():
    plan = _plan_with(
        Operation(
            id="project",
            spec=ProjectSpec(
                input_relation="rows",
                fields=(ProjectField(source="name"),),
            ),
            output_relation="result",
        )
    )
    assert isinstance(plan.outcome, AnswerPlan)
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    render_output_id="answer",
                ),
            ),
            relations=plan.outcome.relations,
            operations=plan.outcome.operations,
            render_spec=RenderSpec(relation_outputs=()),
        )
    )

    with pytest.raises(VerificationError, match="render output"):
        verify_fact_plan(plan)


def test_project_requires_output_relation():
    plan = _plan_with(
        Operation(
            id="project",
            spec=ProjectSpec(
                input_relation="rows",
                fields=(ProjectField(source="name"),),
            ),
        )
    )

    with pytest.raises(VerificationError, match="output relation"):
        verify_fact_plan(plan)


def test_predicate_requires_exactly_one_rhs_for_binary_operators():
    missing_rhs = Operation(
        id="filter",
        spec=FilterSpec(
            input_relation="rows",
            predicate=Predicate(left="field.value", operator=PredicateOperator.EQUALS),
        ),
        output_relation="result",
    )
    duplicate_rhs = Operation(
        id="filter",
        spec=FilterSpec(
            input_relation="rows",
            predicate=Predicate(
                left="field.value",
                operator=PredicateOperator.EQUALS,
                right="field.other",
                right_scalar="literal",
            ),
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="right-hand side"):
        verify_fact_plan(_plan_with(missing_rhs))
    with pytest.raises(VerificationError, match="right-hand side"):
        verify_fact_plan(_plan_with(duplicate_rhs))


def test_predicate_rejects_rhs_for_unary_operators():
    plan = _plan_with(
        Operation(
            id="filter",
            spec=FilterSpec(
                input_relation="rows",
                predicate=Predicate(
                    left="field.value",
                    operator=PredicateOperator.IS_NULL,
                    right_scalar="literal",
                ),
            ),
            output_relation="result",
        )
    )

    with pytest.raises(VerificationError, match="right-hand side"):
        verify_fact_plan(plan)


def test_count_aggregate_can_fulfill_requested_answer_output():
    plan = FactPlan(
        outcome=AnswerPlan(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="row_count",
                    render_output_id="row_count",
                ),
            ),
            relations=(_relation("rows"),),
            operations=(
                Operation(
                    id="count_rows",
                    spec=AggregateSpec(
                        input_relation="rows",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.COUNT,
                                output_field="row_count",
                            ),
                        ),
                    ),
                    output_relation="counts",
                ),
            ),
            render_spec=RenderSpec(
                relation_outputs=(
                    RenderRelationOutput(
                        id="row_count",
                        relation_id="counts",
                        field_id="row_count",
                    ),
                )
            ),
        )
    )

    assert (
        verify_fact_plan(
            plan,
            question_contract=_question_contract(
                "row count",
                binding_target_ids=("row_count",),
            ),
        )
        is plan
    )


def test_project_and_aggregate_reject_duplicate_output_fields():
    duplicate_project = Operation(
        id="project",
        spec=ProjectSpec(
            input_relation="rows",
            fields=(
                ProjectField(source="field.first", output="value"),
                ProjectField(source="field.second", output="value"),
            ),
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="duplicate output field"):
        verify_fact_plan(_plan_with(duplicate_project))

    duplicate_aggregate = Operation(
        id="aggregate",
        spec=AggregateSpec(
            input_relation="rows",
            group_by=("field.group",),
            aggregations=(
                AggregationSpec(
                    function=AggregationFunction.SUM,
                    input_field="field.amount",
                    output_field="field.group",
                ),
            ),
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="duplicate output field"):
        verify_fact_plan(_plan_with(duplicate_aggregate))


def test_rank_rejects_non_positive_limit():
    plan = _plan_with(
        Operation(
            id="ranked",
            spec=RankSpec(
                input_relation="rows",
                order_by=(SortKey(field="field.value", direction=SortDirection.ASC),),
                tie_policy=TiePolicy.FIELD,
                tie_breakers=(
                    SortKey(field="field.value", direction=SortDirection.ASC),
                ),
                limit=0,
            ),
            output_relation="result",
        )
    )

    with pytest.raises(VerificationError, match="positive"):
        verify_fact_plan(plan)
