import pytest

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.verification import (
    verify_fact_plan as verify_fact_plan_impl,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.fact_plan.fact_plan import FactPlan
from fervis.lookup.answer_program.operations import (
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
from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    ExpressionBinaryOperator,
    FieldRef,
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    LiteralType,
)
from fervis.lookup.question_contract import (
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
)
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ScalarResultOutput,
    ResultProjection,
)


def _rank_limit(value: int) -> ConstantRef:
    return ConstantRef(
        constant_id=f"rank-limit.{value}",
        version_ref="rank@1",
        value=FactValue.literal(
            id=f"rank-limit.{value}",
            literal_type=LiteralType.NUMBER,
            value=str(value),
        ),
    )


def _number_ref(
    *,
    input_id: str,
    value: str,
    proof_refs: tuple[str, ...] = (),
) -> ConstantRef:
    return ConstantRef(
        constant_id=f"test.{input_id}",
        version_ref="test@1",
        value=FactValue.literal(
            id=f"value.{input_id}",
            literal_type=LiteralType.NUMBER,
            value=value,
            proof_refs=proof_refs,
        ),
    )


def _subtract(left, right) -> BinaryExpression:
    return BinaryExpression(
        operator=ExpressionBinaryOperator.SUBTRACT,
        left=left,
        right=right,
    )


def _plan_with(operation: Operation) -> FactPlan:
    relation_ids = _input_relation_ids(operation)
    render_field = _render_field(operation)
    fulfillment = _fulfillment(operation, render_field)
    return FactPlan(
        outcome=AnswerProgram(
            fulfillment=fulfillment,
            relations=tuple(_relation(item) for item in sorted(relation_ids)),
            operations=(operation,),
            result_projection=ResultProjection(
                relation_outputs=(
                    ()
                    if isinstance(operation.spec, ComputeSpec)
                    else (
                        RelationResultOutput(
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
                    RequestedFactAnswerOutput(id=binding_target_id, role="ANSWER_VALUE")
                    for binding_target_id in binding_target_ids
                ),
            ),
        )
    )


def verify_fact_plan(plan: FactPlan, **kwargs):
    catalog = kwargs.pop("catalog", RelationCatalog())
    question_contract = kwargs.pop(
        "question_contract",
        _question_contract(
            _default_description(plan),
            binding_target_ids=_result_output_ids(plan),
        ),
    )
    memory_relations = kwargs.pop("memory_relations", _memory_relations(plan))
    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.instantiation import (
        ExecutionEnvironment,
        instantiate_answer_program,
    )

    if not isinstance(plan.outcome, AnswerProgram):
        return verify_fact_plan_impl(
            plan,
            question_contract=question_contract,
            catalog=catalog,
            memory_relations=memory_relations,
            **kwargs,
        )
    program, bindings = compile_answer_program(
        plan.outcome,
        question_contract=question_contract,
        catalog=catalog,
        bindings=plan.bindings,
        memory_relations=memory_relations,
    )
    instantiate_answer_program(
        program,
        bindings,
        ExecutionEnvironment(
            catalog=catalog,
            memory_relations=memory_relations,
        ),
    )
    return plan


def _default_description(plan: FactPlan) -> str:
    outcome = plan.outcome
    result_projection = outcome.result_projection
    if result_projection.scalar_outputs:
        return result_projection.scalar_outputs[0].scalar_id
    if result_projection.relation_outputs:
        return result_projection.relation_outputs[0].field_id
    return "name"


def _result_output_ids(plan: FactPlan) -> tuple[str, ...]:
    outcome = plan.outcome
    binding_target_ids = tuple(
        dict.fromkeys(
            slot.id
            for slot in (
                *outcome.result_projection.relation_outputs,
                *outcome.result_projection.scalar_outputs,
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
                result_output_id="answer",
            ),
        )
    return (
        FactFulfillment(
            requested_fact_id="rf_answer",
            answer_output_id="answer",
            result_output_id="answer",
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
    if not isinstance(outcome, AnswerProgram):
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
            predicate=Predicate(left=FieldRef("invalid"), operator=""),
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
                left=FieldRef("field.quantity"),
                operator=PredicateOperator.LTE,
                right=FieldRef("field.other"),
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
        spec=RankSpec(
            input_relation="totals",
            order_by=(),
            tie_policy="",
            limit=_rank_limit(1),
        ),
        output_relation="result",
    )
    valid = Operation(
        id="ranked",
        spec=RankSpec(
            input_relation="totals",
            order_by=(SortKey(field="total", direction=SortDirection.DESC),),
            tie_policy=TiePolicy.FIELD,
            tie_breakers=(SortKey(field="name", direction=SortDirection.ASC),),
            limit=_rank_limit(1),
        ),
        output_relation="result",
    )

    with pytest.raises(VerificationError, match="tie policy"):
        verify_fact_plan(_plan_with(invalid))
    verify_fact_plan(_plan_with(valid))


def test_compute_references_declared_value_origins_only():
    valid = Operation(
        id="remaining",
        spec=ComputeSpec(
            expression=_subtract(
                _number_ref(
                    input_id="target",
                    value="100",
                    proof_refs=("known_input:target_value",),
                ),
                _number_ref(
                    input_id="total",
                    value="40",
                    proof_refs=("prior:total_value",),
                ),
            ),
            output_scalar="remaining",
        ),
    )

    verify_fact_plan(
        FactPlan(
            outcome=AnswerProgram(
                fulfillment=(
                    FactFulfillment(
                        requested_fact_id="rf_answer",
                        answer_output_id="answer",
                        result_output_id="answer",
                    ),
                    FactFulfillment(
                        requested_fact_id="rf_answer",
                        answer_output_id="remaining",
                        result_output_id="remaining",
                    ),
                ),
                relations=(_relation("rows"),),
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
                result_projection=ResultProjection(
                    relation_outputs=(
                        RelationResultOutput(
                            id="answer",
                            relation_id="result",
                            field_id="name",
                        ),
                    ),
                    scalar_outputs=(
                        ScalarResultOutput(
                            id="remaining",
                            scalar_id="remaining",
                        ),
                    ),
                ),
            )
        ),
    )


def test_one_answer_output_can_be_fulfilled_by_multiple_distinct_result_outputs():
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="day_1_amount",
                ),
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="day_2_amount",
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
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="day_1_amount",
                        relation_id="day_1_result",
                        field_id="amount",
                    ),
                    RelationResultOutput(
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
                        role="ANSWER_VALUE",
                        description="Amounts shown separately by day.",
                    ),
                ),
            ),
        )
    )

    assert verify_fact_plan(plan, question_contract=question_contract) is plan


def test_compute_only_literal_answer_requires_evidence_proof():
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            operations=(
                Operation(
                    id="remaining",
                    spec=ComputeSpec(
                        expression=_subtract(
                            _number_ref(input_id="target", value="100"),
                            _number_ref(input_id="total", value="40"),
                        ),
                        output_scalar="remaining",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(),
                scalar_outputs=(
                    ScalarResultOutput(id="answer", scalar_id="remaining"),
                ),
            ),
        )
    )

    with pytest.raises(VerificationError, match="evidence proof"):
        verify_fact_plan(plan)


def test_scalar_only_terminal_answer_cannot_launder_unrelated_relation_evidence():
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(_relation("rows"),),
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
                        expression=_subtract(
                            _number_ref(input_id="target", value="100"),
                            _number_ref(input_id="total", value="40"),
                        ),
                        output_scalar="remaining",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(),
                scalar_outputs=(
                    ScalarResultOutput(id="answer", scalar_id="remaining"),
                ),
            ),
        )
    )

    with pytest.raises(VerificationError, match="terminal relation output"):
        verify_fact_plan(plan)


def test_unprojected_compute_outputs_are_not_legal_answer_work():
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(_relation("rows"),),
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
                        expression=_number_ref(input_id="target", value="100"),
                        output_scalar="unused_total",
                    ),
                ),
            ),
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="unprojected scalar output"):
        verify_fact_plan(plan)


def test_result_output_ids_must_be_unique():
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
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
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="name"
                    ),
                    RelationResultOutput(
                        id="answer", relation_id="result", field_id="total"
                    ),
                )
            ),
        )
    )

    with pytest.raises(VerificationError, match="duplicate result output"):
        verify_fact_plan(plan)


def test_relation_answer_requires_result_outputs():
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
    assert isinstance(plan.outcome, AnswerProgram)
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=plan.outcome.relations,
            operations=plan.outcome.operations,
            result_projection=ResultProjection(relation_outputs=()),
        )
    )

    with pytest.raises(VerificationError, match="result output"):
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


def test_predicate_requires_rhs_for_binary_operators():
    missing_rhs = Operation(
        id="filter",
        spec=FilterSpec(
            input_relation="rows",
            predicate=Predicate(
                left=FieldRef("field.value"),
                operator=PredicateOperator.EQUALS,
            ),
        ),
        output_relation="result",
    )
    with pytest.raises(VerificationError, match="right-hand side"):
        verify_fact_plan(_plan_with(missing_rhs))


def test_predicate_rejects_rhs_for_unary_operators():
    plan = _plan_with(
        Operation(
            id="filter",
            spec=FilterSpec(
                input_relation="rows",
                predicate=Predicate(
                    left=FieldRef("field.value"),
                    operator=PredicateOperator.IS_NULL,
                    right=_number_ref(input_id="literal", value="1"),
                ),
            ),
            output_relation="result",
        )
    )

    with pytest.raises(VerificationError, match="right-hand side"):
        verify_fact_plan(plan)


def test_count_aggregate_can_fulfill_requested_answer_output():
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_answer",
                    answer_output_id="row_count",
                    result_output_id="row_count",
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
            result_projection=ResultProjection(
                relation_outputs=(
                    RelationResultOutput(
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
                limit=_rank_limit(0),
            ),
            output_relation="result",
        )
    )

    with pytest.raises(VerificationError, match="positive"):
        verify_fact_plan(plan)
