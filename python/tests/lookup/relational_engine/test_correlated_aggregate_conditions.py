from dataclasses import replace
from decimal import Decimal
import pytest

from tests.lookup.relational_engine.test_scoped_compilation import employee_query
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    Comparison,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.expression_operators import ExpressionBinaryOperator
from fervis.lookup.source_binding.verification import (
    verify_source_strategy,
    VerifiedSourceStrategy,
)
from fervis.lookup.fact_compilation import compile_verified_source_strategy
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (AggregateFunction.SUM, {"m", "negative"}),
        (AggregateFunction.COUNT, {"negative"}),
        (AggregateFunction.MINIMUM, {"m"}),
        (AggregateFunction.MAXIMUM, {"m"}),
        (AggregateFunction.AVERAGE, {"m"}),
    ],
)
def test_related_aggregate_can_qualify_parent_rows_with_empty_inputs(
    function, expected
):
    original = employee_query()
    fact = original.request.index.requested_fact
    fact = replace(
        fact,
        expressions=(
            Aggregate(
                "manager_total",
                function,
                "manager_salary",
                None,
                False,
                fact.origin,
            ),
            Comparison(
                "comparison",
                ExpressionBinaryOperator.GT,
                "manager_total",
                "employee_salary",
                fact.origin,
            ),
        ),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    verified = verify_source_strategy(
        original.binding_plan, request=replace(original.request, index=index)
    )
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    rows = (
        {"id": "e", "manager_id": "m", "salary": Decimal(100)},
        {"id": "m", "manager_id": "c", "salary": Decimal(50)},
        {"id": "c", "manager_id": None, "salary": Decimal(200)},
        {"id": "negative", "manager_id": None, "salary": Decimal(-1)},
    )
    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    r.id,
                    rows,
                    ("id",),
                    {"id": "string", "manager_id": "string", "salary": "decimal"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for r in program.relations
            ),
            operations=tuple(
                ExecutableOperation(o.id, o.spec, o.output_relation)
                for o in program.operations
            ),
        )
    )
    assert result.issue is None
    output = program.result_projection.relation_outputs[0]
    assert {
        row[output.entity_key.components[0].field_id]
        for row in result.relation(output.relation_id).rows
    } == expected


@pytest.mark.parametrize("branches", [1, 2])
@pytest.mark.parametrize("shape", ["filter", "nested", "qualified_filter"])
def test_scalar_aggregate_materializes_its_complete_dependency_scope(shape, branches):
    from fervis.lookup.question_contract.model import RequestedOutput, NullCheck
    from fervis.lookup.expression_operators import ExpressionUnaryOperator
    from fervis.lookup.source_binding.model import (
        BooleanRequirementRealization,
        SourceMechanic,
        SourceMechanicKind,
    )

    original = employee_query(branches=branches)
    fact = original.request.index.requested_fact
    comparison = fact.expressions[0]
    if shape == "nested":
        expressions = (
            Aggregate(
                "inner",
                AggregateFunction.SUM,
                "manager_salary",
                None,
                False,
                fact.origin,
            ),
            Aggregate(
                "total", AggregateFunction.SUM, "inner", None, False, fact.origin
            ),
        )
    else:
        expressions = (
            comparison,
            Aggregate(
                "total",
                AggregateFunction.SUM,
                "employee_salary",
                "comparison",
                False,
                fact.origin,
            ),
        )
    qualification = None
    if shape == "qualified_filter":
        expressions += (
            NullCheck(
                "only_null",
                ExpressionUnaryOperator.IS_NULL,
                "employee_salary",
                fact.origin,
            ),
        )
        qualification = "only_null"
    fact = replace(
        fact,
        expressions=expressions,
        qualification_ref=qualification,
        outputs=(RequestedOutput("value", "total", fact.origin),),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    strategy = replace(
        original.request.strategy,
        branches=tuple(
            replace(
                branch,
                qualification_clause_refs=tuple(
                    c.clause_ref for c in index.qualification.clauses
                ),
            )
            for branch in original.request.strategy.branches
        ),
    )
    plan = replace(
        original.binding_plan,
        strategy=strategy,
        boolean_bindings={
            r.requirement_ref: tuple(
                BooleanRequirementRealization(
                    branch.branch_id,
                    (
                        SourceMechanic(
                            "Evaluate the typed condition.",
                            "employees",
                            (),
                            ("employees", r.requirement_ref),
                            SourceMechanicKind.RETURNED_ROW_PREDICATE,
                        ),
                    ),
                )
                for branch in strategy.branches
            )
            for r in index.boolean_requirements
        },
    )
    verified = verify_source_strategy(
        plan, request=replace(original.request, index=index, strategy=strategy)
    )
    assert isinstance(verified, VerifiedSourceStrategy)
    program = compile_verified_source_strategy(verified).answer_program
    from fervis.lookup.plan_execution.verification.result_projection import (
        _verify_result_output_targets,
    )

    _verify_result_output_targets(program)
    rows = (
        {"id": "e", "manager_id": "m", "salary": Decimal(100)},
        {"id": "m", "manager_id": None, "salary": Decimal(200)},
    )
    result = execute_operations(
        RelationEngineInput(
            relations=tuple(
                RelationRows(
                    r.id,
                    rows,
                    ("id",),
                    {"id": "string", "manager_id": "string", "salary": "decimal"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                )
                for r in program.relations
            ),
            operations=tuple(
                ExecutableOperation(o.id, o.spec, o.output_relation)
                for o in program.operations
            ),
        )
    )
    assert result.issue is None and result.undefined is None
    output = program.result_projection.relation_outputs[0]
    assert (
        result.relation(output.relation_id).rows[0][output.field_id]
        == {"filter": 100, "nested": 200, "qualified_filter": 0}[shape]
    )
