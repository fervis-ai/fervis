from decimal import Decimal
import pytest
from fervis.lookup.answer_program.expressions import (
    FieldRef,
    BinaryExpression,
    UnaryExpression,
)
from fervis.lookup.answer_program.operations import (
    ProjectSpec,
    NamedExpression,
    JoinSpec,
    JoinKey,
    JoinMode,
)
from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator as B,
    ExpressionUnaryOperator as U,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_engine.expression_evaluator import (
    ExpressionEnvironment,
    evaluate_expression,
    evaluate_condition,
)
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    CompletenessProof,
    CompletenessStatus,
)

COMPLETE = CompletenessProof(status=CompletenessStatus.COMPLETE)


def test_empty_projected_relation_keeps_schema_through_left_join():
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    "left",
                    ({"id": 1},),
                    ("id",),
                    {"id": "integer"},
                    completeness=COMPLETE,
                ),
                RelationRows(
                    "right",
                    (),
                    ("id",),
                    {"id": "integer", "value": "decimal"},
                    completeness=COMPLETE,
                ),
            ),
            operations=(
                ExecutableOperation(
                    "project",
                    ProjectSpec(
                        "right",
                        (
                            NamedExpression("r_id", FieldRef("id")),
                            NamedExpression("r_value", FieldRef("value")),
                        ),
                    ),
                    "aliased",
                ),
                ExecutableOperation(
                    "join",
                    JoinSpec(
                        "left", "aliased", (JoinKey("id", "r_id"),), JoinMode.LEFT
                    ),
                    "joined",
                ),
            ),
        )
    )
    assert result.relation("aliased").field_types == {
        "r_id": "integer",
        "r_value": "decimal",
    }
    assert result.relation("joined").rows == ({"id": 1, "r_id": None, "r_value": None},)


@pytest.mark.parametrize("rows", [(), ({"amount": Decimal(3)},)])
def test_expression_projection_schema_does_not_depend_on_cardinality(rows):
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    "r", rows, field_types={"amount": "decimal"}, completeness=COMPLETE
                ),
            ),
            operations=(
                ExecutableOperation(
                    "p",
                    ProjectSpec(
                        "r",
                        (
                            NamedExpression(
                                "twice",
                                BinaryExpression(
                                    B.ADD, FieldRef("amount"), FieldRef("amount")
                                ),
                            ),
                            NamedExpression(
                                "positive",
                                BinaryExpression(
                                    B.GT, FieldRef("amount"), FieldRef("amount")
                                ),
                            ),
                        ),
                    ),
                    "p",
                ),
            ),
        )
    )
    assert result.relation("p").field_types == {
        "twice": "decimal",
        "positive": "boolean",
    }


def test_unknown_comparison_and_its_negation_do_not_become_known_true():
    env = ExpressionEnvironment(
        row={"a": None, "b": 1}, field_types={"a": "integer", "b": "integer"}
    )
    eq = BinaryExpression(B.EQUALS, FieldRef("a"), FieldRef("b"))
    ne = BinaryExpression(B.NOT_EQUALS, FieldRef("a"), FieldRef("b"))
    assert evaluate_expression(eq, environment=env).value is None
    assert evaluate_expression(ne, environment=env).value is None
    assert (
        evaluate_expression(UnaryExpression(U.NOT, eq), environment=env).value is None
    )
    assert not evaluate_condition(UnaryExpression(U.NOT, eq), environment=env)


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected"),
    [
        (B.AND, False, None, False),
        (B.AND, True, None, None),
        (B.AND, None, None, None),
        (B.OR, True, None, True),
        (B.OR, False, None, None),
        (B.OR, None, None, None),
    ],
)
def test_nullable_boolean_algebra(operator, left, right, expected):
    env = ExpressionEnvironment(
        row={"a": left, "b": right}, field_types={"a": "boolean", "b": "boolean"}
    )
    assert (
        evaluate_expression(
            BinaryExpression(operator, FieldRef("a"), FieldRef("b")), environment=env
        ).value
        is expected
    )


def test_nullable_arithmetic_keeps_its_declared_numeric_type():
    env = ExpressionEnvironment(
        row={"a": None, "b": Decimal(2)}, field_types={"a": "decimal", "b": "decimal"}
    )
    value = evaluate_expression(
        BinaryExpression(B.ADD, FieldRef("a"), FieldRef("b")), environment=env
    )
    assert value.value is None and value.value_type == "decimal"


def test_scalar_output_types_survive_broadcast_into_an_empty_relation():
    from fervis.lookup.answer_program.operations import (
        AggregateSpec,
        AggregationSpec,
        AggregationFunction,
        ComputeSpec,
    )
    from fervis.lookup.answer_program.values import NodeOutputRef

    count = NodeOutputRef("count_operation", "count")
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    "r", (), field_types={"id": "integer"}, completeness=COMPLETE
                ),
            ),
            operations=(
                ExecutableOperation(
                    "count_operation",
                    AggregateSpec(
                        "r",
                        (),
                        (AggregationSpec(AggregationFunction.COUNT, "count", ""),),
                    ),
                    "count_rows",
                ),
                ExecutableOperation(
                    "check",
                    ComputeSpec(
                        expression=BinaryExpression(B.EQUALS, count, count),
                        output_scalar="flag",
                    ),
                ),
                ExecutableOperation(
                    "broadcast",
                    ProjectSpec(
                        "r", (NamedExpression("flag", NodeOutputRef("check", "flag")),)
                    ),
                    "broadcast_rows",
                ),
            ),
        )
    )
    assert result.scalars["flag"] is True
    assert result.scalar_types["flag"] == "boolean"
    assert result.relation("broadcast_rows").field_types == {"flag": "boolean"}


def test_key_projection_carries_consistent_values_once_per_identity():
    from fervis.lookup.answer_program.operations import ProjectToKeySpec

    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    "r",
                    (
                        {"id": "a", "value": Decimal("2.0")},
                        {"id": "a", "value": Decimal("2.00")},
                    ),
                    field_types={"id": "string", "value": "decimal"},
                    completeness=COMPLETE,
                ),
            ),
            operations=(
                ExecutableOperation(
                    "project", ProjectToKeySpec("r", ("id",), ("value",)), "unique"
                ),
            ),
        )
    )
    assert result.relation("unique").rows == ({"id": "a", "value": Decimal(2)},)
    assert result.relation("unique").grain_keys == ("id",)
    assert result.relation("unique").field_types == {"id": "string", "value": "decimal"}


def test_key_projection_rejects_conflicting_values_for_one_identity():
    from fervis.lookup.answer_program.operations import ProjectToKeySpec
    from fervis.lookup.plan_execution.errors import RelationEngineError

    with pytest.raises(RelationEngineError, match="conflicting values"):
        execute_operations(
            RelationEngineInput(
                relations=(
                    RelationRows(
                        "r",
                        ({"id": "a", "value": 2}, {"id": "a", "value": 3}),
                        field_types={"id": "string", "value": "integer"},
                        completeness=COMPLETE,
                    ),
                ),
                operations=(
                    ExecutableOperation(
                        "project", ProjectToKeySpec("r", ("id",), ("value",)), "unique"
                    ),
                ),
            )
        )
