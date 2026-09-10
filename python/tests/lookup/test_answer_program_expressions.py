from decimal import Decimal

import pytest

from fervis.lookup.contract_codec import (
    canonical_answer_program_payload,
    decode_answer_program,
)
from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    ExpressionBinaryOperator,
    ExpressionFunction,
    ExpressionUnaryOperator,
    FieldRef,
    FunctionExpression,
    ParameterRef,
    UnaryExpression,
    expression_references,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    ComputeSpec,
    FilterSpec,
    Operation,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    EnvironmentRef,
    FactValue,
    LiteralType,
    NodeOutputRef,
)
from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_engine.expression_evaluator import (
    ExpressionEnvironment,
    evaluate_expression,
)
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


def test_expression_tree_is_the_single_predicate_and_compute_language() -> None:
    expression = BinaryExpression(
        operator=ExpressionBinaryOperator.ADD,
        left=FieldRef(field_id="amount"),
        right=ParameterRef(parameter_id="question.adjustment"),
    )

    references = expression_references(expression)

    assert references.fields == (FieldRef(field_id="amount"),)
    assert references.parameters == (ParameterRef(parameter_id="question.adjustment"),)
    condition = BinaryExpression(
        left=FieldRef(field_id="amount"),
        operator=ExpressionBinaryOperator.GT,
        right=ParameterRef(parameter_id="question.threshold"),
    )

    assert expression_references(condition).parameters == (
        ParameterRef(parameter_id="question.threshold"),
    )


def test_every_expression_node_round_trips_through_the_answer_program_codec() -> None:
    expression = FunctionExpression(
        function=ExpressionFunction.TEMPORAL_BUCKET,
        arguments=(
            FieldRef("recorded_at"),
            ConstantRef(
                constant_id="grain",
                version_ref="test@1",
                value=FactValue.literal(
                    id="grain",
                    literal_type=LiteralType.STRING,
                    value="day",
                ),
            ),
            EnvironmentRef("timezone"),
        ),
    )
    program = AnswerProgram(
        operations=(
            Operation(
                id="expression",
                spec=ComputeSpec(expression=expression, output_scalar="value"),
            ),
            Operation(
                id="arithmetic",
                spec=ComputeSpec(
                    expression=UnaryExpression(
                        operator=ExpressionUnaryOperator.NEGATE,
                        operand=BinaryExpression(
                            operator=ExpressionBinaryOperator.ADD,
                            left=ParameterRef("question.offset"),
                            right=NodeOutputRef("prior", "amount"),
                        ),
                    ),
                    output_scalar="arithmetic",
                ),
            ),
            Operation(
                id="row_number",
                spec=ComputeSpec(
                    expression=FunctionExpression(ExpressionFunction.ROW_NUMBER, ()),
                    output_scalar="row_number",
                ),
            ),
        )
    )

    assert decode_answer_program(canonical_answer_program_payload(program)) == program


def test_expression_evaluator_uses_exact_decimal_arithmetic() -> None:
    result = evaluate_expression(
        BinaryExpression(
            operator=ExpressionBinaryOperator.DIVIDE,
            left=ParameterRef("left"),
            right=ParameterRef("right"),
        ),
        environment=ExpressionEnvironment(
            scalars={"parameter:left": Decimal("1"), "parameter:right": Decimal("4")},
            scalar_types={"parameter:left": "number", "parameter:right": "number"},
        ),
    )

    assert result.value == Decimal("0.25")
    assert result.value_type == "decimal"


def test_expression_evaluator_composes_boolean_conditions() -> None:
    result = evaluate_expression(
        BinaryExpression(
            operator=ExpressionBinaryOperator.AND,
            left=BinaryExpression(
                operator=ExpressionBinaryOperator.GT,
                left=FieldRef("amount"),
                right=ParameterRef("minimum"),
            ),
            right=UnaryExpression(
                operator=ExpressionUnaryOperator.NOT,
                operand=UnaryExpression(
                    operator=ExpressionUnaryOperator.IS_NULL,
                    operand=FieldRef("label"),
                ),
            ),
        ),
        environment=ExpressionEnvironment(
            row={"amount": Decimal("12"), "label": "ready"},
            field_types={"amount": "decimal", "label": "string"},
            scalars={"parameter:minimum": Decimal("10")},
            scalar_types={"parameter:minimum": "decimal"},
        ),
    )

    assert result.value is True
    assert result.value_type == "boolean"


@pytest.mark.parametrize(
    "operator",
    (
        ExpressionBinaryOperator.EQUALS,
        ExpressionBinaryOperator.NOT_EQUALS,
        ExpressionBinaryOperator.LT,
        ExpressionBinaryOperator.LTE,
        ExpressionBinaryOperator.GT,
        ExpressionBinaryOperator.GTE,
    ),
)
def test_comparisons_with_null_are_unknown(
    operator: ExpressionBinaryOperator,
) -> None:
    result = evaluate_expression(
        BinaryExpression(
            operator=operator,
            left=FieldRef("value"),
            right=ParameterRef("expected"),
        ),
        environment=ExpressionEnvironment(
            row={"value": None},
            field_types={"value": "decimal"},
            scalars={"parameter:expected": Decimal("1")},
            scalar_types={"parameter:expected": "decimal"},
        ),
    )

    assert result.value is None
    assert result.value_type == "boolean"


def test_field_expression_requires_row_context() -> None:
    with pytest.raises(RelationEngineError, match="row context"):
        evaluate_expression(
            FieldRef("amount"),
            environment=ExpressionEnvironment(),
        )


@pytest.mark.parametrize(
    ("condition", "scalars", "scalar_types", "expected_ids"),
    (
        (
            BinaryExpression(
                left=FieldRef("state"),
                operator=ExpressionBinaryOperator.IN,
                right=ParameterRef("states"),
            ),
            {"parameter:states": ("open",)},
            {"parameter:states": "list"},
            ("a",),
        ),
        (
            BinaryExpression(
                left=FieldRef("label"),
                operator=ExpressionBinaryOperator.CONTAINS,
                right=ParameterRef("fragment"),
            ),
            {"parameter:fragment": "pha"},
            {"parameter:fragment": "string"},
            ("a",),
        ),
        (
            UnaryExpression(
                operand=FieldRef("deleted_at"),
                operator=ExpressionUnaryOperator.IS_NULL,
            ),
            {},
            {},
            ("a",),
        ),
    ),
)
def test_filter_conditions_share_one_typed_runtime(
    condition: BinaryExpression | UnaryExpression,
    scalars: dict[str, object],
    scalar_types: dict[str, str],
    expected_ids: tuple[str, ...],
) -> None:
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    id="rows",
                    rows=(
                        {
                            "id": "a",
                            "state": "open",
                            "label": "Alpha",
                            "deleted_at": None,
                        },
                        {
                            "id": "b",
                            "state": "closed",
                            "label": "Beta",
                            "deleted_at": "2026-01-01",
                        },
                    ),
                    field_types={
                        "id": "string",
                        "state": "string",
                        "label": "string",
                        "deleted_at": "date",
                    },
                ),
            ),
            operations=(
                ExecutableOperation(
                    id="filter",
                    spec=FilterSpec(input_relation="rows", condition=condition),
                    output_relation="filtered",
                ),
            ),
            scalar_inputs=tuple(
                ScalarInput(id=key, value=value, value_type=scalar_types[key])
                for key, value in scalars.items()
            ),
        )
    )

    assert tuple(row["id"] for row in result.relation("filtered").rows) == expected_ids


def test_compute_consumes_one_scalar_output_from_a_prior_aggregate() -> None:
    result = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    id="sales",
                    rows=(
                        {"amount": Decimal("20")},
                        {"amount": Decimal("30")},
                    ),
                    field_types={"amount": "decimal"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=(
                ExecutableOperation(
                    id="sales_total",
                    spec=AggregateSpec(
                        input_relation="sales",
                        group_by=(),
                        aggregations=(
                            AggregationSpec(
                                function=AggregationFunction.SUM,
                                input_field="amount",
                                output_field="total",
                            ),
                        ),
                    ),
                    output_relation="sales_total_rows",
                ),
                ExecutableOperation(
                    id="fraction",
                    spec=ComputeSpec(
                        expression=BinaryExpression(
                            operator=ExpressionBinaryOperator.MULTIPLY,
                            left=NodeOutputRef("sales_total", "total"),
                            right=ParameterRef("fraction"),
                        ),
                        output_scalar="fraction",
                    ),
                ),
            ),
            scalar_inputs=(
                ScalarInput(
                    id="parameter:fraction",
                    value=Decimal("0.1"),
                    value_type="number",
                ),
            ),
        )
    )

    assert result.scalars["fraction"] == Decimal("5.0")
