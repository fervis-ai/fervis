"""Deterministic evaluator for the canonical answer-program expression tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing_extensions import assert_never

from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    Expression,
    ExpressionBinaryOperator,
    ExpressionFunction,
    ExpressionUnaryOperator,
    FieldRef,
    FunctionExpression,
    UnaryExpression,
    expression_input_id,
    fold_expression,
)
from fervis.lookup.answer_program.values import (
    EnvironmentRef,
    NodeOutputRef,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.outcomes.errors import UndefinedOperationError
from fervis.lookup.outcomes.operation_semantics import division_undefined_reason
from fervis.lookup.plan_execution.declared_values import declared_number
from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.plan_execution.relations import Row


@dataclass(frozen=True)
class EvaluatedExpression:
    value: RuntimeValue
    value_type: str = ""


@dataclass(frozen=True)
class ExpressionEnvironment:
    row: Row | None = None
    field_types: Mapping[str, str] | None = None
    scalars: Mapping[str, RuntimeValue] | None = None
    scalar_types: Mapping[str, str] | None = None
    computed_outputs: Mapping[str, tuple[str, RuntimeValue]] | None = None


def evaluate_expression(
    expression: Expression,
    *,
    environment: ExpressionEnvironment,
) -> EvaluatedExpression:
    """Evaluate one expression against an explicit typed environment."""

    return fold_expression(
        expression,
        field=lambda item: _field(item, environment=environment),
        parameter=lambda item: _scalar(
            expression_input_id(item),
            environment=environment,
        ),
        output=lambda item: _output(item, environment=environment),
        constant=lambda item: _scalar(
            expression_input_id(item),
            environment=environment,
        ),
        environment=lambda item: _environment(item),
        unary=_unary,
        binary=_binary,
        function=_function,
    )


def _field(
    expression: FieldRef,
    *,
    environment: ExpressionEnvironment,
) -> EvaluatedExpression:
    if environment.row is None:
        raise RelationEngineError("field expression requires row context")
    if expression.field_id not in environment.row:
        raise RelationEngineError(f"unknown expression field {expression.field_id}")
    return EvaluatedExpression(
        value=environment.row[expression.field_id],
        value_type=(environment.field_types or {}).get(expression.field_id, ""),
    )


def _scalar(
    input_id: str,
    *,
    environment: ExpressionEnvironment,
) -> EvaluatedExpression:
    scalars = environment.scalars or {}
    if input_id not in scalars:
        raise RelationEngineError(f"unknown scalar input {input_id}")
    return EvaluatedExpression(
        value=scalars[input_id],
        value_type=(environment.scalar_types or {}).get(input_id, ""),
    )


def _output(
    expression: NodeOutputRef,
    *,
    environment: ExpressionEnvironment,
) -> EvaluatedExpression:
    produced = (environment.computed_outputs or {}).get(expression.node_id)
    if produced is None or produced[0] != expression.output_id:
        raise RelationEngineError(f"unknown scalar input {expression.output_id}")
    return EvaluatedExpression(value=produced[1], value_type="decimal")


def _environment(expression: EnvironmentRef) -> EvaluatedExpression:
    raise RelationEngineError(f"unavailable expression environment {expression.key}")


def _unary(
    expression: UnaryExpression,
    operand: EvaluatedExpression,
) -> EvaluatedExpression:
    value = declared_number(operand.value, operand.value_type or "decimal")
    if expression.operator is ExpressionUnaryOperator.NEGATE:
        return EvaluatedExpression(value=-value, value_type="decimal")
    assert_never(expression.operator)


def _binary(
    expression: BinaryExpression,
    left: EvaluatedExpression,
    right: EvaluatedExpression,
) -> EvaluatedExpression:
    left_value = declared_number(left.value, left.value_type or "decimal")
    right_value = declared_number(right.value, right.value_type or "decimal")
    operator = expression.operator
    if operator is ExpressionBinaryOperator.ADD:
        value = left_value + right_value
    elif operator is ExpressionBinaryOperator.SUBTRACT:
        value = left_value - right_value
    elif operator is ExpressionBinaryOperator.MULTIPLY:
        value = left_value * right_value
    elif operator is ExpressionBinaryOperator.DIVIDE:
        reason = division_undefined_reason(right_value)
        if reason is not None:
            raise UndefinedOperationError(reason_code=reason)
        value = left_value / right_value
    else:
        assert_never(operator)
    return EvaluatedExpression(value=value, value_type="decimal")


def _function(
    expression: FunctionExpression,
    arguments: tuple[EvaluatedExpression, ...],
) -> EvaluatedExpression:
    if expression.function is ExpressionFunction.TEMPORAL_BUCKET:
        raise RelationEngineError("temporal bucket expression is not enabled")
    assert_never(expression.function)
