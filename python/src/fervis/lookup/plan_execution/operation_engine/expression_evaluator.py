"""Deterministic evaluator for the canonical answer-program expression tree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import operator as operator_module
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
    expression_references,
    fold_expression,
)
from fervis.lookup.answer_program.values import (
    EnvironmentRef,
    NodeOutputRef,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.expression_operators import OperatorKind, operator_signature
from fervis.lookup.outcomes.errors import UndefinedOperationError
from fervis.lookup.outcomes.operation_semantics import division_undefined_reason
from fervis.lookup.plan_execution.declared_values import (
    declared_equal,
    declared_number,
    declared_order_pair,
    parse_declared_value,
)
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
    node_outputs: Mapping[str, Mapping[str, RuntimeValue]] | None = None
    environment_values: Mapping[str, RuntimeValue] | None = None
    environment_types: Mapping[str, str] | None = None


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
        environment=lambda item: _environment(item, environment=environment),
        unary=_unary,
        binary=_binary,
        function=_function,
    )


def evaluate_condition(
    expression: Expression,
    *,
    environment: ExpressionEnvironment,
) -> bool:
    """Evaluate one Boolean expression and reject non-Boolean results."""

    return _boolean(evaluate_expression(expression, environment=environment))


def evaluated_condition_fact(
    expression: Expression,
    *,
    environment: ExpressionEnvironment,
) -> tuple[RuntimeValue, ...]:
    """Return a condition result with the leaf observations that established it."""

    return (
        evaluate_condition(expression, environment=environment),
        *(
            evaluate_expression(leaf, environment=environment).value
            for leaf in expression_references(expression).leaves
        ),
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
    produced = (environment.node_outputs or {}).get(expression.node_id)
    if produced is None or expression.output_id not in produced:
        raise RelationEngineError(f"unknown scalar input {expression.output_id}")
    return EvaluatedExpression(value=produced[expression.output_id], value_type="decimal")


def _environment(
    expression: EnvironmentRef,
    *,
    environment: ExpressionEnvironment,
) -> EvaluatedExpression:
    values = environment.environment_values or {}
    if expression.key not in values:
        raise RelationEngineError(
            f"unavailable expression environment {expression.key}"
        )
    return EvaluatedExpression(
        value=values[expression.key],
        value_type=(environment.environment_types or {}).get(expression.key, ""),
    )


def _unary(
    expression: UnaryExpression,
    operand: EvaluatedExpression,
) -> EvaluatedExpression:
    signature = operator_signature(expression.operator)
    if signature.kind is OperatorKind.ARITHMETIC:
        value = declared_number(operand.value, operand.value_type or "decimal")
        return EvaluatedExpression(value=-value, value_type="decimal")
    if signature.kind is OperatorKind.BOOLEAN:
        return EvaluatedExpression(value=not _boolean(operand), value_type="boolean")
    if expression.operator is ExpressionUnaryOperator.IS_NULL:
        return EvaluatedExpression(value=operand.value is None, value_type="boolean")
    if expression.operator is ExpressionUnaryOperator.NOT_NULL:
        return EvaluatedExpression(value=operand.value is not None, value_type="boolean")
    raise RelationEngineError(f"unsupported unary operator {expression.operator}")


def _binary(
    expression: BinaryExpression,
    left: EvaluatedExpression,
    right: EvaluatedExpression,
) -> EvaluatedExpression:
    operator = expression.operator
    signature = operator_signature(operator)
    if signature.kind is OperatorKind.ARITHMETIC:
        return _arithmetic(operator, left, right)
    if signature.kind is OperatorKind.BOOLEAN and operator is ExpressionBinaryOperator.AND:
        value = _boolean(left) and _boolean(right)
    elif signature.kind is OperatorKind.BOOLEAN and operator is ExpressionBinaryOperator.OR:
        value = _boolean(left) or _boolean(right)
    elif signature.kind in {OperatorKind.COMPARISON, OperatorKind.MEMBERSHIP}:
        value = _comparison(operator, left, right)
    else:
        raise RelationEngineError(f"unsupported binary operator {operator}")
    return EvaluatedExpression(value=value, value_type="boolean")


def _arithmetic(
    operator: ExpressionBinaryOperator,
    left: EvaluatedExpression,
    right: EvaluatedExpression,
) -> EvaluatedExpression:
    left_value = declared_number(left.value, left.value_type or "decimal")
    right_value = declared_number(right.value, right.value_type or "decimal")
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
        raise RelationEngineError(f"unsupported arithmetic operator {operator}")
    return EvaluatedExpression(value=value, value_type="decimal")


def _boolean(value: EvaluatedExpression) -> bool:
    if not isinstance(value.value, bool):
        raise RelationEngineError("boolean expression requires boolean operand")
    return value.value


def _comparison(
    operator: ExpressionBinaryOperator,
    left: EvaluatedExpression,
    right: EvaluatedExpression,
) -> bool:
    if left.value is None or right.value is None:
        return False
    if operator is ExpressionBinaryOperator.EQUALS:
        return declared_equal(left.value, left.value_type, right.value, right.value_type)
    if operator is ExpressionBinaryOperator.NOT_EQUALS:
        return not declared_equal(
            left.value, left.value_type, right.value, right.value_type
        )
    if operator in {
        ExpressionBinaryOperator.LT,
        ExpressionBinaryOperator.LTE,
        ExpressionBinaryOperator.GT,
        ExpressionBinaryOperator.GTE,
    }:
        left_value, right_value = declared_order_pair(
            left.value,
            left.value_type,
            right.value,
            right.value_type,
        )
        comparison = {
            ExpressionBinaryOperator.LT: operator_module.lt,
            ExpressionBinaryOperator.LTE: operator_module.le,
            ExpressionBinaryOperator.GT: operator_module.gt,
            ExpressionBinaryOperator.GTE: operator_module.ge,
        }
        return bool(comparison[operator](left_value, right_value))
    if operator is ExpressionBinaryOperator.CONTAINS:
        return _contains(left.value, right.value, right.value_type)
    if operator is ExpressionBinaryOperator.IN:
        return _contains(right.value, left.value, left.value_type)
    if operator is ExpressionBinaryOperator.WITHIN:
        if not isinstance(right.value, Mapping) or set(right.value) != {"start", "end"}:
            raise RelationEngineError("within requires start and end bounds")
        left_start, start = declared_order_pair(
            left.value,
            left.value_type,
            right.value["start"],
            left.value_type,
        )
        left_end, end = declared_order_pair(
            left.value,
            left.value_type,
            right.value["end"],
            left.value_type,
        )
        return left_start >= start and left_end <= end
    raise RelationEngineError(f"unsupported comparison operator {operator}")


def _contains(left: RuntimeValue, right: RuntimeValue, right_type: str | None) -> bool:
    if isinstance(left, str):
        return isinstance(right, str) and right in left
    if isinstance(left, (tuple, list)):
        return any(declared_equal(item, None, right, right_type) for item in left)
    if isinstance(left, dict):
        return isinstance(right, str) and right in left
    raise RelationEngineError("contains requires a string, collection, or mapping")


def _function(
    expression: FunctionExpression,
    arguments: tuple[EvaluatedExpression, ...],
) -> EvaluatedExpression:
    if expression.function is ExpressionFunction.TEMPORAL_BUCKET:
        return _temporal_bucket(arguments)
    assert_never(expression.function)


def _temporal_bucket(
    arguments: tuple[EvaluatedExpression, ...],
) -> EvaluatedExpression:
    if len(arguments) != 3:
        raise RelationEngineError("temporal bucket requires value, grain, and timezone")
    raw_value, raw_grain, raw_timezone = arguments
    if not isinstance(raw_grain.value, str):
        raise RelationEngineError("temporal bucket grain must be string")
    grain = raw_grain.value.strip().casefold()
    if grain not in {"day", "week", "month", "quarter", "year"}:
        raise RelationEngineError("temporal bucket grain is unsupported")
    if not isinstance(raw_timezone.value, str) or not raw_timezone.value:
        raise RelationEngineError("temporal bucket timezone must be named")
    try:
        timezone = ZoneInfo(raw_timezone.value)
    except ZoneInfoNotFoundError as exc:
        raise RelationEngineError("temporal bucket timezone is invalid") from exc
    parsed = parse_declared_value(raw_value.value, raw_value.value_type)
    if isinstance(parsed, datetime):
        local = (
            parsed.replace(tzinfo=timezone)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone)
        ).date()
    elif isinstance(parsed, date):
        local = parsed
    else:
        raise RelationEngineError("temporal bucket value must be date or datetime")
    if grain == "day":
        bucket = local
    elif grain == "week":
        bucket = local - timedelta(days=local.weekday())
    elif grain == "month":
        bucket = local.replace(day=1)
    elif grain == "quarter":
        bucket = local.replace(month=((local.month - 1) // 3) * 3 + 1, day=1)
    else:
        bucket = local.replace(month=1, day=1)
    return EvaluatedExpression(value=bucket, value_type="date")
