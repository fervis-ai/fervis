"""Scalar compute operation implementation."""

from __future__ import annotations

from decimal import Decimal
from typing_extensions import assert_never

from fervis.lookup.answer_program.operations import ComputeBinaryOperator
from fervis.lookup.outcomes.errors import UndefinedOperationError
from fervis.lookup.outcomes.operation_semantics import division_undefined_reason
from fervis.lookup.plan_execution.operation_runtime import (
    ResolvedComputeBinary,
    ResolvedComputeExpression,
    ResolvedComputeOutput,
    ResolvedComputeSpec,
    RelationEngineError,
    fold_resolved_compute_expression,
    resolved_compute_references,
)

from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.plan_execution.declared_values import declared_number


def _compute(
    spec: ResolvedComputeSpec,
    computed_outputs: dict[str, tuple[str, RuntimeValue]],
) -> RuntimeValue:
    try:
        return _eval_expression(spec.expression, computed_outputs)
    except UndefinedOperationError as exc:
        if exc.input_refs:
            raise
        raise UndefinedOperationError(
            reason_code=exc.reason_code,
            input_refs=resolved_compute_references(spec.expression).input_refs,
        ) from exc


def _eval_expression(
    expression: ResolvedComputeExpression,
    computed_outputs: dict[str, tuple[str, RuntimeValue]],
) -> Decimal:
    return fold_resolved_compute_expression(
        expression,
        value=lambda item: declared_number(item.value, "decimal"),
        output=lambda item: _output_value(item, computed_outputs),
        negation=lambda _expression, operand: -operand,
        binary=_binary_value,
    )


def _output_value(
    expression: ResolvedComputeOutput,
    computed_outputs: dict[str, tuple[str, RuntimeValue]],
) -> Decimal:
    produced = computed_outputs.get(expression.node_id)
    if produced is None or produced[0] != expression.output_id:
        raise RelationEngineError(f"unknown scalar input {expression.output_id}")
    return declared_number(produced[1], "decimal")


def _binary_value(
    expression: ResolvedComputeBinary,
    left: Decimal,
    right: Decimal,
) -> Decimal:
    operator = expression.operator
    if operator is ComputeBinaryOperator.ADD:
        return left + right
    if operator is ComputeBinaryOperator.SUBTRACT:
        return left - right
    if operator is ComputeBinaryOperator.MULTIPLY:
        return left * right
    if operator is ComputeBinaryOperator.DIVIDE:
        reason = division_undefined_reason(right)
        if reason is not None:
            raise UndefinedOperationError(reason_code=reason)
        return left / right
    assert_never(operator)
