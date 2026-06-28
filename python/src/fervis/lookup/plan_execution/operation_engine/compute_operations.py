"""Scalar compute operation implementation."""

from __future__ import annotations

import ast
from decimal import Decimal

from fervis.lookup.outcomes.errors import UndefinedOperationError
from fervis.lookup.outcomes.operation_semantics import division_undefined_reason
from fervis.lookup.plan_execution.operation_runtime import RelationEngineError
from fervis.lookup.fact_plan.operations import ComputeSpec

from .shared import _number


def _compute(spec: ComputeSpec, scalars: dict[str, object]) -> object:
    missing = [item for item in spec.scalar_inputs if item not in scalars]
    if missing:
        raise RelationEngineError(f"unknown scalar input {missing[0]}")
    expression = ast.parse(spec.expression, mode="eval")
    declared_scalars = {name: scalars[name] for name in spec.scalar_inputs}
    try:
        return _eval_expression(expression.body, declared_scalars)
    except UndefinedOperationError as exc:
        if exc.input_refs:
            raise
        raise UndefinedOperationError(
            reason_code=exc.reason_code,
            input_refs=spec.scalar_inputs,
        ) from exc


def _eval_expression(node: ast.AST, scalars: dict[str, object]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_expression(node.body, scalars)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _number(node.value)
    if isinstance(node, ast.Name):
        if node.id not in scalars:
            raise RelationEngineError(f"undeclared scalar input {node.id}")
        return _number(scalars[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_expression(node.operand, scalars)
    if isinstance(node, ast.BinOp):
        left = _eval_expression(node.left, scalars)
        right = _eval_expression(node.right, scalars)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            reason = division_undefined_reason(right)
            if reason is not None:
                raise UndefinedOperationError(reason_code=reason)
            return left / right
    raise RelationEngineError("unsupported compute expression")
