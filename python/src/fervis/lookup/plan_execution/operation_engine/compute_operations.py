"""Scalar compute operation implementation."""

from __future__ import annotations

from fervis.lookup.answer_program.expressions import expression_references
from fervis.lookup.answer_program.operations import ComputeSpec
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.outcomes.errors import UndefinedOperationError

from .expression_evaluator import ExpressionEnvironment, evaluate_expression


def _compute(
    spec: ComputeSpec,
    computed_outputs: dict[str, tuple[str, RuntimeValue]],
    *,
    scalars: dict[str, RuntimeValue],
    scalar_types: dict[str, str],
) -> RuntimeValue:
    try:
        return evaluate_expression(
            spec.expression,
            environment=ExpressionEnvironment(
                scalars=scalars,
                scalar_types=scalar_types,
                computed_outputs=computed_outputs,
            ),
        ).value
    except UndefinedOperationError as exc:
        if exc.input_refs:
            raise
        raise UndefinedOperationError(
            reason_code=exc.reason_code,
            input_refs=tuple(
                _expression_reference_id(item)
                for item in expression_references(spec.expression).leaves
            ),
        ) from exc


def _expression_reference_id(expression: object) -> str:
    for attribute in ("parameter_id", "output_id", "constant_id", "key"):
        value = getattr(expression, attribute, "")
        if value:
            return str(value)
    return type(expression).__name__
