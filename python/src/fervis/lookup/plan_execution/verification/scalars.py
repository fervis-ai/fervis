"""Scalar-input helpers for fact-plan verification."""

from ._shared import ComputeSpec, FilterSpec, Operation, UniversalConditionSpec
from fervis.lookup.answer_program.operations import (
    Predicate,
    compute_expression_references,
)


def _operation_scalar_inputs(operation: Operation) -> tuple[str, ...]:
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        return tuple(
            reference.output_id
            for reference in compute_expression_references(spec.expression).outputs
        )
    if isinstance(spec, FilterSpec):
        return _predicate_scalar_inputs(spec.predicate)
    if isinstance(spec, UniversalConditionSpec):
        return _predicate_scalar_inputs(spec.predicate)
    return ()


def _predicate_scalar_inputs(predicate: Predicate) -> tuple[str, ...]:
    return (predicate.right_scalar,) if predicate.right_scalar else ()
