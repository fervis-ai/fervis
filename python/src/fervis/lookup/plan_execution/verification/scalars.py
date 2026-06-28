"""Scalar-input helpers for fact-plan verification."""

from ._shared import ComputeSpec, FilterSpec, Operation, UniversalConditionSpec


def _operation_scalar_inputs(operation: Operation) -> tuple[str, ...]:
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        return spec.scalar_inputs
    if isinstance(spec, FilterSpec):
        return _predicate_scalar_inputs(spec.predicate)
    if isinstance(spec, UniversalConditionSpec):
        return _predicate_scalar_inputs(spec.predicate)
    return ()


def _predicate_scalar_inputs(predicate: object) -> tuple[str, ...]:
    scalar = getattr(predicate, "right_scalar", "")
    return (scalar,) if scalar else ()
