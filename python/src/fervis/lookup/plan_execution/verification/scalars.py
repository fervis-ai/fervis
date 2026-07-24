"""Scalar-input helpers for answer-program verification."""

from ._shared import (
    AggregateSpec,
    ComputeSpec,
    FilterSpec,
    Operation,
    ProjectSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.expressions import expression_references


def _operation_scalar_inputs(operation: Operation) -> tuple[str, ...]:
    references = _operation_expression_references(operation)
    return tuple(
        dict.fromkeys(
            (
                *(item.output_id for refs in references for item in refs.outputs),
                *(
                    item.parameter_id
                    for refs in references
                    for item in refs.parameters
                ),
            )
        )
    )


def _operation_node_output_refs(operation: Operation):
    return tuple(
        item
        for references in _operation_expression_references(operation)
        for item in references.outputs
    )


def _operation_expression_references(operation: Operation):
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        return (expression_references(spec.expression),)
    if isinstance(spec, FilterSpec):
        return (expression_references(spec.condition),)
    if isinstance(spec, ProjectSpec):
        return tuple(
            expression_references(output.expression) for output in spec.outputs
        )
    if isinstance(spec, UniversalConditionSpec):
        return (expression_references(spec.condition),)
    if isinstance(spec, AggregateSpec):
        return tuple(
            expression_references(aggregation.filter)
            for aggregation in spec.aggregations
            if aggregation.filter is not None
        )
    return ()
