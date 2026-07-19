"""Scalar-input helpers for fact-plan verification."""

from ._shared import (
    ComputeSpec,
    FilterSpec,
    Operation,
    ProjectSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.operations import (
    Predicate,
)
from fervis.lookup.answer_program.expressions import expression_references


def _operation_scalar_inputs(operation: Operation) -> tuple[str, ...]:
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        references = expression_references(spec.expression)
        return tuple(
            dict.fromkeys(
                (
                    *(reference.output_id for reference in references.outputs),
                    *(reference.parameter_id for reference in references.parameters),
                )
            )
        )
    if isinstance(spec, FilterSpec):
        return _predicate_scalar_inputs(spec.predicate)
    if isinstance(spec, ProjectSpec):
        project_references = tuple(
            expression_references(output.expression) for output in spec.outputs
        )
        return tuple(
            dict.fromkeys(
                (
                    *(
                        item.output_id
                        for refs in project_references
                        for item in refs.outputs
                    ),
                    *(
                        item.parameter_id
                        for refs in project_references
                        for item in refs.parameters
                    ),
                )
            )
        )
    if isinstance(spec, UniversalConditionSpec):
        return _predicate_scalar_inputs(spec.predicate)
    return ()


def _predicate_scalar_inputs(predicate: Predicate) -> tuple[str, ...]:
    references = expression_references(predicate.left)
    if predicate.right is not None:
        right = expression_references(predicate.right)
        return tuple(
            dict.fromkeys(
                (
                    *(item.output_id for item in references.outputs),
                    *(item.parameter_id for item in references.parameters),
                    *(item.output_id for item in right.outputs),
                    *(item.parameter_id for item in right.parameters),
                )
            )
        )
    return tuple(
        dict.fromkeys(
            (
                *(item.output_id for item in references.outputs),
                *(item.parameter_id for item in references.parameters),
            )
        )
    )
