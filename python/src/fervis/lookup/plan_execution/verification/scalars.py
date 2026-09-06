"""Scalar-input helpers for answer-program verification."""

from ._shared import Operation
from fervis.lookup.answer_program.operations import operation_expression_references


def _operation_scalar_inputs(operation: Operation) -> tuple[str, ...]:
    references = operation_expression_references(operation.spec)
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
