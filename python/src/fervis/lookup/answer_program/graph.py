"""Reachability in the immutable read and computation graph."""

from dataclasses import replace

from .model import AnswerProgram
from .expressions import expression_references
from .inputs import _operation_value_expressions


def prune_unused_program_nodes(program: AnswerProgram) -> AnswerProgram:
    """Keep precisely the read/computation nodes contributing to result outputs."""
    operations = {operation.id: operation for operation in program.operations}
    relation_producers = {
        operation.output_relation: operation.id
        for operation in program.operations
        if operation.output_relation
    }
    needed_relations: set[str] = set()
    needed_operations: set[str] = set()

    def relation(ref: str) -> None:
        needed_relations.add(ref)
        if ref in relation_producers:
            operation(relation_producers[ref])

    def operation(ref: str) -> None:
        if ref in needed_operations:
            return
        needed_operations.add(ref)
        node = operations[ref]
        if node.output_relation:
            needed_relations.add(node.output_relation)
        for input_ref in node.input_relation_ids:
            relation(input_ref)
        for value in _operation_value_expressions(node):
            for output in expression_references(value.expression).outputs:
                operation(output.node_id)

    for output in program.result_projection.relation_outputs:
        relation(output.relation_id)
    scalars = {output.scalar_id for output in program.result_projection.scalar_outputs}
    for node in program.operations:
        if node.output_scalar in scalars:
            operation(node.id)
    return replace(
        program,
        relations=tuple(
            node for node in program.relations if node.id in needed_relations
        ),
        operations=tuple(
            node for node in program.operations if node.id in needed_operations
        ),
        relation_guarantees=tuple(
            value
            for value in program.relation_guarantees
            if value.relation_id in needed_relations
        ),
    )
