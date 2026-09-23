"""Lower row-valued request computations into the canonical relational engine."""

from dataclasses import replace
from typing import TypeVar
from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.relations import Relation
from fervis.lookup.answer_program.operations import (
    Operation,
    ProjectSpec,
    NamedExpression,
)
from fervis.lookup.answer_program.expressions import FieldRef, expression_references
from fervis.lookup.answer_program.dependencies import execution_schedule


_Program = TypeVar("_Program", bound=RelationProgram)


def project_request_arguments(program: _Program) -> _Program:
    operations = []
    replacements = {}
    used = {relation.id for relation in program.relations} | {
        identifier
        for operation in program.operations
        for identifier in (operation.id, operation.output_relation)
    }
    for item in execution_schedule(program, allow_source_expressions=True):
        if isinstance(item, Operation):
            operations.append(item)
            continue
        assert isinstance(item, Relation)
        dynamic = tuple(
            binding
            for binding in item.source.param_bindings
            if expression_references(binding.value_expr).fields
        )
        if not any(not isinstance(binding.value_expr, FieldRef) for binding in dynamic):
            continue
        base = item.id + ".request_arguments"
        identifier = base
        suffix = 1
        while identifier in used or identifier + ".operation" in used:
            identifier = base + "." + str(suffix)
            suffix += 1
        used.update((identifier, identifier + ".operation"))
        fields = {
            binding.param_id: "argument:" + binding.param_id for binding in dynamic
        }
        outputs = tuple(
            NamedExpression(fields[binding.param_id], binding.value_expr)
            for binding in sorted(dynamic, key=lambda value: value.param_id)
        )
        operations.append(
            Operation(
                identifier + ".operation",
                ProjectSpec(item.source.argument_relation_id, outputs),
                output_relation=identifier,
            )
        )
        replacements[item.id] = replace(
            item,
            source=replace(
                item.source,
                argument_relation_id=identifier,
                param_bindings=tuple(
                    replace(binding, value_expr=FieldRef(fields[binding.param_id]))
                    if binding.param_id in fields
                    else binding
                    for binding in item.source.param_bindings
                ),
            ),
        )
    return replace(
        program,
        relations=tuple(
            replacements.get(relation.id, relation) for relation in program.relations
        ),
        operations=tuple(operations),
    )
