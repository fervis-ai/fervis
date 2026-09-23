"""The dependency order shared by source and relational-operation contracts."""

from __future__ import annotations
from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.relations import Relation, SourceKind
from fervis.lookup.answer_program.operations import Operation
from fervis.lookup.answer_program.expressions import FieldRef, expression_references
from fervis.lookup.plan_execution.errors import VerificationError


def execution_schedule(
    program: RelationProgram, *, allow_source_expressions: bool = False
) -> tuple[Relation | Operation, ...]:
    sources = {relation.id: relation for relation in program.relations}
    available: set[str] = set()
    loading: set[str] = set()
    schedule: list[Relation | Operation] = []

    def source(relation_id: str) -> None:
        if relation_id in available:
            return
        if relation_id in loading:
            raise VerificationError("cyclic source relation dependency")
        relation = sources.get(relation_id)
        if relation is None:
            raise VerificationError(
                f"unknown or unavailable argument relation {relation_id}"
            )
        loading.add(relation_id)
        targets = [binding.param_id for binding in relation.source.param_bindings]
        if len(set(targets)) != len(targets):
            raise VerificationError("source repeats a request parameter assignment")
        if not allow_source_expressions and any(
            expression_references(binding.value_expr).fields
            and not isinstance(binding.value_expr, FieldRef)
            for binding in relation.source.param_bindings
        ):
            raise VerificationError(
                "computed request arguments require a preceding projection"
            )
        dependency = relation.source.argument_relation_id
        has_fields = any(
            expression_references(binding.value_expr).fields
            for binding in relation.source.param_bindings
        )
        if bool(dependency) != has_fields:
            raise VerificationError(
                "row-valued request arguments require exactly one argument relation"
            )
        if dependency:
            if relation.source.kind is not SourceKind.API_READ:
                raise VerificationError(
                    "dependent request arguments require an API read"
                )
            source(dependency)
        loading.remove(relation_id)
        available.add(relation_id)
        schedule.append(relation)

    for operation in program.operations:
        for input_relation in operation.input_relation_ids:
            source(input_relation)
        if operation.output_relation:
            if (
                operation.output_relation in available
                or operation.output_relation in sources
            ):
                raise VerificationError(
                    f"duplicate relation {operation.output_relation}"
                )
            available.add(operation.output_relation)
        schedule.append(operation)
    for relation in program.relations:
        source(relation.id)
    return tuple(schedule)
