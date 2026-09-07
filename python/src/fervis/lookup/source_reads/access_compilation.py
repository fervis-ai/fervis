"""Expand logical source reads into explicit, correlated physical prerequisites."""

from dataclasses import replace
from typing import TypeVar
from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.relations import (
    Relation,
    RelationSource,
    RelationField,
    FieldBindingRole,
    SourceKind,
    EndpointParamBinding,
)
from fervis.lookup.answer_program.operations import Operation, FilterSpec
from fervis.lookup.answer_program.expressions import (
    FieldRef,
    BinaryExpression,
    ExpressionBinaryOperator,
)
from fervis.lookup.answer_program.values import EnvironmentRef, ConstantRef
from fervis.lookup.answer_program.compilation import close_catalog_defaults
from fervis.lookup.relation_catalog.row_sources import RowSourceCatalog
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.relation_catalog.model import requires_caller_supplied_input
from .access_model import ReadAccessCatalog

_Program = TypeVar("_Program", bound=RelationProgram)


def expand_read_access(program: _Program, access: ReadAccessCatalog) -> _Program:
    access.validate()
    known = {source.id: source for source in access.sources}
    original_sources = {relation.id: relation for relation in program.relations}
    completed: dict[str, Relation] = {}
    inserted_operations = {}
    visiting = set()
    reserved = set(original_sources) | {
        name for op in program.operations for name in (op.id, op.output_relation)
    }

    def unique(base):
        result = base
        ordinal = 1
        while result in reserved or result + ".operation" in reserved:
            result = base + "." + str(ordinal)
            ordinal += 1
        reserved.update((result, result + ".operation"))
        return result

    def expand(relation):
        if relation.id in completed:
            return completed[relation.id]
        source = known.get(relation.source.row_source_id)
        if (
            source is None
            or relation.source.kind is not SourceKind.API_READ
            or relation.source.argument_relation_id
        ):
            completed[relation.id] = relation
            return relation
        relation = close_catalog_defaults(
            RelationProgram(relations=(relation,)),
            row_sources=RowSourceCatalog((source,)),
        ).relations[0]
        supplied = {
            binding.param_id: binding for binding in relation.source.param_bindings
        }
        missing = {
            param.param_ref
            for param in source.params
            if requires_caller_supplied_input(param) and param.id not in supplied
        }
        if not missing:
            completed[relation.id] = relation
            return relation
        if source.id in visiting:
            raise VerificationError("access prerequisites have no callable entry point")
        dependency = next(
            (
                item
                for item in access.alternatives(source.id)
                if missing <= {arg.parameter_ref for arg in item.arguments}
                and access.can_enumerate(access.source(item.parent_source_ref))
            ),
            None,
        )
        if dependency is None:
            raise VerificationError("source has no complete prerequisite traversal")
        visiting.add(source.id)
        parent = access.source(dependency.parent_source_ref)
        parent_fields = {
            field.field_ref: field
            for field in (*parent.fields, *parent.request_argument_fields)
        }
        mapped = {
            arg.parameter_ref: parent_fields[arg.parent_field_ref]
            for arg in dependency.arguments
        }
        needed = {field.id: field for field in mapped.values()}
        parent_id = unique(relation.id + ".access." + parent.id)
        fields = tuple(
            RelationField(
                field.id,
                (
                    FieldBindingRole.OUTPUT
                    if FieldBindingRole.OUTPUT in field.allowed_roles
                    else field.allowed_roles[0],
                ),
            )
            for field in needed.values()
        )
        parent_relation = Relation(
            parent_id,
            RelationSource(
                SourceKind.API_READ,
                read_id=parent.read_id,
                row_source_id=parent.id,
                proof_refs=dependency.evidence_refs,
            ),
            fields,
        )
        expand(parent_relation)
        parent_output = parent_id
        conditions = []
        dynamic = []
        for param in source.params:
            if param.param_ref not in mapped:
                continue
            field = mapped[param.param_ref]
            if param.id in supplied:
                expression = supplied[param.id].value_expr
                if isinstance(expression, EnvironmentRef):
                    from fervis.lookup.available_sources import source_value_literal

                    value = source_value_literal(
                        value_ref=f"access_default:{source.id}:{param.id}",
                        value=param.default,
                        declared_type=param.type,
                        label=param.name,
                        source_ref=source.id,
                        proof_refs=(param.param_ref,),
                    )
                    expression = ConstantRef(value.id, "catalog_param_default", value)
                conditions.append(
                    BinaryExpression(
                        ExpressionBinaryOperator.EQUALS, FieldRef(field.id), expression
                    )
                )
            else:
                dynamic.append(
                    EndpointParamBinding(
                        param.id, FieldRef(field.id), dependency.evidence_refs
                    )
                )
        if conditions:
            condition = conditions[0]
            for other in conditions[1:]:
                condition = BinaryExpression(
                    ExpressionBinaryOperator.AND, condition, other
                )
            parent_output = unique(parent_id + ".matching_arguments")
            inserted_operations[relation.id] = Operation(
                parent_output + ".operation",
                FilterSpec(parent_id, condition),
                output_relation=parent_output,
            )
        completed[relation.id] = replace(
            relation,
            source=replace(
                relation.source,
                argument_relation_id=parent_output,
                param_bindings=(*relation.source.param_bindings, *dynamic),
            ),
        )
        visiting.remove(source.id)
        return completed[relation.id]

    # Existing operation order is preserved; prerequisite filters precede their consumers.
    operations = []
    emitted = set()

    def ensure(ref):
        relation = original_sources.get(ref)
        if relation is None:
            return
        expand(relation)
        if ref in inserted_operations and ref not in emitted:
            operations.append(inserted_operations[ref])
            emitted.add(ref)

    for operation in program.operations:
        for ref in operation.input_relation_ids:
            ensure(ref)
        operations.append(operation)
    for relation in program.relations:
        ensure(relation.id)
    return replace(
        program, relations=tuple(completed.values()), operations=tuple(operations)
    )
