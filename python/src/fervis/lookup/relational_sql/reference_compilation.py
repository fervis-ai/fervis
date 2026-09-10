"""Expose a verified reference query as a guarded relation in an answer plan."""

from dataclasses import dataclass, replace
from typing import Mapping, Any
from sqlglot import exp

from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.operations import (
    Operation,
    SqlQuerySpec,
    SqlRelationInput,
    SqlColumnBinding,
    SqlOutputField,
    SqlNamedInput,
)
from fervis.lookup.answer_program.values import (
    BindingSet,
    ParameterBinding,
    ConstantRef,
)
from fervis.lookup.canonical_data import EntityKeyValue
from fervis.lookup.answer_program.relations import Relation
from .acquisition import RelationView
from .compiler import CompiledQueryAnswer
from .execution import QueryValidationError


@dataclass(frozen=True)
class CompiledReference:
    program: RelationProgram
    bindings: BindingSet
    view: RelationView
    table: Mapping[str, Any]
    input_refs: tuple[str, ...]


def compile_reference_result(
    answer: CompiledQueryAnswer,
    *,
    input_ref: str,
    output_types: Mapping[str, str],
    reference_id: str | None = None,
    operand: str = "",
    selected_key: EntityKeyValue | None = None,
    selection_proof_ref: str = "",
) -> CompiledReference:
    subject = next(
        (item for item in answer.question_contract.inputs if item.id == input_ref), None
    )
    if subject is None:
        raise QueryValidationError("Reference query requires a declared question input")
    if isinstance(subject.operand, tuple):
        if not operand or operand not in subject.operand:
            raise QueryValidationError(
                "Collection reference must identify its exact input member"
            )
    elif operand:
        raise QueryValidationError(
            "Scalar reference cannot declare a collection member"
        )
    projections = answer.program.result_projection.relation_outputs
    if (
        len(projections) != 1
        or not (projections[0].entity_key is not None or projections[0].record_fields)
        or answer.program.result_projection.scalar_outputs
    ):
        raise QueryValidationError("Reference query must identify one entity or observed record")
    projection = projections[0]
    key = projection.entity_key
    fields = tuple(dict.fromkeys(projection.value_field_ids))
    if not set(fields) <= set(output_types):
        raise QueryValidationError("Reference query key types are incomplete")
    relation_name = reference_id or f"reference_{input_ref}"
    query = (
        exp.select(
            *(
                exp.Column(this=exp.to_identifier(field, quoted=True))
                for field in fields
            )
        )
        .from_("matches")
    )
    if key is not None:
        query = query.distinct()
    selection_parameters = []
    if selected_key is not None:
        if key is None:
            raise QueryValidationError('Observed references cannot accept a nominal identity choice')
        from fervis.lookup.available_sources import source_value_literal
        from fervis.lookup.relation_catalog.row_sources import RowSourceValueType

        if (
            not selection_proof_ref
            or (selected_key.entity_kind, selected_key.key_id)
            != (key.entity_kind, key.key_id)
            or {c.component_id for c in selected_key.components}
            != {c.component_id for c in key.components}
        ):
            raise QueryValidationError(
                "Reference choice must preserve the complete identity authority and user proof"
            )
        for index, component in enumerate(key.components):
            name = f"selected_{index}"
            value = source_value_literal(
                value_ref=relation_name + "." + name,
                value=selected_key.component_value(component.component_id),
                declared_type=RowSourceValueType(output_types[component.field_id]),
                label=component.component_id,
                source_ref=selection_proof_ref,
                proof_refs=(selection_proof_ref,),
            )
            selection_parameters.append(
                SqlNamedInput(name, ConstantRef(value.id, "reference_choice@1", value))
            )
            query = query.where(
                exp.EQ(
                    this=exp.Column(
                        this=exp.to_identifier(component.field_id, quoted=True)
                    ),
                    expression=exp.Placeholder(this=name),
                )
            )
    guard = Operation(
        relation_name + ".guard",
        SqlQuerySpec(
            query.sql(dialect="duckdb"),
            (
                SqlRelationInput(
                    "matches",
                    projection.relation_id,
                    tuple(SqlColumnBinding(field, field) for field in fields),
                ),
            ),
            tuple(SqlOutputField(field, output_types[field]) for field in fields),
            parameters=tuple(selection_parameters),
            scalar=True,
            entity_keys=(key,) if key is not None else (),
            reference_input_ref=input_ref,
            reference_operand=operand,
        ),
        output_relation=relation_name + ".rows",
    )
    reserved = {
        name
        for operation in answer.program.operations
        for name in (operation.id, operation.output_relation)
    }
    reserved.update(relation.id for relation in answer.program.relations)
    if guard.id in reserved or guard.output_relation in reserved:
        raise QueryValidationError("Reference guard identifiers collide with its query")
    columns = ({component.component_id:component.field_id for component in key.components}
               if key is not None else dict(projection.record_fields))
    input_refs = tuple(
        dict.fromkeys(
            ref
            for fact in answer.question_contract.requested_facts
            for ref in fact.input_refs
        )
    )
    parameters = answer.program.parameters
    # Member queries and clarification subjects are compiled from this collection.
    # Replacing it requires recompilation, even when its cardinality is unchanged.
    if selected_key is not None or isinstance(subject.operand, tuple):
        from fervis.lookup.contract_codec import canonical_contract_fingerprint

        fixed = []
        for parameter in parameters:
            if parameter.input_ref:
                binding = answer.bindings.get(parameter.id)
                if binding is None:
                    raise QueryValidationError(
                        "Compiled reference requires all original input bindings"
                    )
                parameter = replace(
                    parameter,
                    fixed_value_fingerprint=canonical_contract_fingerprint(
                        binding.value.payload
                    ),
                )
            fixed.append(parameter)
        parameters = tuple(fixed)
    return CompiledReference(
        RelationProgram(
            parameters=parameters,
            relations=answer.program.relations,
            operations=(*answer.program.operations, guard),
        ),
        answer.bindings,
        RelationView(relation_name, guard.output_relation, columns),
        {
            "kind": "resolved_reference",
            "input_ref": input_ref,
            "reference_operand": operand,
            "input_refs": list(input_refs),
            "columns": {name:{"type":output_types[field],
                "description":"Canonical identity key component" if key is not None else "Observed record property",
                "nullable":key is None} for name,field in columns.items()},
            "candidate_keys": ([{"entity_kind":key.entity_kind,"key_id":key.key_id,
                "components":{component.component_id:component.component_id for component in key.components},
                "context_columns":[]}] if key is not None else []),
            "observed_record": key is None,
            "entity_references": [],
            "request_parameters": [],
            "automatic_request_parameters": [],
        },
        input_refs,
    )


def combine_reference_members(
    input_term, members: tuple[CompiledReference, ...], *, reference_id: str | None = None
) -> CompiledReference:
    """Publish an identity set only after every declared member has a guard."""
    from copy import deepcopy
    from fervis.lookup.answer_program.operations import (
        ProjectSpec,
        NamedExpression,
        UnionSpec,
    )
    from fervis.lookup.answer_program.expressions import FieldRef
    from .compiler import _merge_parameter_declarations

    if not isinstance(input_term.operand, tuple) or len(members) != len(
        input_term.operand
    ):
        raise QueryValidationError(
            "Reference collection must resolve every input member"
        )
    operands = [member.table.get("reference_operand") for member in members]
    if len(set(operands)) != len(operands) or set(operands) != set(input_term.operand):
        raise QueryValidationError(
            "Reference collection members do not match the input"
        )
    relation_name = reference_id or f"reference_{input_term.id}"
    first = members[0]
    identity = first.table["candidate_keys"]
    if any(
        member.table["input_ref"] != input_term.id
        or member.table["candidate_keys"] != identity
        or member.table["columns"] != first.table["columns"]
        for member in members
    ):
        raise QueryValidationError(
            "Reference collection members disagree on identity authority"
        )
    relations: dict[str, Relation] = {}
    operations: dict[str, Operation] = {}
    bindings: dict[str, ParameterBinding] = {}
    for member in members:
        for item in member.program.relations:
            if item.id in relations and relations[item.id] != item:
                raise QueryValidationError("Reference relation identifiers collide")
            relations[item.id] = item
        for operation in member.program.operations:
            if operation.id in operations and operations[operation.id] != operation:
                raise QueryValidationError("Reference operation identifiers collide")
            operations[operation.id] = operation
        for binding in member.bindings.bindings:
            if (
                binding.parameter_id in bindings
                and bindings[binding.parameter_id] != binding
            ):
                raise QueryValidationError("Reference input bindings conflict")
            bindings[binding.parameter_id] = binding
    projected = []
    for position, member in enumerate(members):
        name = f"{relation_name}.member_{position}"
        op = Operation(
            name,
            ProjectSpec(
                member.view.relation_id,
                tuple(
                    NamedExpression(column, FieldRef(field))
                    for column, field in member.view.columns.items()
                ),
            ),
            output_relation=name + ".rows",
        )
        if op.id in operations:
            raise QueryValidationError("Reference projection identifiers collide")
        operations[op.id] = op
        projected.append(op.output_relation)
    columns = tuple(first.view.columns)
    union = Operation(
        f"{relation_name}.union",
        UnionSpec(tuple(projected), columns, columns if identity else ()),
        output_relation=f"{relation_name}.rows",
    )
    if union.id in operations:
        raise QueryValidationError("Reference union identifiers collide")
    operations[union.id] = union
    table = deepcopy(dict(first.table))
    table["reference_operand"] = ""
    table["input_refs"] = list(
        dict.fromkeys(ref for member in members for ref in member.input_refs)
    )
    return CompiledReference(
        RelationProgram(
            parameters=_merge_parameter_declarations(
                tuple(p for member in members for p in member.program.parameters)
            ),
            relations=tuple(relations.values()),
            operations=tuple(operations.values()),
        ),
        BindingSet.from_bindings(tuple(bindings.values())),
        RelationView(
            relation_name,
            union.output_relation,
            {column: column for column in columns},
        ),
        table,
        tuple(dict.fromkeys(ref for member in members for ref in member.input_refs)),
    )
