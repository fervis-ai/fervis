"""Expose compiled candidate records as guarded references without query syntax."""

from dataclasses import dataclass, replace
from typing import Mapping, Any

from fervis.lookup.answer_program.model import AnswerProgram, RelationProgram
from fervis.lookup.answer_program.expressions import (
    FieldRef,
    BinaryExpression,
    ExpressionBinaryOperator,
    UnaryExpression,
    ExpressionUnaryOperator,
    Expression,
)
from fervis.lookup.answer_program.operations import (
    Operation,
    ReferenceGuardSpec,
    FilterSpec,
)
from fervis.lookup.answer_program.values import (
    BindingSet,
    ParameterBinding,
    ConstantRef,
)
from fervis.lookup.canonical_data import EntityKeyValue
from fervis.lookup.identity_types import ObservedReferenceValue
from fervis.lookup.answer_program.relations import Relation
from fervis.lookup.answer_program.result_projection import EntityKeyProjection
from .relation_views import RelationView


@dataclass(frozen=True)
class CompiledReference:
    program: RelationProgram
    bindings: BindingSet
    view: RelationView
    table: Mapping[str, Any]
    input_refs: tuple[str, ...]


def selected_reference_filter(
    *, relation_id: str, namespace: str, projection: EntityKeyProjection,
    selected_key: EntityKeyValue, proof_ref: str, field_types: Mapping[str, str],
) -> Operation:
    """Recheck a selected nominal key against current candidate rows."""
    from fervis.lookup.available_sources import source_value_literal
    from fervis.lookup.relation_catalog.row_sources import RowSourceValueType

    if (
        not proof_ref
        or (selected_key.entity_kind, selected_key.key_id)
        != (projection.entity_kind, projection.key_id)
        or {item.component_id for item in selected_key.components}
        != {item.component_id for item in projection.components}
    ):
        raise ValueError("Reference choice must preserve complete identity authority and user proof")
    conditions: list[Expression] = []
    for index, component in enumerate(projection.components):
        value = source_value_literal(
            value_ref=f"{namespace}.selected_{index}",
            value=selected_key.component_value(component.component_id),
            declared_type=RowSourceValueType(field_types[component.field_id]),
            label=component.component_id,
            source_ref=proof_ref,
            proof_refs=(proof_ref,),
        )
        conditions.append(BinaryExpression(
            ExpressionBinaryOperator.EQUALS, FieldRef(component.field_id),
            ConstantRef(value.id, "reference_choice@1", value),
        ))
    condition = conditions[0]
    for other in conditions[1:]:
        condition = BinaryExpression(ExpressionBinaryOperator.AND, condition, other)
    return Operation(
        f"{namespace}.choice",
        FilterSpec(relation_id, condition, proof_refs=(proof_ref,)),
        output_relation=f"{namespace}.chosen_rows",
    )


def selected_observed_filter(
    *, relation_id: str, namespace: str, source,
    selected_source_ref: str, properties: tuple[ObservedReferenceValue, ...],
    proof_ref: str, execution_field_ids: Mapping[str, str],
) -> Operation:
    """Recheck selected observed properties without promoting them to a key."""
    from fervis.lookup.available_sources import source_value_literal

    if not proof_ref or source.id != selected_source_ref or not properties:
        raise ValueError("Observed reference selection lacks current carrier authority")
    fields = {field.field_ref: field for field in source.fields}
    conditions: list[Expression] = []
    for index, property_value in enumerate(properties):
        field = fields.get(property_value.field_ref)
        if field is None or field.type.value != property_value.type_name:
            raise ValueError("Observed reference property changed on replay")
        field_id = execution_field_ids.get(property_value.field_ref)
        if field_id is None:
            raise ValueError("Observed reference property is absent from candidate rows")
        condition: Expression
        if property_value.value is None:
            condition = UnaryExpression(
                ExpressionUnaryOperator.IS_NULL, FieldRef(field_id)
            )
        else:
            value = source_value_literal(
                value_ref=f"{namespace}.observed_choice_{index}",
                value=property_value.value, declared_type=field.type,
                label=property_value.label, source_ref=source.id,
                proof_refs=(proof_ref,),
            )
            condition = BinaryExpression(
                ExpressionBinaryOperator.EQUALS, FieldRef(field_id),
                ConstantRef(value.id, "observed_reference_choice@1", value),
            )
        conditions.append(condition)
    combined: Expression = conditions[0]
    for other in conditions[1:]:
        combined = BinaryExpression(ExpressionBinaryOperator.AND, combined, other)
    return Operation(
        f"{namespace}.choice",
        FilterSpec(relation_id, combined, proof_refs=(proof_ref,)),
        output_relation=f"{namespace}.chosen_rows",
    )


def compile_reference_result(
    program: AnswerProgram,
    *,
    bindings: BindingSet,
    input_ref: str,
    output_types: Mapping[str, str],
    reference_id: str | None = None,
    operand: str = "",
    selected_key: EntityKeyValue | None = None,
    selection_proof_ref: str = "",
) -> CompiledReference:
    subject = next((item for item in program.inputs if item.id == input_ref), None)
    if subject is None:
        raise ValueError("Reference query requires a declared question input")
    if isinstance(subject.operand, tuple):
        if not operand or operand not in subject.operand:
            raise ValueError(
                "Collection reference must identify its exact input member"
            )
    elif operand:
        raise ValueError("Scalar reference cannot declare a collection member")
    projections = program.result_projection.relation_outputs
    if (
        len(projections) != 1
        or not (projections[0].entity_key is not None or projections[0].record_fields)
        or program.result_projection.scalar_outputs
    ):
        raise ValueError("Reference query must identify one entity or observed record")
    projection = projections[0]
    key = projection.entity_key
    fields = tuple(dict.fromkeys(projection.value_field_ids))
    if not set(fields) <= set(output_types):
        raise ValueError("Reference query key types are incomplete")
    relation_name = reference_id or f"reference_{input_ref}"
    selected_relation = projection.relation_id
    selection_operations = []
    if selected_key is not None:
        if key is None:
            raise ValueError(
                "Observed references cannot accept a nominal identity choice"
            )
        selection = selected_reference_filter(
            relation_id=selected_relation, namespace=relation_name,
            projection=key, selected_key=selected_key,
            proof_ref=selection_proof_ref, field_types=output_types,
        )
        selection_operations.append(selection)
        selected_relation = selection.output_relation
    guard = Operation(
        relation_name + ".guard",
        ReferenceGuardSpec(
            selected_relation, fields, input_ref, operand, entity_key=key
        ),
        output_relation=relation_name + ".rows",
    )
    reserved = {
        name
        for operation in program.operations
        for name in (operation.id, operation.output_relation)
    }
    reserved.update(relation.id for relation in program.relations)
    if any(
        item.id in reserved or item.output_relation in reserved
        for item in (*selection_operations, guard)
    ):
        raise ValueError("Reference guard identifiers collide with its query")
    columns = (
        {component.component_id: component.field_id for component in key.components}
        if key is not None
        else dict(projection.record_fields)
    )
    parameter_inputs = {
        input_ref,
        *(
            parameter.input_ref
            for parameter in program.parameters
            if parameter.input_ref
        ),
    }
    input_refs = tuple(
        term.id for term in program.inputs if term.id in parameter_inputs
    )
    parameters = program.parameters
    # Member queries and clarification subjects are compiled from this collection.
    # Replacing it requires recompilation, even when its cardinality is unchanged.
    if selected_key is not None or isinstance(subject.operand, tuple):
        from fervis.lookup.contract_codec import canonical_contract_fingerprint

        fixed = []
        for parameter in parameters:
            if parameter.input_ref:
                binding = bindings.get(parameter.id)
                if binding is None:
                    raise ValueError(
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
            relations=program.relations,
            operations=(*program.operations, *selection_operations, guard),
        ),
        bindings,
        RelationView(relation_name, guard.output_relation, columns),
        {
            "kind": "resolved_reference",
            "input_ref": input_ref,
            "reference_operand": operand,
            "input_refs": list(input_refs),
            "columns": {
                name: {
                    "type": output_types[field],
                    "description": "Canonical identity key component"
                    if key is not None
                    else "Observed record property",
                    "nullable": key is None,
                }
                for name, field in columns.items()
            },
            "candidate_keys": (
                [
                    {
                        "entity_kind": key.entity_kind,
                        "key_id": key.key_id,
                        "components": {
                            component.component_id: component.component_id
                            for component in key.components
                        },
                        "context_columns": [],
                    }
                ]
                if key is not None
                else []
            ),
            "observed_record": key is None,
            "entity_references": [],
            "request_parameters": [],
            "automatic_request_parameters": [],
        },
        input_refs,
    )


def combine_reference_members(
    input_term,
    members: tuple[CompiledReference, ...],
    *,
    reference_id: str | None = None,
) -> CompiledReference:
    """Publish an identity set only after every declared member has a guard."""
    from copy import deepcopy
    from fervis.lookup.answer_program.operations import (
        ProjectSpec,
        NamedExpression,
        UnionSpec,
    )
    from fervis.lookup.answer_program.expressions import FieldRef
    from .inputs import merge_parameter_declarations

    if not isinstance(input_term.operand, tuple) or len(members) != len(
        input_term.operand
    ):
        raise ValueError("Reference collection must resolve every input member")
    operands = [member.table.get("reference_operand") for member in members]
    if len(set(operands)) != len(operands) or set(operands) != set(input_term.operand):
        raise ValueError("Reference collection members do not match the input")
    relation_name = reference_id or f"reference_{input_term.id}"
    first = members[0]
    identity = first.table["candidate_keys"]
    if any(
        member.table["input_ref"] != input_term.id
        or member.table["candidate_keys"] != identity
        or member.table["columns"] != first.table["columns"]
        for member in members
    ):
        raise ValueError("Reference collection members disagree on identity authority")
    relations: dict[str, Relation] = {}
    operations: dict[str, Operation] = {}
    bindings: dict[str, ParameterBinding] = {}
    for member in members:
        for item in member.program.relations:
            if item.id in relations and relations[item.id] != item:
                raise ValueError("Reference relation identifiers collide")
            relations[item.id] = item
        for operation in member.program.operations:
            if operation.id in operations and operations[operation.id] != operation:
                raise ValueError("Reference operation identifiers collide")
            operations[operation.id] = operation
        for binding in member.bindings.bindings:
            if (
                binding.parameter_id in bindings
                and bindings[binding.parameter_id] != binding
            ):
                raise ValueError("Reference input bindings conflict")
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
            raise ValueError("Reference projection identifiers collide")
        operations[op.id] = op
        projected.append(op.output_relation)
    columns = tuple(first.view.columns)
    union = Operation(
        f"{relation_name}.union",
        UnionSpec(tuple(projected), columns, columns if identity else ()),
        output_relation=f"{relation_name}.rows",
    )
    if union.id in operations:
        raise ValueError("Reference union identifiers collide")
    operations[union.id] = union
    table = deepcopy(dict(first.table))
    table["reference_operand"] = ""
    table["input_refs"] = list(
        dict.fromkeys(ref for member in members for ref in member.input_refs)
    )
    return CompiledReference(
        RelationProgram(
            parameters=merge_parameter_declarations(
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
