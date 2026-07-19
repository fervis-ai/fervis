from __future__ import annotations

from typing import Any

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.expression_instantiation import (
    instantiate_program_expressions,
)
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    DraftRelationSourceRowFilter,
    RelationInputOrigin,
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    RelationField,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import (
    build_row_source_catalog,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    LiteralType,
    TimeComponent,
    ValueComponent,
)
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.fact_planning.value_validation import verify_value_contract
from fervis.lookup.answer_program import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRef,
    ParameterRole,
    ProgramInputs,
)
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.contracts import parameter_value_type
from fervis.lookup.fact_planning.pattern_plan.parameterization import (
    ParameterizedRelation,
    compiled_program_inputs,
    parameterize_relation,
)
from fervis.lookup.answer_program.expressions import (
    FieldRef,
    expression_input_id,
    expression_references,
)
from fervis.lookup.answer_program.operations import FilterSpec
from fervis.lookup.fact_planning.value_components import value_component

from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)
from tests.testkit.catalog import catalog_from_payload


def run_value_uses_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    catalog = catalog_from_payload(input_payload["catalog"])
    values = tuple(_value(item) for item in input_payload.get("values") or ())
    parameter_ids = {value.id: f"value.{value.id}" for value in values}
    parameter_declarations = tuple(
        ParameterDeclaration(
            id=parameter_ids[value.id],
            role=ParameterRole.QUESTION_INPUT,
            value_type=parameter_value_type(value),
        )
        for value in values
    )
    binding_set = BindingSet.from_bindings(
        tuple(
            ParameterBinding(
                parameter_id=parameter_ids[value.id],
                value=value,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                    refs=value.proof_refs,
                ),
            )
            for value in values
        )
    )
    memory_relations = tuple(
        _memory_relation(item) for item in input_payload.get("memory_relations") or ()
    )
    row_sources = build_row_source_catalog(
        catalog,
        memory_relations=memory_relations,
    )
    input_context = CompilerInputContext(
        program_inputs=ProgramInputs(
            parameters=parameter_declarations,
            bindings=binding_set,
        ),
        expressions_by_value_id={
            value.id: ParameterRef(parameter_id=parameter_ids[value.id])
            for value in values
        },
        population_coverage_by_value_id={},
        value_types_by_value_id={
            value.id: parameter.value_type
            for value, parameter in zip(values, parameter_declarations, strict=True)
        },
    )
    try:
        parameters = {item.id: item for item in parameter_declarations}
        bindings = {item.parameter_id: item for item in binding_set.bindings}
        values_by_id = {value.id: value for value in values}
        relations = tuple(
            _relation(
                item,
                parameter_ids=parameter_ids,
                values_by_id=values_by_id,
                input_context=input_context,
                parameters=parameters,
                bindings=bindings,
            )
            for item in input_payload.get("relations") or ()
        )
        compiled_inputs = compiled_program_inputs(
            parameters=parameters,
            bindings=bindings,
        )
        compiled = instantiate_program_expressions(
            bindings=compiled_inputs.bindings,
            catalog=catalog,
            relations=tuple(item.relation for item in relations),
            row_sources=row_sources,
        )
    except (ValueError, VerificationError) as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return subset_mismatches(
        actual=_compiled_payload(compiled, relations=relations),
        expected_subset=payload["expect"]["result_contains"],
    )


def run_value_contract_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    values = tuple(_value(item) for item in input_payload.get("values") or ())
    try:
        verify_value_contract(
            values=values,
        )
    except VerificationError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return []


def _value(payload: dict[str, Any]) -> FactValue:
    kind = str(payload["kind"])
    if kind == "identity":
        return FactValue.identity(
            id=str(payload["id"]),
            key=entity_key_value(
                str(payload["entity_kind"]),
                str(payload["key_id"]),
                {str(payload["key_component_id"]): str(payload["value"])},
            ),
            proof_refs=tuple(payload.get("proof_refs") or ()),
        )
    if kind == "named":
        return FactValue.named(
            id=str(payload["id"]),
            text=str(payload["text"]),
            proof_refs=tuple(payload.get("proof_refs") or ()),
        )
    if kind == "literal":
        return FactValue.literal(
            id=str(payload["id"]),
            literal_type=LiteralType(str(payload["literal_type"])),
            value=str(payload["value"]),
            proof_refs=tuple(payload.get("proof_refs") or ()),
        )
    if kind == "time":
        return FactValue.time(
            id=str(payload["id"]),
            expression=str(payload["expression"]),
            intent=dict(payload.get("intent") or {}),
            resolved_start=str(payload.get("resolved_start") or ""),
            resolved_end=str(payload.get("resolved_end") or ""),
            granularity=str(payload["granularity"]),
            proof_refs=tuple(payload.get("proof_refs") or ()),
        )
    raise ValueError(f"unsupported conformance value kind: {kind}")


def _relation(
    payload: dict[str, Any],
    *,
    parameter_ids: dict[str, str],
    values_by_id: dict[str, FactValue],
    input_context: CompilerInputContext,
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
) -> ParameterizedRelation:
    relation_id = str(payload["id"])
    return parameterize_relation(
        relation_id=relation_id,
        source=_relation_source(
            payload["source"],
            relation_id=relation_id,
            parameter_ids=parameter_ids,
            values_by_id=values_by_id,
        ),
        fields=tuple(_relation_field(item) for item in payload.get("fields") or ()),
        input_context=input_context,
        parameters=parameters,
        bindings=bindings,
    )


def _relation_source(
    payload: dict[str, Any],
    *,
    relation_id: str,
    parameter_ids: dict[str, str],
    values_by_id: dict[str, FactValue],
) -> DraftRelationSource:
    read_id = str(payload.get("read_id") or "")
    return DraftRelationSource(
        kind=SourceKind(str(payload["kind"])),
        read_id=read_id,
        memory_relation_id=str(payload.get("memory_relation_id") or ""),
        param_bindings=tuple(
            _source_param_binding(
                item,
                relation_id=relation_id,
                values_by_id=values_by_id,
            )
            for item in payload.get("param_bindings") or ()
        ),
        row_filters=tuple(
            _source_row_filter(
                item,
                relation_id=relation_id,
                parameter_ids=parameter_ids,
            )
            for item in payload.get("row_filters") or ()
        ),
    )


def _source_param_binding(
    payload: dict[str, Any],
    *,
    relation_id: str,
    values_by_id: dict[str, FactValue],
) -> DraftEndpointParamBinding:
    proof_refs = tuple(str(ref) for ref in payload.get("proof_refs") or ())
    value_id = str(payload.get("value_id") or "")
    if not value_id:
        return DraftEndpointParamBinding(
            param_id=str(payload["param_id"]),
            value_expr=_constant_expression(
                payload["value"],
                ref_id=f"{relation_id}.{payload['param_id']}",
                proof_refs=proof_refs,
            ),
            proof_refs=proof_refs,
        )
    fact_value = values_by_id.get(value_id)
    if fact_value is None:
        raise ValueError("source param binding references unknown value")
    component = _value_component(
        str(payload.get("value_component") or ValueComponent.VALUE.value)
    )
    return DraftEndpointParamBinding(
        param_id=str(payload["param_id"]),
        value=value_component(fact_value, component),
        origin_kind=RelationInputOrigin.QUESTION_INPUT,
        value_id=value_id,
        value_component=component.value,
        proof_refs=proof_refs,
    )


def _relation_field(payload: dict[str, Any]) -> RelationField:
    return RelationField(
        field_id=str(payload["field_id"]),
        roles=tuple(FieldBindingRole(str(role)) for role in payload.get("roles") or ()),
    )


def _source_row_filter(
    payload: dict[str, Any],
    *,
    relation_id: str,
    parameter_ids: dict[str, str],
) -> DraftRelationSourceRowFilter:
    value_id = str(payload.get("value_id") or "")
    proof_refs = tuple(str(ref) for ref in payload.get("proof_refs") or ())
    expression = (
        ParameterRef(
            parameter_id=parameter_ids.get(value_id, f"missing.{value_id}"),
            component=str(payload.get("value_component") or "value"),
        )
        if value_id
        else _constant_expression(
            tuple(str(value) for value in payload.get("values") or ()),
            ref_id=f"{relation_id}.{payload['field_id']}",
            proof_refs=proof_refs,
        )
    )
    return DraftRelationSourceRowFilter(
        field_id=str(payload["field_id"]),
        operator=str(payload["operator"]),
        value_expr=expression,
        proof_refs=proof_refs,
    )


def _constant_expression(
    value: object,
    *,
    ref_id: str,
    proof_refs: tuple[str, ...],
) -> ConstantRef:
    if isinstance(value, tuple):
        fact_value = FactValue.string_set(
            id=f"fixture.{ref_id}",
            values=tuple(str(item) for item in value),
            proof_refs=proof_refs,
        )
    elif isinstance(value, bool):
        fact_value = FactValue.literal(
            id=f"fixture.{ref_id}",
            literal_type=LiteralType.BOOLEAN,
            value=str(value).lower(),
            proof_refs=proof_refs,
        )
    elif isinstance(value, (int, float)):
        fact_value = FactValue.literal(
            id=f"fixture.{ref_id}",
            literal_type=LiteralType.NUMBER,
            value=str(value),
            proof_refs=proof_refs,
        )
    else:
        fact_value = FactValue.named(
            id=f"fixture.{ref_id}",
            text=str(value),
            proof_refs=proof_refs,
        )
    return ConstantRef(
        constant_id=f"fixture.{ref_id}",
        version_ref="conformance-fixture@1",
        value=fact_value,
    )


def _memory_relation(payload: dict[str, Any]) -> RelationRows:
    return RelationRows(
        id=str(payload["id"]),
        rows=tuple(dict(row) for row in payload.get("rows") or ()),
        grain_keys=tuple(payload.get("grain_keys") or ()),
    )


def _value_component(value: str) -> ValueComponent | TimeComponent:
    if value in {item.value for item in TimeComponent}:
        return TimeComponent(value)
    return ValueComponent(value)


def _compiled_payload(
    compiled: Any,
    *,
    relations: tuple[ParameterizedRelation, ...],
) -> dict[str, Any]:
    return {
        "endpoint_args": [
            {
                "relation_id": item.relation_id,
                "read_id": item.read_id,
                "param_ref": item.param_ref,
                "value": item.value,
                "proof_refs": [
                    ref for ref in item.proof_refs if not ref.startswith("row_source:")
                ],
            }
            for item in compiled.endpoint_args
        ],
        "filters": [
            _filter_payload(operation.spec, output_relation=operation.output_relation)
            for relation in relations
            for operation in relation.operations
            if isinstance(operation.spec, FilterSpec)
        ],
    }


def _filter_payload(spec: FilterSpec, *, output_relation: str) -> dict[str, Any]:
    right = spec.predicate.right
    references = expression_references(right) if right is not None else None
    left = spec.predicate.left
    return {
        "input_relation": spec.input_relation,
        "output_relation": output_relation,
        "field_id": left.field_id if isinstance(left, FieldRef) else "",
        "operator": spec.predicate.operator.value,
        "value_ref": (
            expression_input_id(references.leaves[0])
            if references is not None and len(references.leaves) == 1
            else ""
        ),
        "proof_refs": list(spec.proof_refs),
    }
