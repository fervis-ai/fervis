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
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import (
    build_row_source_catalog,
    row_sources_for_read_id,
)
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.source_binding.grounded_params import grounded_param_bindings
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
    compiled_program_inputs,
    parameterize_relation,
)

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
    grounded_input_uses = tuple(
        _grounded_input_use(item, row_sources=row_sources)
        for item in input_payload.get("grounded_input_uses") or ()
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
    )
    try:
        grounded_bindings_by_read_id = {
            source.read_id: grounded_param_bindings(
                available_values=values,
                available_value_uses=grounded_input_uses,
                row_source=source,
            )
            for source in row_sources.sources
            if source.read_id
        }
        parameters = {item.id: item for item in parameter_declarations}
        bindings = {item.parameter_id: item for item in binding_set.bindings}
        relations = tuple(
            _relation(
                item,
                parameter_ids=parameter_ids,
                grounded_bindings_by_read_id=grounded_bindings_by_read_id,
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
            relations=relations,
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
        actual=_compiled_payload(compiled),
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
    grounded_bindings_by_read_id: dict[str, tuple[DraftEndpointParamBinding, ...]],
    input_context: CompilerInputContext,
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
) -> Relation:
    relation_id = str(payload["id"])
    return parameterize_relation(
        relation_id=relation_id,
        source=_relation_source(
            payload["source"],
            relation_id=relation_id,
            parameter_ids=parameter_ids,
            grounded_bindings_by_read_id=grounded_bindings_by_read_id,
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
    grounded_bindings_by_read_id: dict[str, tuple[DraftEndpointParamBinding, ...]],
) -> DraftRelationSource:
    read_id = str(payload.get("read_id") or "")
    return DraftRelationSource(
        kind=SourceKind(str(payload["kind"])),
        read_id=read_id,
        memory_relation_id=str(payload.get("memory_relation_id") or ""),
        param_bindings=(
            *grounded_bindings_by_read_id.get(read_id, ()),
            *tuple(
                DraftEndpointParamBinding(
                    param_id=str(item["param_id"]),
                    value_expr=_constant_expression(
                        item["value"],
                        ref_id=f"{relation_id}.{item['param_id']}",
                        proof_refs=tuple(
                            str(ref) for ref in item.get("proof_refs") or ()
                        ),
                    ),
                    proof_refs=tuple(str(ref) for ref in item.get("proof_refs") or ()),
                )
                for item in payload.get("param_bindings") or ()
            ),
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


def _grounded_input_use(payload: dict[str, Any], *, row_sources) -> GroundedInputUse:
    source_ref = payload["row_source"]
    read_id = str(source_ref["read_id"])
    row_path_id = str(source_ref.get("row_path_id") or "root")
    candidates = tuple(
        source
        for source in row_sources_for_read_id(read_id, row_sources=row_sources)
        if (source.row_path_id or "root") == row_path_id
    )
    if len(candidates) != 1:
        raise ValueError("grounded input use requires one canonical row source")
    return GroundedInputUse(
        id=str(payload["id"]),
        value_id=str(payload["value_id"]),
        row_source_id=candidates[0].id,
        param_id=str(payload["param_id"]),
        value_component=_value_component(
            str(payload.get("value_component") or ValueComponent.VALUE.value)
        ),
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


def _compiled_payload(compiled: Any) -> dict[str, Any]:
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
        "row_filters": [
            {
                "relation_id": item.relation_id,
                "field_id": item.field_id,
                "operator": item.operator.value,
                "value": list(item.value)
                if isinstance(item.value, tuple)
                else item.value,
                "proof_refs": list(item.proof_refs),
            }
            for item in compiled.row_filters
        ],
    }
