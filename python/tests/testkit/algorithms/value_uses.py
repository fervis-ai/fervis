from __future__ import annotations

from typing import Any

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.value_compiler import compile_value_uses
from fervis.lookup.plan_execution.relations import RelationRows
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.fact_plan.relations import (
    EndpointParamBinding,
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    RelationSourceRowFilter,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import (
    api_row_source_id,
    build_row_source_catalog,
)
from fervis.lookup.fact_plan.values import (
    FactValue,
    LiteralType,
    RankLimitUse,
    RowFilterUse,
    ScalarInputUse,
    TimeComponent,
    ValueComponent,
    ValueFilterOperator,
    ValueUse,
)
from fervis.lookup.fact_planning.value_validation import verify_value_contract

from tests.testkit.assertions import subset_mismatches
from tests.testkit.catalog import catalog_from_payload


def run_value_uses_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    catalog = catalog_from_payload(input_payload["catalog"])
    try:
        compiled = compile_value_uses(
            values=tuple(_value(item) for item in input_payload.get("values") or ()),
            value_uses=tuple(
                _value_use(item) for item in input_payload.get("value_uses") or ()
            ),
            catalog=catalog,
            relations=tuple(
                _relation(item) for item in input_payload.get("relations") or ()
            ),
            row_sources=(
                build_row_source_catalog(
                    catalog,
                    memory_relations=tuple(
                        _memory_relation(item)
                        for item in input_payload.get("memory_relations") or ()
                    ),
                )
                if input_payload.get("memory_relations")
                else None
            ),
            grounded_input_uses=tuple(
                _grounded_input_use(item)
                for item in input_payload.get("grounded_input_uses") or ()
            ),
        )
    except VerificationError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    return subset_mismatches(
        actual=_compiled_payload(compiled),
        expected_subset=payload["expect"]["result_contains"],
    )


def run_value_contract_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        verify_value_contract(
            values=tuple(_value(item) for item in input_payload.get("values") or ()),
            value_uses=tuple(
                _value_use(item) for item in input_payload.get("value_uses") or ()
            ),
        )
    except VerificationError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    return []


def _value(payload: dict[str, Any]) -> FactValue:
    kind = str(payload["kind"])
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
            proof_refs=tuple(payload.get("proof_refs") or ()),
        )
    raise ValueError(f"unsupported conformance value kind: {kind}")


def _relation(payload: dict[str, Any]) -> Relation:
    return Relation(
        id=str(payload["id"]),
        source=_relation_source(payload["source"]),
        fields=tuple(_relation_field(item) for item in payload.get("fields") or ()),
    )


def _relation_source(payload: dict[str, Any]) -> RelationSource:
    return RelationSource(
        kind=SourceKind(str(payload["kind"])),
        read_id=str(payload.get("read_id") or ""),
        memory_relation_id=str(payload.get("memory_relation_id") or ""),
        param_bindings=tuple(
            EndpointParamBinding(
                param_id=str(item["param_id"]),
                value=item["value"],
                proof_refs=tuple(str(ref) for ref in item.get("proof_refs") or ()),
            )
            for item in payload.get("param_bindings") or ()
        ),
        row_filters=tuple(
            RelationSourceRowFilter(
                field_id=str(item["field_id"]),
                operator=str(item["operator"]),
                values=tuple(str(value) for value in item.get("values") or ()),
                proof_refs=tuple(str(ref) for ref in item.get("proof_refs") or ()),
            )
            for item in payload.get("row_filters") or ()
        ),
    )


def _relation_field(payload: dict[str, Any]) -> RelationField:
    return RelationField(
        field_id=str(payload["field_id"]),
        roles=tuple(FieldBindingRole(str(role)) for role in payload.get("roles") or ()),
    )


def _value_use(payload: dict[str, Any]) -> ValueUse:
    return ValueUse(
        id=str(payload["id"]),
        value_id=str(payload["value_id"]),
        target=_value_use_target(payload["target"]),
    )


def _value_use_target(
    payload: dict[str, Any],
) -> RowFilterUse | ScalarInputUse | RankLimitUse:
    kind = str(payload["kind"])
    if kind == "row_filter":
        return RowFilterUse(
            relation_id=str(payload["relation_id"]),
            field_id=str(payload["field_id"]),
            operator=ValueFilterOperator(str(payload["operator"])),
            value_component=_value_component(
                str(payload.get("value_component") or ValueComponent.VALUE)
            ),
        )
    if kind == "scalar_input":
        return ScalarInputUse(
            operation_id=str(payload["operation_id"]),
            input_id=str(payload["input_id"]),
        )
    if kind == "rank_limit":
        return RankLimitUse(operation_id=str(payload["operation_id"]))
    raise ValueError(f"unsupported conformance value-use target: {kind}")


def _grounded_input_use(payload: dict[str, Any]) -> GroundedInputUse:
    row_source = payload["row_source"]
    return GroundedInputUse(
        id=str(payload["id"]),
        value_id=str(payload["value_id"]),
        row_source_id=api_row_source_id(
            str(row_source["read_id"]),
            str(row_source.get("row_path_id") or "root"),
        ),
        param_id=str(payload["param_id"]),
        value_component=_value_component(str(payload["value_component"])),
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
                "proof_refs": list(item.proof_refs),
            }
            for item in compiled.endpoint_args
        ],
        "row_filters": [
            {
                "relation_id": item.relation_id,
                "field_id": item.field_id,
                "operator": item.operator.value,
                "value": list(item.value) if isinstance(item.value, tuple) else item.value,
                "proof_refs": list(item.proof_refs),
            }
            for item in compiled.row_filters
        ],
        "scalar_inputs": [
            {
                "operation_id": item.operation_id,
                "input_id": item.input_id,
                "value": item.value,
            }
            for item in compiled.scalar_inputs
        ],
        "rank_limits": [
            {
                "operation_id": item.operation_id,
                "value": item.value,
                "proof_refs": list(item.proof_refs),
            }
            for item in compiled.rank_limits
        ],
    }
