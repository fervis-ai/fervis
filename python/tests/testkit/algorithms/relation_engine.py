from __future__ import annotations

from datetime import date
from typing import Any

from fervis.lookup.plan_execution.errors import RelationEngineError
from fervis.lookup.plan_execution.generated_relations import (
    GeneratedCalendarRelationSource,
    generate_calendar_relation,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    RelationRows,
    RelationSetKind,
)
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    AntiJoinSpec,
    ComputeSpec,
    CrossJoinSpec,
    FilterSpec,
    JoinKey,
    JoinSpec,
    KeepAll,
    OrderSpec,
    Predicate,
    PredicateOperator,
    NamedExpression,
    ProjectSpec,
    ProjectToKeySpec,
    RelationRole,
    RelationRoleRef,
    RoleExpandSpec,
    RoleMapping,
    SortDirection,
    SortKey,
    Take,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    EnvironmentRef,
    ExpressionBinaryOperator,
    ExpressionFunction,
    ExpressionUnaryOperator,
    FieldRef,
    FunctionExpression,
    UnaryExpression,
)
from fervis.lookup.answer_program.values import NodeOutputRef, ParameterRef

from tests.testkit.assertions import (
    exact_mismatches,
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def run_relation_engine_case(payload: dict[str, Any]) -> list[str]:
    try:
        output = execute_operations(engine_input_from_payload(payload["input"]))
    except RelationEngineError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    actual: dict[str, Any] = {
        relation.id: {
            "rows": list(relation.rows),
            "grain_keys": list(relation.grain_keys),
            "completeness_status": (
                relation.completeness.status.value
                if relation.completeness is not None
                else None
            ),
            "completeness_pagination": relation.completeness.pagination.value,
            "completeness_proof_refs": list(relation.completeness.proof_refs),
            "completeness_scope": relation.completeness.scope_fingerprint,
        }
        for relation in output.relations
    }
    if output.scalars:
        actual["scalars"] = dict(output.scalars)
    if output.scalar_proofs:
        actual["scalar_proofs"] = {
            scalar_id: list(proof_refs)
            for scalar_id, proof_refs in output.scalar_proofs.items()
        }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual, expected=payload["expect"]["result_equals"]
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_calendar_relation_case(payload: dict[str, Any]) -> list[str]:
    try:
        relation = generate_calendar_relation(
            GeneratedCalendarRelationSource(
                id=str(payload["input"].get("id") or "calendar"),
                start=date.fromisoformat(str(payload["input"]["start"])),
                end=date.fromisoformat(str(payload["input"]["end"])),
                output_date_field=str(
                    payload["input"].get("output_date_field") or "runtime_date"
                ),
                max_rows=int(payload["input"].get("max_rows") or 500),
            )
        )
    except RelationEngineError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    actual = {
        "rows": list(relation.rows),
        "grain_keys": list(relation.grain_keys),
        "completeness_status": (
            relation.completeness.status.value
            if relation.completeness is not None
            else None
        ),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def engine_input_from_payload(payload: dict[str, Any]) -> RelationEngineInput:
    operation_payloads = tuple(payload.get("operations") or ())
    return RelationEngineInput(
        scalar_inputs=_scalar_inputs(payload, operation_payloads=operation_payloads),
        relations=tuple(_relation(item) for item in payload.get("relations") or ()),
        operations=tuple(_operation(item) for item in operation_payloads),
        environment_values=dict(payload.get("environment_values") or {}),
        environment_types=dict(payload.get("environment_types") or {}),
    )


def _scalar_inputs(
    payload: dict[str, Any],
    *,
    operation_payloads: tuple[dict[str, Any], ...],
) -> tuple[ScalarInput, ...]:
    explicit = tuple(
        ScalarInput(
            id=f"parameter:{item['id']}",
            value=item["value"],
            value_type=str(item.get("value_type") or ""),
        )
        for item in payload.get("scalar_inputs") or ()
    )
    embedded = tuple(
        scalar
        for operation in operation_payloads
        for scalar in _expression_scalar_inputs(
            (operation.get("spec") or {}).get("expression")
        )
    )
    return tuple({item.id: item for item in (*explicit, *embedded)}.values())


def _expression_scalar_inputs(payload: object) -> tuple[ScalarInput, ...]:
    if not isinstance(payload, dict):
        return ()
    if set(payload) == {"value"}:
        value = payload["value"]
        if not isinstance(value, dict):
            return ()
        return (
            ScalarInput(
                id=f"parameter:{value['input_ref']}",
                value=value["value"],
                value_type=str(value.get("value_type") or "decimal"),
                proof_refs=tuple(str(ref) for ref in value.get("proof_refs") or ()),
            ),
        )
    return tuple(
        scalar
        for child in payload.values()
        for scalar in _expression_scalar_inputs(child)
    )


def _relation(payload: dict[str, Any]) -> RelationRows:
    completeness = payload.get("completeness")
    return RelationRows(
        id=str(payload["id"]),
        rows=tuple(dict(row) for row in payload.get("rows") or ()),
        grain_keys=tuple(str(item) for item in payload.get("grain_keys") or ()),
        field_types={
            str(field_id): str(field_type)
            for field_id, field_type in (payload.get("field_types") or {}).items()
        },
        completeness=(
            CompletenessProof(
                status=CompletenessStatus(str(completeness.get("status"))),
                source_kind=CompletenessSourceKind(
                    str(completeness.get("source_kind") or "api_read")
                ),
                set_kind=RelationSetKind(
                    str(completeness.get("set_kind") or "universe")
                ),
                scope_fingerprint=str(completeness.get("scope") or "scope"),
                proof_refs=tuple(
                    str(item) for item in completeness.get("proof_refs") or ()
                ),
            )
            if isinstance(completeness, dict)
            else CompletenessProof()
        ),
    )


def _operation(payload: dict[str, Any]) -> ExecutableOperation:
    return ExecutableOperation(
        id=str(payload["id"]),
        spec=operation_spec_from_payload(payload["spec"]),
        output_relation=str(payload["output_relation"]),
    )


def operation_spec_from_payload(payload: dict[str, Any]) -> Any:
    kind = str(payload["kind"])
    if kind == "filter":
        return FilterSpec(
            input_relation=str(payload["input_relation"]),
            predicate=_predicate(payload["predicate"]),
        )
    if kind == "project":
        return ProjectSpec(
            input_relation=str(payload["input_relation"]),
            outputs=tuple(_project_field(item) for item in payload["fields"]),
        )
    if kind == "project_to_key":
        return ProjectToKeySpec(
            input_relation=str(payload["input_relation"]),
            key_fields=tuple(str(item) for item in payload["key_fields"]),
        )
    if kind == "join":
        return JoinSpec(
            left=str(payload["left"]),
            right=str(payload["right"]),
            join_keys=tuple(_join_key(item) for item in payload["join_keys"]),
        )
    if kind == "cross_join":
        return CrossJoinSpec(left=str(payload["left"]), right=str(payload["right"]))
    if kind == "role_expand":
        return RoleExpandSpec(
            input_relation=str(payload["input_relation"]),
            carry_fields=tuple(str(item) for item in payload.get("carry_fields") or ()),
            mappings=tuple(_role_mapping(item) for item in payload["mappings"]),
            output_fields=tuple(str(item) for item in payload["output_fields"]),
            role_field=str(payload.get("role_field") or "role"),
        )
    if kind == "anti_join":
        return AntiJoinSpec(
            candidate=_role_ref(payload["candidate"]),
            observed=_role_ref(payload["observed"]),
            join_keys=tuple(_join_key(item) for item in payload["join_keys"]),
            output_fields=tuple(
                _project_field(item) for item in payload.get("output_fields") or ()
            ),
        )
    if kind == "universal_condition":
        return UniversalConditionSpec(
            candidate_subject=_role_ref(payload["candidate_subject"]),
            required_dimension=_role_ref(payload["required_dimension"]),
            observation=_role_ref(payload["observation"]),
            subject_keys=tuple(_join_key(item) for item in payload["subject_keys"]),
            dimension_keys=tuple(_join_key(item) for item in payload["dimension_keys"]),
            predicate=_predicate(payload["predicate"]),
            output_fields=tuple(
                _project_field(item) for item in payload.get("output_fields") or ()
            ),
        )
    if kind == "union":
        return UnionSpec(
            inputs=tuple(str(item) for item in payload["inputs"]),
            output_fields=tuple(str(item) for item in payload["output_fields"]),
            identity_fields=tuple(
                str(item) for item in payload.get("identity_fields") or ()
            ),
        )
    if kind == "aggregate":
        return AggregateSpec(
            input_relation=str(payload["input_relation"]),
            group_by=tuple(str(item) for item in payload.get("group_by") or ()),
            aggregations=tuple(_aggregation(item) for item in payload["aggregations"]),
        )
    if kind == "order":
        selection_payload = payload["selection"]
        selection = (
            KeepAll()
            if selection_payload["kind"] == "keep_all"
            else Take(
                limit=ParameterRef(
                    parameter_id=str(selection_payload["limit_input_id"])
                )
            )
        )
        return OrderSpec(
            input_relation=str(payload["input_relation"]),
            order_by=tuple(_sort_key(item) for item in payload["order_by"]),
            selection=selection,
            tie_breakers=tuple(
                _sort_key(item) for item in payload.get("tie_breakers") or ()
            ),
        )
    if kind == "compute":
        return ComputeSpec(
            expression=_compute_expression(payload["expression"]),
            output_scalar=str(payload.get("output_scalar") or ""),
        )
    raise ValueError(f"unsupported relation operation kind: {kind}")


def _compute_expression(payload: dict[str, Any]):
    if set(payload) == {"field"}:
        return FieldRef(str(payload["field"]))
    if set(payload) == {"environment"}:
        return EnvironmentRef(key=str(payload["environment"]))
    if set(payload) == {"function", "arguments"}:
        return FunctionExpression(
            function=ExpressionFunction(str(payload["function"])),
            arguments=tuple(_compute_expression(item) for item in payload["arguments"]),
        )
    if set(payload) == {"value"}:
        value = payload["value"]
        return ParameterRef(parameter_id=str(value["input_ref"]))
    if set(payload) == {"output"}:
        output = payload["output"]
        return NodeOutputRef(
            node_id=str(output["node_id"]),
            output_id=str(output["output_id"]),
        )
    if set(payload) == {"negate"}:
        return UnaryExpression(
            operator=ExpressionUnaryOperator.NEGATE,
            operand=_compute_expression(payload["negate"]),
        )
    if set(payload) == {"operator", "left", "right"}:
        return BinaryExpression(
            operator=ExpressionBinaryOperator(str(payload["operator"])),
            left=_compute_expression(payload["left"]),
            right=_compute_expression(payload["right"]),
        )
    raise ValueError("compute expression does not match the closed contract")


def _role_ref(payload: dict[str, Any]) -> RelationRoleRef:
    return RelationRoleRef(
        relation_id=str(payload["relation_id"]),
        role=RelationRole(str(payload["role"])),
        required_identity_fields=tuple(
            str(item) for item in payload.get("required_identity_fields") or ()
        ),
    )


def _join_key(payload: dict[str, Any]) -> JoinKey:
    return JoinKey(left=str(payload["left"]), right=str(payload["right"]))


def _project_field(payload: dict[str, Any]) -> NamedExpression:
    return NamedExpression(
        output_field=str(payload.get("output") or payload["source"]),
        expression=(
            _compute_expression(payload["expression"])
            if "expression" in payload
            else FieldRef(str(payload["source"]))
        ),
    )


def _role_mapping(payload: dict[str, Any]) -> RoleMapping:
    return RoleMapping(
        role=str(payload["role"]),
        source_field=str(payload["source_field"]),
        output_field=str(payload["output_field"]),
    )


def _aggregation(payload: dict[str, Any]) -> AggregationSpec:
    return AggregationSpec(
        function=AggregationFunction(str(payload["function"])),
        output_field=str(payload["output_field"]),
        input_field=str(payload.get("input_field") or ""),
    )


def _sort_key(payload: dict[str, Any]) -> SortKey:
    return SortKey(
        field=str(payload["field"]),
        direction=SortDirection(str(payload["direction"])),
    )


def _predicate(payload: dict[str, Any]) -> Predicate:
    from fervis.lookup.answer_program.expressions import FieldRef, ParameterRef

    right_field = str(payload.get("right") or "")
    right_scalar = str(payload.get("right_scalar") or "")
    return Predicate(
        left=FieldRef(str(payload["left"])),
        operator=PredicateOperator(str(payload["operator"])),
        right=(
            FieldRef(right_field)
            if right_field
            else ParameterRef(right_scalar)
            if right_scalar
            else None
        ),
    )
