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
    ResolvedComputeBinary,
    ResolvedComputeNegation,
    ResolvedComputeOutput,
    ResolvedComputeSpec,
    ResolvedComputeValue,
    ResolvedRankSpec,
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
    ComputeBinaryOperator,
    CrossJoinSpec,
    FilterSpec,
    JoinKey,
    JoinSpec,
    Predicate,
    PredicateOperator,
    ProjectField,
    ProjectSpec,
    ProjectToIdentitySpec,
    RelationRole,
    RelationRoleRef,
    RoleExpandSpec,
    RoleMapping,
    SortDirection,
    SortKey,
    TiePolicy,
    UnionSpec,
    UniversalConditionSpec,
)

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
    actual = {
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
            "identity_type": relation.identity_type,
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
        return exact_mismatches(actual=actual, expected=payload["expect"]["result_equals"])
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
    return RelationEngineInput(
        scalar_inputs=tuple(
            ScalarInput(id=str(item["id"]), value=item["value"])
            for item in payload.get("scalar_inputs") or ()
        ),
        relations=tuple(_relation(item) for item in payload.get("relations") or ()),
        operations=tuple(_operation(item) for item in payload.get("operations") or ()),
    )


def _relation(payload: dict[str, Any]) -> RelationRows:
    completeness = payload.get("completeness")
    return RelationRows(
        id=str(payload["id"]),
        rows=tuple(dict(row) for row in payload.get("rows") or ()),
        grain_keys=tuple(str(item) for item in payload.get("grain_keys") or ()),
        identity_type=str(payload.get("identity_type") or ""),
        completeness=(
            CompletenessProof(
                status=CompletenessStatus(str(completeness.get("status"))),
                source_kind=CompletenessSourceKind(
                    str(completeness.get("source_kind") or "api_read")
                ),
                set_kind=RelationSetKind(str(completeness.get("set_kind") or "universe")),
                scope_fingerprint=str(completeness.get("scope") or "scope"),
                proof_refs=tuple(str(item) for item in completeness.get("proof_refs") or ()),
            )
            if isinstance(completeness, dict)
            else CompletenessProof()
        ),
    )


def _operation(payload: dict[str, Any]) -> ExecutableOperation:
    return ExecutableOperation(
        id=str(payload["id"]),
        spec=_operation_spec(payload["spec"]),
        output_relation=str(payload["output_relation"]),
    )


def _operation_spec(payload: dict[str, Any]) -> Any:
    kind = str(payload["kind"])
    if kind == "filter":
        return FilterSpec(
            input_relation=str(payload["input_relation"]),
            predicate=_predicate(payload["predicate"]),
        )
    if kind == "project":
        return ProjectSpec(
            input_relation=str(payload["input_relation"]),
            fields=tuple(_project_field(item) for item in payload["fields"]),
        )
    if kind == "project_to_identity":
        return ProjectToIdentitySpec(
            input_relation=str(payload["input_relation"]),
            identity_fields=tuple(str(item) for item in payload["identity_fields"]),
            fields=tuple(_project_field(item) for item in payload.get("fields") or ()),
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
            identity_fields=tuple(str(item) for item in payload.get("identity_fields") or ()),
        )
    if kind == "aggregate":
        return AggregateSpec(
            input_relation=str(payload["input_relation"]),
            group_by=tuple(str(item) for item in payload.get("group_by") or ()),
            aggregations=tuple(_aggregation(item) for item in payload["aggregations"]),
            carry_fields=tuple(
                _project_field(item) for item in payload.get("carry_fields") or ()
            ),
        )
    if kind == "rank":
        return ResolvedRankSpec(
            input_relation=str(payload["input_relation"]),
            order_by=tuple(_sort_key(item) for item in payload["order_by"]),
            tie_policy=TiePolicy(str(payload.get("tie_policy") or "field")),
            limit=int(payload["limit"]),
            tie_breakers=tuple(
                _sort_key(item) for item in payload.get("tie_breakers") or ()
            ),
        )
    if kind == "compute":
        return ResolvedComputeSpec(
            expression=_compute_expression(payload["expression"]),
            output_scalar=str(payload.get("output_scalar") or ""),
        )
    raise ValueError(f"unsupported relation operation kind: {kind}")


def _compute_expression(payload: dict[str, Any]):
    if set(payload) == {"value"}:
        value = payload["value"]
        return ResolvedComputeValue(
            input_ref=str(value["input_ref"]),
            value=value["value"],
            proof_refs=tuple(str(ref) for ref in value.get("proof_refs") or ()),
        )
    if set(payload) == {"output"}:
        output = payload["output"]
        return ResolvedComputeOutput(
            node_id=str(output["node_id"]),
            output_id=str(output["output_id"]),
        )
    if set(payload) == {"negate"}:
        return ResolvedComputeNegation(
            operand=_compute_expression(payload["negate"])
        )
    if set(payload) == {"operator", "left", "right"}:
        return ResolvedComputeBinary(
            operator=ComputeBinaryOperator(str(payload["operator"])),
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


def _project_field(payload: dict[str, Any]) -> ProjectField:
    return ProjectField(
        source=str(payload["source"]),
        output=(
            str(payload["output"])
            if payload.get("output") is not None
            else None
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
    return Predicate(
        left=str(payload["left"]),
        operator=PredicateOperator(str(payload["operator"])),
        right_scalar=str(payload["right_scalar"]),
    )
