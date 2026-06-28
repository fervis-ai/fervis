"""Terminal-result semantics for deterministic relation operations."""

from __future__ import annotations

from fervis.lookup.fact_plan.operations import (
    AggregationFunction,
    AntiJoinSpec,
    FilterSpec,
    Operation,
    ProjectSpec,
    ProjectToIdentitySpec,
    RankSpec,
    UniversalConditionSpec,
)
from fervis.lookup.outcomes.model import (
    EmptyRelationKind,
    UndefinedReasonCode,
)


def empty_aggregation_undefined_reason(
    function: AggregationFunction,
) -> UndefinedReasonCode | None:
    if function == AggregationFunction.AVG:
        return UndefinedReasonCode.EMPTY_AVERAGE
    if function == AggregationFunction.MIN:
        return UndefinedReasonCode.EMPTY_MIN
    if function == AggregationFunction.MAX:
        return UndefinedReasonCode.EMPTY_MAX
    return None


def division_undefined_reason(denominator: object) -> UndefinedReasonCode | None:
    return UndefinedReasonCode.DIVISION_BY_ZERO if denominator == 0 else None


def empty_relation_kind_for_operation_spec(spec: object) -> EmptyRelationKind:
    if isinstance(spec, (AntiJoinSpec, UniversalConditionSpec)):
        return EmptyRelationKind.OPERATION_ROWS
    return EmptyRelationKind.ANSWER_ROWS


def empty_relation_kind_for_output_relation(
    operations: tuple[Operation, ...],
    relation_id: str,
) -> EmptyRelationKind:
    operations_by_output = {
        operation.output_relation: operation
        for operation in operations
        if operation.output_relation
    }
    return _empty_relation_kind_for_relation(
        operations_by_output,
        relation_id,
        visited=frozenset(),
    )


def _empty_relation_kind_for_relation(
    operations_by_output: dict[str, Operation],
    relation_id: str,
    *,
    visited: frozenset[str],
) -> EmptyRelationKind:
    if relation_id in visited:
        return EmptyRelationKind.ANSWER_ROWS
    operation = operations_by_output.get(relation_id)
    if operation is None:
        return EmptyRelationKind.ANSWER_ROWS
    direct_kind = empty_relation_kind_for_operation_spec(operation.spec)
    if direct_kind == EmptyRelationKind.OPERATION_ROWS:
        return direct_kind
    input_relation = _empty_preserving_input_relation(operation.spec)
    if not input_relation:
        return direct_kind
    return _empty_relation_kind_for_relation(
        operations_by_output,
        input_relation,
        visited=frozenset({*visited, relation_id}),
    )


def _empty_preserving_input_relation(spec: object) -> str:
    if isinstance(spec, (FilterSpec, ProjectSpec, ProjectToIdentitySpec, RankSpec)):
        return spec.input_relation
    return ""
