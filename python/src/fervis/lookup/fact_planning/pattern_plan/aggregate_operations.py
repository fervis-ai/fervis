"""Executable operation builders for aggregate pattern compilers."""

from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationSpec,
    FilterSpec,
    Operation,
    Predicate,
    PredicateOperator,
    OrderSpec,
    SortDirection,
    SortKey,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.fact_planning.compiled_patterns import CompiledMetric, CompiledOrdering


def _aggregate_operations(
    *,
    input_relation_id: str,
    output_relation_id: str,
    group_fields: tuple[dict[str, str], ...],
    metric: CompiledMetric,
    required_group_fields: tuple[str, ...] = (),
) -> tuple[Operation, ...]:
    filtered_input, filters = _not_null_group_filters(
        input_relation_id=input_relation_id,
        output_relation_id=output_relation_id,
        required_group_fields=required_group_fields,
    )
    return (
        *filters,
        Operation(
            id=f"{output_relation_id}_aggregate",
            spec=_aggregate_spec(
                input_relation_id=filtered_input,
                group_fields=group_fields,
                metric=metric,
            ),
            output_relation=output_relation_id,
        ),
    )


def _ordered_aggregate_operations(
    *,
    input_relation_id: str,
    aggregate_relation_id: str,
    output_relation_id: str,
    order_operation_id: str,
    group_fields: tuple[dict[str, str], ...],
    metric: CompiledMetric,
    ordering: CompiledOrdering,
    ordering_field_id: str,
    required_group_fields: tuple[str, ...] = (),
) -> tuple[Operation, ...]:
    filtered_input, filters = _not_null_group_filters(
        input_relation_id=input_relation_id,
        output_relation_id=aggregate_relation_id,
        required_group_fields=required_group_fields,
    )
    return (
        *filters,
        Operation(
            id=f"{aggregate_relation_id}_aggregate",
            spec=_aggregate_spec(
                input_relation_id=filtered_input,
                group_fields=group_fields,
                metric=metric,
            ),
            output_relation=aggregate_relation_id,
        ),
        Operation(
            id=order_operation_id,
            spec=OrderSpec(
                input_relation=aggregate_relation_id,
                order_by=(
                    SortKey(
                        field=ordering_field_id,
                        direction=ordering.direction,
                    ),
                ),
                selection=ordering.selection,
                tie_breakers=tuple(
                    SortKey(field=item["field_id"], direction=SortDirection.ASC)
                    for item in group_fields
                    if item["field_id"] != ordering_field_id
                ),
            ),
            output_relation=output_relation_id,
        ),
    )


def _not_null_group_filters(
    *,
    input_relation_id: str,
    output_relation_id: str,
    required_group_fields: tuple[str, ...],
) -> tuple[str, tuple[Operation, ...]]:
    current_relation = input_relation_id
    operations: list[Operation] = []
    for index, field_id in enumerate(dict.fromkeys(required_group_fields), start=1):
        next_relation = f"{output_relation_id}_nonnull_{index}"
        operations.append(
            Operation(
                id=f"{next_relation}_filter",
                spec=FilterSpec(
                    input_relation=current_relation,
                    predicate=Predicate(
                        left=FieldRef(field_id),
                        operator=PredicateOperator.NOT_NULL,
                    ),
                ),
                output_relation=next_relation,
            )
        )
        current_relation = next_relation
    return current_relation, tuple(operations)


def _aggregate_spec(
    *,
    input_relation_id: str,
    group_fields: tuple[dict[str, str], ...],
    metric: CompiledMetric,
) -> AggregateSpec:
    return AggregateSpec(
        input_relation=input_relation_id,
        group_by=tuple(item["field_id"] for item in group_fields),
        aggregations=(
            AggregationSpec(
                function=metric.function,
                output_field=metric.output_field_id,
                input_field=metric.field_id,
            ),
        ),
    )
