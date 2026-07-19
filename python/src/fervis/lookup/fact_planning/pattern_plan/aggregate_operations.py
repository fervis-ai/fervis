"""Executable operation builders for aggregate pattern compilers."""

from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationSpec,
    FilterSpec,
    Operation,
    Predicate,
    PredicateOperator,
    RankSpec,
    SortDirection,
    SortKey,
    TiePolicy,
)
from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.fact_planning.compiled_patterns import CompiledMetric, CompiledRank
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext


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


def _ranked_aggregate_operations(
    *,
    input_relation_id: str,
    aggregate_relation_id: str,
    output_relation_id: str,
    rank_operation_id: str,
    group_fields: tuple[dict[str, str], ...],
    metric: CompiledMetric,
    rank: CompiledRank,
    input_context: CompilerInputContext,
    required_group_fields: tuple[str, ...] = (),
) -> tuple[Operation, ...]:
    aggregate_output_id = metric.output_field_id
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
            id=rank_operation_id,
            spec=RankSpec(
                input_relation=aggregate_relation_id,
                order_by=(
                    SortKey(
                        field=aggregate_output_id,
                        direction=rank.direction,
                    ),
                ),
                tie_policy=TiePolicy.FIELD,
                limit=rank.limit_expression(input_context),
                tie_breakers=tuple(
                    SortKey(field=item["field_id"], direction=SortDirection.ASC)
                    for item in group_fields
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
