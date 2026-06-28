"""Executable operation builders for aggregate pattern compilers."""

from fervis.lookup.fact_plan.operations import (
    AggregateSpec,
    AggregationSpec,
    Operation,
    ProjectField,
    RankSpec,
    SortDirection,
    SortKey,
    TiePolicy,
)
from fervis.lookup.fact_plan.values import RankLimitUse, ValueUse


def _aggregate_operations(
    *,
    input_relation_id: str,
    output_relation_id: str,
    group_fields: tuple[dict[str, str], ...],
    carry_fields: tuple[dict[str, str], ...],
    metric: dict[str, str],
) -> tuple[Operation, ...]:
    return (
        Operation(
            id=f"{output_relation_id}_aggregate",
            spec=_aggregate_spec(
                input_relation_id=input_relation_id,
                group_fields=group_fields,
                carry_fields=carry_fields,
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
    carry_fields: tuple[dict[str, str], ...],
    metric: dict[str, str],
    rank: dict[str, object],
) -> tuple[Operation, ...]:
    aggregate_output_id = metric["output_field_id"]
    return (
        Operation(
            id=f"{aggregate_relation_id}_aggregate",
            spec=_aggregate_spec(
                input_relation_id=input_relation_id,
                group_fields=group_fields,
                carry_fields=carry_fields,
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
                        direction=rank["sort"],
                    ),
                ),
                tie_policy=TiePolicy.FIELD,
                limit=rank["limit"],
                tie_breakers=tuple(
                    SortKey(field=item["field_id"], direction=SortDirection.ASC)
                    for item in group_fields
                ),
            ),
            output_relation=output_relation_id,
        ),
    )


def _rank_value_uses(
    *,
    rank_operation_id: str,
    rank: dict[str, object],
) -> tuple[ValueUse, ...]:
    limit_value_id = str(rank["limit_value_id"])
    if not limit_value_id:
        return ()
    return (
        ValueUse(
            id=f"{rank_operation_id}_limit",
            value_id=limit_value_id,
            target=RankLimitUse(operation_id=rank_operation_id),
        ),
    )


def _aggregate_spec(
    *,
    input_relation_id: str,
    group_fields: tuple[dict[str, str], ...],
    carry_fields: tuple[dict[str, str], ...],
    metric: dict[str, str],
) -> AggregateSpec:
    return AggregateSpec(
        input_relation=input_relation_id,
        group_by=tuple(item["field_id"] for item in group_fields),
        carry_fields=tuple(
            ProjectField(
                source=item["field_id"],
                output=item["output_field_id"],
            )
            for item in carry_fields
        ),
        aggregations=(
            AggregationSpec(
                function=metric["function"],
                output_field=metric["output_field_id"],
                input_field=metric["field_id"],
            ),
        ),
    )
