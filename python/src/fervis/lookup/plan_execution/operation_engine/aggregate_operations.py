"""Aggregate and rank operation implementations."""

from __future__ import annotations

from collections import OrderedDict

from fervis.lookup.plan_execution.operation_runtime import RelationEngineError
from fervis.lookup.plan_execution.relations import (
    CompletenessStatus,
    RelationRows,
    Row,
)
from fervis.lookup.outcomes.errors import IncompleteEvidenceError
from fervis.lookup.fact_plan.operations import (
    AggregateSpec,
    Operation,
    RankSpec,
    SortDirection,
    TiePolicy,
)

from .shared import (
    _Descending,
    _aggregate_value,
    _field,
    _operation_relation,
    _raise_undefined_empty_aggregation,
    _relation,
    _sort_value,
)


def _aggregate(
    operation: Operation,
    spec: AggregateSpec,
    relations: dict[str, RelationRows],
    *,
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    if input_relation.completeness.status != CompletenessStatus.COMPLETE:
        raise IncompleteEvidenceError(
            relation_id=input_relation.id,
            proof_refs=input_relation.completeness.proof_refs,
        )
    if not input_relation.rows:
        _raise_undefined_empty_aggregation(spec.aggregations)
        if spec.carry_fields:
            raise RelationEngineError("aggregate carry field requires rows")
    grouped: OrderedDict[tuple[object, ...], list[Row]] = OrderedDict()
    for row in input_relation.rows:
        key = tuple(_field(row, field) for field in spec.group_by)
        grouped.setdefault(key, []).append(row)
    if not grouped and not spec.group_by:
        grouped[()] = []

    output = []
    for key, rows in grouped.items():
        result = dict(zip(spec.group_by, key, strict=True))
        for field in spec.carry_fields:
            result[field.output or field.source] = _consistent_carry_value(
                rows,
                field.source,
            )
        for aggregation in spec.aggregations:
            result[aggregation.output_field] = _aggregate_value(aggregation, rows)
        output.append(result)
    return _operation_relation(
        operation,
        output,
        grain_keys=spec.group_by,
        inputs=(input_relation,),
        scalar_refs=operation_refs,
    )


def _consistent_carry_value(rows: list[Row], field: str) -> object:
    first = _field(rows[0], field)
    for row in rows[1:]:
        if _field(row, field) != first:
            raise RelationEngineError("conflicting aggregate carry field")
    return first


def _rank(
    operation: Operation,
    spec: RankSpec,
    relations: dict[str, RelationRows],
    *,
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    if spec.limit < 1:
        raise RelationEngineError("rank requires positive limit")
    rows = tuple(_relation(relations, spec.input_relation).rows)
    order_by = (*spec.order_by, *spec.tie_breakers)

    def key(row: Row) -> tuple[object, ...]:
        values: list[object] = []
        for sort in order_by:
            value = _sort_value(_field(row, sort.field))
            values.append(
                _Descending(value) if sort.direction == SortDirection.DESC else value
            )
        if spec.tie_policy != TiePolicy.FIELD:
            raise RelationEngineError(f"unsupported tie policy {spec.tie_policy}")
        return tuple(values)

    sorted_keyed_rows = sorted(
        ((key(row), row) for row in rows),
        key=lambda item: item[0],
    )
    _verify_rank_keys_are_unique_at_limit(sorted_keyed_rows, limit=spec.limit)
    sorted_rows = [dict(row) for _, row in sorted_keyed_rows[: spec.limit]]
    input_relation = _relation(relations, spec.input_relation)
    return _operation_relation(
        operation,
        sorted_rows,
        grain_keys=input_relation.grain_keys,
        inputs=(input_relation,),
        scalar_refs=operation_refs,
    )


def _verify_rank_keys_are_unique_at_limit(
    keyed_rows: list[tuple[tuple[object, ...], Row]],
    *,
    limit: int,
) -> None:
    for index in range(1, min(limit, len(keyed_rows))):
        if keyed_rows[index - 1][0] == keyed_rows[index][0]:
            raise RelationEngineError("rank requires unique ordering keys")
    if limit < len(keyed_rows) and keyed_rows[limit - 1][0] == keyed_rows[limit][0]:
        raise RelationEngineError("rank requires unique ordering keys")
