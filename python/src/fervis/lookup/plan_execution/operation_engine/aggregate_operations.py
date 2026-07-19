"""Aggregate and rank operation implementations."""

from __future__ import annotations

from collections import OrderedDict
from typing_extensions import assert_never

from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineError,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessStatus,
    RelationRows,
    Row,
)
from fervis.lookup.outcomes.errors import IncompleteEvidenceError
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    SortDirection,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.answer_program.operations import KeepAll, OrderSpec, Take
from fervis.lookup.plan_execution.operation_engine.expression_evaluator import (
    ExpressionEnvironment,
    evaluate_expression,
)
from fervis.lookup.plan_execution.declared_values import (
    declared_key,
    declared_order_key,
    exact_positive_integer,
)

from .shared import (
    _Descending,
    _aggregate_value,
    _field,
    _operation_relation,
    _raise_undefined_empty_aggregation,
    _relation,
)


def _aggregate(
    operation: ExecutableOperation,
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
    field_types = dict(input_relation.field_types or {})
    grouped: OrderedDict[tuple[object, ...], list[Row]] = OrderedDict()
    for row in input_relation.rows:
        key = tuple(
            declared_key(_field(row, field), field_types.get(field))
            for field in spec.group_by
        )
        grouped.setdefault(key, []).append(row)
    if not grouped and not spec.group_by:
        grouped[()] = []

    output = []
    for rows in grouped.values():
        result = (
            {field: _field(rows[0], field) for field in spec.group_by} if rows else {}
        )
        for aggregation in spec.aggregations:
            result[aggregation.output_field] = _aggregate_value(
                aggregation, rows, field_types
            )
        output.append(result)
    return _operation_relation(
        operation,
        output,
        grain_keys=spec.group_by,
        inputs=(input_relation,),
        scalar_refs=operation_refs,
        field_types={
            **{field: field_types.get(field, "") for field in spec.group_by},
            **{
                aggregation.output_field: (
                    "integer"
                    if aggregation.function.value == "count"
                    else "decimal"
                    if aggregation.function.value in {"sum", "avg"}
                    else field_types.get(aggregation.input_field, "")
                )
                for aggregation in spec.aggregations
            },
        },
    )


def _order(
    operation: ExecutableOperation,
    spec: OrderSpec,
    relations: dict[str, RelationRows],
    *,
    scalars: dict[str, RuntimeValue],
    scalar_types: dict[str, str],
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    rows = tuple(input_relation.rows)
    field_types = dict(input_relation.field_types or {})
    order_by = (*spec.order_by, *spec.tie_breakers)

    def key(row: Row) -> tuple[object, ...]:
        values: list[object] = []
        for sort in order_by:
            value = declared_order_key(
                _field(row, sort.field), field_types.get(sort.field)
            )
            values.append(
                _Descending(value) if sort.direction == SortDirection.DESC else value
            )
        return tuple(values)

    sorted_keyed_rows = sorted(
        ((key(row), row) for row in rows),
        key=lambda item: item[0],
    )
    limit = len(sorted_keyed_rows)
    if isinstance(spec.selection, Take):
        evaluated = evaluate_expression(
            spec.selection.limit,
            environment=ExpressionEnvironment(
                scalars=scalars,
                scalar_types=scalar_types,
            ),
        )
        try:
            limit = exact_positive_integer(evaluated.value)
        except (TypeError, ValueError) as exc:
            raise RelationEngineError("order take requires a positive integer") from exc
    elif not isinstance(spec.selection, KeepAll):
        assert_never(spec.selection)
    _verify_order_keys_are_unique_at_limit(sorted_keyed_rows, limit=limit)
    sorted_rows = [dict(row) for _, row in sorted_keyed_rows[:limit]]
    return _operation_relation(
        operation,
        sorted_rows,
        grain_keys=input_relation.grain_keys,
        inputs=(input_relation,),
        scalar_refs=operation_refs,
        field_types=field_types,
    )


def _verify_order_keys_are_unique_at_limit(
    keyed_rows: list[tuple[tuple[object, ...], Row]],
    *,
    limit: int,
) -> None:
    for index in range(1, min(limit, len(keyed_rows))):
        if keyed_rows[index - 1][0] == keyed_rows[index][0]:
            raise RelationEngineError("order requires unique ordering keys")
    if limit < len(keyed_rows) and keyed_rows[limit - 1][0] == keyed_rows[limit][0]:
        raise RelationEngineError("order requires unique ordering keys")
