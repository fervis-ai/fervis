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
from fervis.lookup.outcomes.errors import UndefinedOperationError
from fervis.lookup.outcomes.operation_semantics import empty_aggregation_undefined_reason
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    SortDirection,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.answer_program.operations import AtPosition, KeepAll, OrderSpec, Take, order_selection_expression
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
    _relation,
)


def _aggregate(
    operation: ExecutableOperation,
    spec: AggregateSpec,
    relations: dict[str, RelationRows],
    *,
    node_outputs: dict[str, dict[str, RuntimeValue]],
    node_output_types: dict[str, dict[str, str]],
    scalars: dict[str, RuntimeValue],
    scalar_types: dict[str, str],
    environment_values: dict[str, RuntimeValue],
    environment_types: dict[str, str],
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    if input_relation.completeness.status != CompletenessStatus.COMPLETE:
        raise IncompleteEvidenceError(
            relation_id=input_relation.id,
            proof_refs=input_relation.completeness.proof_refs,
        )
    if not input_relation.rows and not spec.group_by:
        for aggregation in spec.aggregations:
            reason = empty_aggregation_undefined_reason(aggregation.function)
            if reason is not None:
                raise UndefinedOperationError(
                    reason_code=reason,
                    input_refs=(aggregation.input_field,),
                )
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
                aggregation,
                rows,
                field_types,
                empty_is_null=bool(spec.group_by),
                node_outputs=node_outputs,
                node_output_types=node_output_types,
                scalars=scalars,
                scalar_types=scalar_types,
                environment_values=environment_values,
                environment_types=environment_types,
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
                    "boolean" if aggregation.function in {AggregationFunction.BOOL_ANY, AggregationFunction.BOOL_ALL} else "integer"
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
    node_outputs: dict[str, dict[str, RuntimeValue]],
    node_output_types: dict[str, dict[str, str]],
    scalars: dict[str, RuntimeValue],
    scalar_types: dict[str, str],
    environment_values: dict[str, RuntimeValue],
    environment_types: dict[str, str],
    operation_refs: tuple[str, ...] = (),
) -> RelationRows:
    input_relation = _relation(relations, spec.input_relation)
    rows = tuple(input_relation.rows)
    field_types = dict(input_relation.field_types or {})
    order_by = spec.order_by

    def key(row: Row) -> tuple[object, ...]:
        values: list[object] = []
        for sort in order_by:
            raw_value = _field(row, sort.field)
            if raw_value is None:
                values.append((1, ""))
                continue
            value = declared_order_key(raw_value, field_types.get(sort.field))
            directed = (
                _Descending(value)
                if sort.direction == SortDirection.DESC
                else value
            )
            values.append((0, directed))
        return tuple(values)

    sorted_keyed_rows = sorted(
        ((key(row), row) for row in rows),
        key=lambda item: item[0],
    )
    limit = len(sorted_keyed_rows)
    if isinstance(spec.selection, (Take, AtPosition)):
        selection_expression = order_selection_expression(spec.selection)
        assert selection_expression is not None
        evaluated = evaluate_expression(
            selection_expression,
            environment=ExpressionEnvironment(
                node_outputs=node_outputs, node_output_types=node_output_types,
                scalars=scalars,
                scalar_types=scalar_types,
                environment_values=environment_values,
                environment_types=environment_types,
            ),
        )
        try:
            limit = exact_positive_integer(evaluated.value)
        except (TypeError, ValueError) as exc:
            raise RelationEngineError("order take requires a positive integer") from exc
    elif not isinstance(spec.selection, KeepAll):
        assert_never(spec.selection)
    if isinstance(spec.selection, AtPosition):
        boundary_key = sorted_keyed_rows[limit - 1][0] if limit <= len(sorted_keyed_rows) else None
        selected_rows = [item for item in sorted_keyed_rows if item[0] == boundary_key]
    else:
        selected_rows = _rows_through_boundary_ties(sorted_keyed_rows, limit=limit)
    sorted_rows = [dict(row) for _, row in selected_rows]
    return _operation_relation(
        operation,
        sorted_rows,
        grain_keys=input_relation.grain_keys,
        inputs=(input_relation,),
        scalar_refs=operation_refs,
        field_types=field_types,
    )


def _rows_through_boundary_ties(
    keyed_rows: list[tuple[tuple[object, ...], Row]],
    *,
    limit: int,
) -> list[tuple[tuple[object, ...], Row]]:
    selected = keyed_rows[:limit]
    if not selected or limit >= len(keyed_rows):
        return selected
    boundary_key = selected[-1][0]
    index = limit
    while index < len(keyed_rows) and keyed_rows[index][0] == boundary_key:
        selected.append(keyed_rows[index])
        index += 1
    return selected
