from datetime import datetime, timezone

import pytest
from fervis.lookup.answer_program.operations import FilterSpec, AggregateSpec

from fervis.lookup.answer_program.expressions import (
    ExpressionBinaryOperator,
    expression_input_id,
    expression_references,
)
from fervis.lookup.answer_program.inputs import (
    resolve_value_expression,
    resolved_value_expression_type,
)
from fervis.lookup.answer_program.values import (
    ANCHOR_TIMEZONE_REF,
    ConstantRef,
    ParameterRef,
    FactValue,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import (
    ExecutableOperation,
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.question_contract.model import InputTerm
from fervis.lookup.relation_catalog.row_sources import RowSourceValueType
from fervis.lookup.semantic_types import (
    DateTimeType,
    SourceOrigin,
    SourceOriginKind,
    TemporalScopeType,
)
from tests.lookup.fact_compilation.test_compiler import _compile_returned_row_predicate


@pytest.mark.parametrize("aggregate_filter", [False, True])
@pytest.mark.parametrize("precision", ["day", "hour"])
def test_temporal_scope_membership_preserves_calendar_and_instant_bounds(
    aggregate_filter, precision
):
    period = InputTerm(
        "i1",
        SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "March 2, 2026"),
        "March 2, 2026",
        TemporalScopeType(),
    )
    result, _ = _compile_returned_row_predicate(
        input_term=period,
        typed_value=FactValue.time(
            id="period",
            known_input_id="i1",
            expression="March 2, 2026",
            resolved_start="2026-03-02"
            if precision == "day"
            else "2026-03-02T09:00:00+00:00",
            resolved_end="2026-03-02"
            if precision == "day"
            else "2026-03-02T20:59:59+00:00",
            granularity=precision,
            proof_refs=("question_input:i1",),
        ),
        fact_type=DateTimeType(),
        field_type=RowSourceValueType.DATETIME,
        operator=ExpressionBinaryOperator.WITHIN,
        aggregate_filter=aggregate_filter,
    )
    program = result.answer_program
    operation = program.operations[-1]
    scalars = {}
    conditions = [
        op.spec.condition
        for op in program.operations
        if isinstance(op.spec, FilterSpec)
    ]
    conditions.extend(
        aggregation.filter
        for op in program.operations
        if isinstance(op.spec, AggregateSpec)
        for aggregation in op.spec.aggregations
        if aggregation.filter is not None
    )
    for condition in conditions:
        for leaf in expression_references(condition).leaves:
            if isinstance(leaf, (ParameterRef, ConstantRef)):
                value = resolve_value_expression(leaf, bindings=result.initial_bindings)
                scalars[expression_input_id(leaf)] = ScalarInput(
                    expression_input_id(leaf),
                    value.value,
                    resolved_value_expression_type(leaf, value),
                )
    values = (
        datetime(2026, 3, 1, 20, 59, tzinfo=timezone.utc),
        datetime(2026, 3, 1, 21, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 9, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 20, 59, 59, tzinfo=timezone.utc),
        datetime(2026, 3, 2, 21, tzinfo=timezone.utc),
        None,
    )
    execution = execute_operations(
        RelationEngineInput(
            relations=(
                RelationRows(
                    program.relations[0].id,
                    tuple({"value": value} for value in values),
                    field_types={"value": "datetime"},
                    completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
                ),
            ),
            operations=tuple(
                ExecutableOperation(op.id, op.spec, op.output_relation)
                for op in program.operations
            ),
            scalar_inputs=tuple(scalars.values()),
            environment_values={ANCHOR_TIMEZONE_REF: "Africa/Nairobi"},
            environment_types={ANCHOR_TIMEZONE_REF: "string"},
        )
    )
    assert execution.issue is None
    assert execution.relation(operation.output_relation).rows == (
        {"aggregate_1": 3 if precision == "day" else 2},
    )
