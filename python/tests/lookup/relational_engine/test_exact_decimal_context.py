"""Typed factual arithmetic must not inherit a caller's Decimal precision."""

from decimal import Decimal, localcontext
from dataclasses import replace
import pytest

from fervis.lookup.answer_program.expressions import (
    BinaryExpression, ExpressionBinaryOperator, FieldRef,
)
from fervis.lookup.answer_program.operations import AggregationFunction, AggregationSpec
from fervis.lookup.plan_execution.operation_engine.expression_evaluator import (
    ExpressionEnvironment, evaluate_expression,
)
from fervis.lookup.plan_execution.operation_engine.shared import _aggregate_value
from fervis.lookup.plan_execution.exact_decimal import stable_divide


def test_typed_sum_average_and_addition_keep_exact_cancelling_decimals():
    left = Decimal("12345678901234567890.123456789012345678")
    right = Decimal("-12345678901234567890.123456789012345677")
    rows = [{"value": left}, {"value": right}]

    def aggregate(function):
        return _aggregate_value(
            AggregationSpec(function, "out", "value"), rows,
            {"value": "decimal"}, node_outputs={}, node_output_types={},
            scalars={}, scalar_types={}, environment_values={}, environment_types={},
        )

    arithmetic = BinaryExpression(
        ExpressionBinaryOperator.ADD, FieldRef("left"), FieldRef("right")
    )
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            assert aggregate(AggregationFunction.SUM) == Decimal("0.000000000000000001")
            assert aggregate(AggregationFunction.AVG) == Decimal("0.0000000000000000005")
            assert evaluate_expression(arithmetic, environment=ExpressionEnvironment(
                row={"left": left, "right": right},
                field_types={"left": "decimal", "right": "decimal"},
            )).value == Decimal("0.000000000000000001")


def test_typed_product_and_nonterminating_division_ignore_ambient_precision():
    left = Decimal("12345678901234567890.123456789012345678")
    right = Decimal("3.000000000000000001")
    exact_product = Decimal(
        str(12345678901234567890123456789012345678 * 3000000000000000001)
        + "e-36"
    )
    multiply = BinaryExpression(
        ExpressionBinaryOperator.MULTIPLY, FieldRef("left"), FieldRef("right")
    )
    divisions = []
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            assert evaluate_expression(multiply, environment=ExpressionEnvironment(
                row={"left": left, "right": right},
                field_types={"left": "decimal", "right": "decimal"},
            )).value == exact_product
            divisions.append(stable_divide(Decimal(1), Decimal(3)))
    assert divisions[0] == divisions[1] == divisions[2]
    assert len(divisions[0].as_tuple().digits) == 50


def test_saved_program_with_previous_numeric_semantics_is_not_reused():
    from tests.lookup.fact_compilation.test_compiler import _compile_memory_count
    from fervis.lookup.answer_program.compatibility import verify_program_compatibility
    from fervis.lookup.plan_execution.errors import VerificationError
    from fervis.lookup.relation_catalog import RelationCatalog

    _, _, memory_relation, compiled, _, _ = _compile_memory_count(({"event_id": "a"},))
    program = compiled.answer_program
    stale = replace(program, compatibility=replace(
        program.compatibility,
        function_semantics=tuple(
            replace(item, version="3") if item.function_key == "relation.aggregate" else item
            for item in program.compatibility.function_semantics
        ),
    ))
    with pytest.raises(VerificationError, match="incompatible_function_semantics"):
        verify_program_compatibility(
            stale, catalog=RelationCatalog(), memory_relations=(memory_relation,)
        )
