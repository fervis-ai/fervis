"""Data-independent schema interpretation of the executable expression algebra."""
from collections.abc import Mapping
from typing_extensions import assert_never

from fervis.lookup.answer_program.expressions import Expression, ExpressionFunction, fold_expression, expression_input_id
from fervis.lookup.answer_program.inputs import value_expression_type
from fervis.lookup.expression_operators import OperatorKind, operator_signature
from fervis.lookup.plan_execution.errors import RelationEngineError


def expression_value_type(
    expression: Expression, *, field_types: Mapping[str, str] | None = None,
    scalar_types: Mapping[str, str] | None = None,
    node_output_types: Mapping[str, Mapping[str, str]] | None = None,
    environment_types: Mapping[str, str] | None = None,
) -> str:
    def operator_type(operator):
        return 'decimal' if operator_signature(operator).kind is OperatorKind.ARITHMETIC else 'boolean'

    def function_type(node, arguments):
        if node.function is ExpressionFunction.ROW_NUMBER:
            return "integer"
        if node.function is ExpressionFunction.TEMPORAL_BUCKET:
            if len(arguments) != 3:
                raise RelationEngineError('temporal bucket requires value, grain, and timezone')
            return 'date'
        assert_never(node.function)

    return fold_expression(
        expression,
        field=lambda ref: (field_types or {}).get(ref.field_id, ''),
        parameter=lambda ref: (scalar_types or {}).get(expression_input_id(ref), ''),
        constant=lambda ref: value_expression_type(ref, ref.value),
        output=lambda ref: (node_output_types or {}).get(ref.node_id, {}).get(ref.output_id, ''),
        environment=lambda ref: (environment_types or {}).get(ref.key, ''),
        unary=lambda node, operand: operator_type(node.operator),
        binary=lambda node, left, right: operator_type(node.operator),
        function=function_type,
    )


def projected_grain(input_grain, outputs):
    from fervis.lookup.answer_program.expressions import FieldRef, FunctionExpression
    mapping = {item.expression.field_id:item.output_field for item in outputs if isinstance(item.expression, FieldRef)}
    if input_grain and all(key in mapping for key in input_grain):
        return tuple(mapping[key] for key in input_grain)
    row_numbers = tuple(item.output_field for item in outputs if isinstance(item.expression, FunctionExpression) and item.expression.function is ExpressionFunction.ROW_NUMBER)
    return row_numbers[:1]
