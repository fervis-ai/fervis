"""Computed scalar pattern compiler."""

from __future__ import annotations

from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.answer_program.operations import (
    ComputeBinary,
    ComputeBinaryOperator,
    ComputeExpression,
    ComputeNegation,
    ComputeSpec,
    Operation,
)
from fervis.lookup.answer_program.result_projection import ScalarResultOutput
from fervis.lookup.provider_contract import ProviderObject
from fervis.lookup.source_binding import BoundSource
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.fact_planning.provider_contract import (
    ComputeInputTokenOutput,
    ComputeOperatorTokenOutput,
    ComputedScalarAnswerOutput,
    parse_compute_expression_token,
)

from .shared import RelationBuilder, _pattern_output_relation_id
from .result_ids import _result_output_id
from fervis.lookup.fact_planning.compiled_patterns import CompiledPattern


def _compile_computed_scalar_answer(
    *,
    index: int,
    answer: ComputedScalarAnswerOutput,
    namespace_result_outputs: bool,
    bound_sources: dict[str, BoundSource],
    input_context: CompilerInputContext,
    relation_builder: RelationBuilder,
) -> CompiledPattern:
    del relation_builder
    scalar_id = answer.output.scalar_id
    label = answer.output.label or scalar_id
    result_output_id = _result_output_id(
        index,
        scalar_id,
        namespace_result_outputs=namespace_result_outputs,
    )
    operation_id = f"{_pattern_output_relation_id(index)}_compute"
    inputs: dict[str, ComputeExpression] = {}
    for item in answer.scalar_inputs:
        bound = bound_sources.get(item.source_binding_id)
        if bound is None or not bound.value_id:
            raise ValueError("scalar input requires value source binding")
        inputs[item.input_id] = input_context.compute_expression_for_value(
            bound.value_id
        )
    fulfillment = tuple(
        FactFulfillment(
            requested_fact_id=answer.requested_fact_id,
            answer_output_id=answer_output_id,
            result_output_id=result_output_id,
        )
        for answer_output_id in answer.answer_output_ids
    )
    operations = (
        Operation(
            id=operation_id,
            spec=ComputeSpec(
                expression=_compute_expression(
                    answer.expression,
                    inputs=inputs,
                ),
                output_scalar=scalar_id,
            ),
        ),
    )
    scalar_outputs = (
        ScalarResultOutput(
            id=result_output_id,
            scalar_id=scalar_id,
            label=label if namespace_result_outputs else "",
            role="answer_value",
        ),
    )
    return CompiledPattern(
        fulfillment=fulfillment,
        relations=(),
        operations=operations,
        relation_outputs=(),
        scalar_outputs=scalar_outputs,
    )


def _compute_expression(
    values: tuple[ProviderObject, ...],
    *,
    inputs: dict[str, ComputeExpression],
) -> ComputeExpression:
    if not values:
        raise ValueError("computed scalar expression must be a non-empty token array")
    stack: list[ComputeExpression] = []
    for value in values:
        token = parse_compute_expression_token(value)
        match token:
            case ComputeInputTokenOutput():
                input_id = token.input_id
                if input_id not in inputs:
                    raise ValueError("computed scalar expression input is not declared")
                stack.append(inputs[input_id])
            case ComputeOperatorTokenOutput(operator="negate"):
                if not stack:
                    raise ValueError("computed scalar negation requires one operand")
                stack.append(ComputeNegation(operand=stack.pop()))
            case ComputeOperatorTokenOutput():
                operator = ComputeBinaryOperator(token.operator)
                if len(stack) < 2:
                    raise ValueError("computed scalar operator requires two operands")
                right = stack.pop()
                left = stack.pop()
                stack.append(ComputeBinary(operator=operator, left=left, right=right))
    if len(stack) != 1:
        raise ValueError("computed scalar expression must produce one value")
    return stack[0]
