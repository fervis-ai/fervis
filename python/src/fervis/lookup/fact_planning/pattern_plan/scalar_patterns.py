"""Computed scalar pattern compiler."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    Expression,
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
    UnaryExpression,
    NodeOutputRef,
)
from fervis.lookup.answer_program.operations import (
    ComputeInputPopulationCoverage,
    ComputeSpec,
    Operation,
    compute_value_input_id,
)
from fervis.lookup.answer_program.relations import (
    Relation,
    merge_population_coverage_claims,
)
from fervis.lookup.answer_program.result_projection import ScalarResultOutput
from fervis.lookup.provider_contract import ProviderObject
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
from fervis.lookup.fact_planning.compiled_patterns import PatternAddress
from fervis.lookup.fact_planning.scalar_values import SourceDerivedScalarValue
from fervis.lookup.source_binding import BoundSource

from .aggregate_patterns import _compile_scalar_aggregate_source
from .shared import _pattern_relation_id


def _compile_computed_scalar_answer(
    *,
    index: int,
    answer: ComputedScalarAnswerOutput,
    namespace_result_outputs: bool,
    input_context: CompilerInputContext,
    relation_builder: RelationBuilder,
    bound_sources: dict[str, BoundSource],
    source_scalar_values_by_id: Mapping[str, SourceDerivedScalarValue],
) -> CompiledPattern:
    scalar_id = answer.output.scalar_id
    label = answer.output.label or scalar_id
    result_output_id = _result_output_id(
        index,
        scalar_id,
        namespace_result_outputs=namespace_result_outputs,
    )
    operation_id = f"{_pattern_output_relation_id(index)}_compute"
    inputs: dict[str, Expression] = {}
    population_claims_by_input: dict[str, list] = {}
    relations: list[Relation] = []
    source_operations: list[Operation] = []
    for input_index, item in enumerate(answer.scalar_inputs, start=1):
        if item.input_id in inputs:
            raise ValueError("computed scalar input id must be unique")
        source_value = source_scalar_values_by_id.get(item.value_id)
        if source_value is not None:
            if source_value.requested_fact_id != answer.requested_fact_id:
                raise ValueError("computed scalar source value belongs to another fact")
            source_relation_id = (
                f"{_pattern_relation_id(index)}_scalar_input_{input_index}"
            )
            source_output_relation_id = f"{source_relation_id}_aggregate"
            compiled_source = _compile_scalar_aggregate_source(
                address=PatternAddress(
                    requested_fact_id=answer.requested_fact_id,
                    answer_output_ids=(source_value.metric.answer_output_id,),
                    plan_shape=answer.pattern,
                    source_binding_id=source_value.source_binding_id,
                ),
                metric=source_value.metric,
                relation_id=source_relation_id,
                output_relation_id=source_output_relation_id,
                bound_sources=bound_sources,
                relation_builder=relation_builder,
            )
            relations.extend(compiled_source.relations)
            source_operations.extend(compiled_source.operations)
            expression: Expression = NodeOutputRef(
                node_id=compiled_source.aggregate_operation_id,
                output_id=source_value.metric.output_field_id,
            )
        else:
            if input_context.value_type(item.value_id) != "number":
                raise ValueError("computed scalar inputs must be numeric")
            expression = input_context.compute_expression_for_value(item.value_id)
        inputs[item.input_id] = expression
        claims = input_context.population_coverage_for_value(item.value_id)
        if claims:
            input_ref = compute_value_input_id(expression)
            population_claims_by_input.setdefault(input_ref, []).extend(claims)
    fulfillment = tuple(
        FactFulfillment(
            requested_fact_id=answer.requested_fact_id,
            answer_output_id=answer_output_id,
            result_output_id=result_output_id,
        )
        for answer_output_id in answer.answer_output_ids
    )
    operations = (
        *source_operations,
        Operation(
            id=operation_id,
            spec=ComputeSpec(
                expression=_compute_expression(
                    answer.expression,
                    inputs=inputs,
                ),
                output_scalar=scalar_id,
                input_population_coverage=tuple(
                    ComputeInputPopulationCoverage(
                        input_id=input_id,
                        claims=merge_population_coverage_claims(tuple(claims)),
                    )
                    for input_id, claims in population_claims_by_input.items()
                ),
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
        relations=tuple(relations),
        operations=operations,
        relation_outputs=(),
        scalar_outputs=scalar_outputs,
    )


def _compute_expression(
    values: tuple[ProviderObject, ...],
    *,
    inputs: dict[str, Expression],
) -> Expression:
    if not values:
        raise ValueError("computed scalar expression must be a non-empty token array")
    stack: list[Expression] = []
    used_input_ids: set[str] = set()
    for value in values:
        token = parse_compute_expression_token(value)
        match token:
            case ComputeInputTokenOutput():
                input_id = token.input_id
                if input_id not in inputs:
                    raise ValueError("computed scalar expression input is not declared")
                used_input_ids.add(input_id)
                stack.append(inputs[input_id])
            case ComputeOperatorTokenOutput(operator="negate"):
                if not stack:
                    raise ValueError("computed scalar negation requires one operand")
                stack.append(
                    UnaryExpression(
                        operator=ExpressionUnaryOperator.NEGATE,
                        operand=stack.pop(),
                    )
                )
            case ComputeOperatorTokenOutput():
                operator = ExpressionBinaryOperator(token.operator)
                if len(stack) < 2:
                    raise ValueError("computed scalar operator requires two operands")
                right = stack.pop()
                left = stack.pop()
                stack.append(
                    BinaryExpression(operator=operator, left=left, right=right)
                )
    if len(stack) != 1:
        raise ValueError("computed scalar expression must produce one value")
    if used_input_ids != set(inputs):
        raise ValueError("computed scalar declares an unused scalar input")
    return stack[0]
