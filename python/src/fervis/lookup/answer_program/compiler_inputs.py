"""One compiler boundary for initial answer-program inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NoReturn

from fervis.lookup.answer_program.contracts import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRole,
    ProgramInputs,
    parameter_value_type,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    NodeOutputRef,
    ParameterRef,
    ValueExpression,
)
from fervis.lookup.answer_program.operations import ComputeExpressionLeaf
from fervis.lookup.answer_program.values import fold_value_expression
from fervis.lookup.question_contract import QuestionContract


@dataclass(frozen=True)
class CompilerInputContext:
    program_inputs: ProgramInputs
    expressions_by_value_id: dict[str, ValueExpression]

    def expression_for_value(
        self,
        value_id: str,
        *,
        component: str = "value",
        item_index: int | None = None,
    ) -> ValueExpression:
        expression = self.expressions_by_value_id.get(value_id)
        if expression is None:
            raise ValueError(f"no declared value origin for {value_id}")
        if isinstance(expression, ParameterRef):
            return replace(
                expression,
                component=component,
                item_index=item_index,
            )
        if isinstance(expression, ConstantRef):
            return replace(
                expression,
                component=component,
                item_index=item_index,
            )
        if component != "value" or item_index is not None:
            raise ValueError(f"{value_id} does not support value components")
        return expression

    def compute_expression_for_value(self, value_id: str) -> ComputeExpressionLeaf:
        expression = self.expression_for_value(value_id)
        return fold_value_expression(
            expression,
            parameter=_compute_parameter,
            output=_compute_output,
            constant=_compute_constant,
            environment=lambda _item: _unsupported_compute_environment(value_id),
        )


def _compute_parameter(item: ParameterRef) -> ComputeExpressionLeaf:
    return item


def _compute_output(item: NodeOutputRef) -> ComputeExpressionLeaf:
    return item


def _compute_constant(item: ConstantRef) -> ComputeExpressionLeaf:
    return item


def _unsupported_compute_environment(value_id: str) -> NoReturn:
    raise ValueError(f"{value_id} cannot be used as a compute operand")


def compiler_input_context(
    *,
    values: tuple[FactValue, ...],
    question_contract: QuestionContract,
) -> CompilerInputContext:
    """Classify current values once from their explicit owning contracts."""

    question_inputs = {item.id: item for item in question_contract.question_inputs}
    parameters: list[ParameterDeclaration] = []
    bindings: list[ParameterBinding] = []
    expressions: dict[str, ValueExpression] = {}
    seen_parameters: set[str] = set()
    for value in values:
        known_input_id = value.known_input_id
        if known_input_id:
            known_input = question_inputs.get(known_input_id)
            if known_input is None:
                raise ValueError(
                    f"value {value.id} references unknown question input "
                    f"{known_input_id}"
                )
            parameter_id = f"question.{known_input_id}"
            if parameter_id in seen_parameters:
                raise ValueError(f"duplicate binding for {parameter_id}")
            seen_parameters.add(parameter_id)
            role = (
                ParameterRole.PLAN_CONTROL
                if known_input.is_result_limit
                else ParameterRole.QUESTION_INPUT
            )
            parameters.append(
                ParameterDeclaration(
                    id=parameter_id,
                    role=role,
                    value_type=parameter_value_type(value),
                )
            )
            bindings.append(
                ParameterBinding(
                    parameter_id=parameter_id,
                    value=value,
                    provenance=BindingProvenance(
                        kind=BindingProvenanceKind.QUESTION_INPUT,
                        refs=(f"known_input:{known_input_id}",),
                    ),
                )
            )
            expressions[value.id] = ParameterRef(parameter_id=parameter_id)
            continue
        expressions[value.id] = ConstantRef(
            constant_id=f"context.{value.id}",
            version_ref="context-value@1",
            value=value,
        )
    return CompilerInputContext(
        program_inputs=ProgramInputs(
            parameters=tuple(parameters),
            bindings=BindingSet.from_bindings(tuple(bindings)),
        ),
        expressions_by_value_id=expressions,
    )
