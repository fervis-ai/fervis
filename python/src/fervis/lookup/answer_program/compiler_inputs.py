"""One compiler boundary for initial answer-program inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fervis.lookup.answer_program.contracts import ProgramInputs, parameter_value_type
from fervis.lookup.answer_program.expressions import (
    ConstantRef,
    Expression,
    ExpressionLeaf,
    NodeOutputRef,
    ParameterRef,
)


@dataclass(frozen=True)
class CompilerInputContext:
    program_inputs: ProgramInputs
    expressions_by_value_id: dict[str, Expression]
    value_types_by_value_id: dict[str, str]

    @property
    def expressions_by_input_ref(self) -> dict[str, Expression]:
        declarations = {item.id: item for item in self.program_inputs.parameters}
        output: dict[str, Expression] = {}
        for binding in self.program_inputs.bindings.bindings:
            declaration = declarations[binding.parameter_id]
            input_ref = declaration.input_ref or binding.value.known_input_id
            if not input_ref:
                continue
            if input_ref in output:
                raise ValueError("question input values must have one declared origin")
            output[input_ref] = ParameterRef(parameter_id=binding.parameter_id)
        return output

    def expression_for_value(
        self,
        value_id: str,
        *,
        component: str = "value",
        item_index: int | None = None,
    ) -> Expression:
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

    def compute_expression_for_value(self, value_id: str) -> ExpressionLeaf:
        expression = self.expression_for_value(value_id)
        if isinstance(expression, (ParameterRef, NodeOutputRef, ConstantRef)):
            return expression
        raise ValueError(f"{value_id} cannot be used as a compute operand")

    def value_type(self, value_id: str) -> str:
        value_type = self.value_types_by_value_id.get(value_id)
        if value_type is None:
            raise ValueError(f"no declared value type for {value_id}")
        return value_type

    def expression_for_question_input(
        self,
        question_input_id: str,
        *,
        component: str = "value",
    ) -> Expression:
        expression = self.expressions_by_input_ref.get(question_input_id)
        if expression is None:
            raise ValueError(f"question input {question_input_id} has no value")
        if isinstance(expression, ParameterRef):
            return replace(expression, component=component)
        if isinstance(expression, ConstantRef):
            return replace(expression, component=component)
        if component != "value":
            raise ValueError(
                f"question input {question_input_id} does not support value components"
            )
        return expression


def compiler_input_context_from_program_inputs(
    program_inputs: ProgramInputs,
    *,
    constant_expressions: dict[str, ConstantRef] | None = None,
) -> CompilerInputContext:
    """Build the one compiler value index from already-declared inputs."""

    declared = {item.id: item for item in program_inputs.parameters}
    if len(declared) != len(program_inputs.parameters):
        raise ValueError("compiler input parameters must be unique")
    constants = dict(constant_expressions or {})
    expressions: dict[str, Expression] = dict(constants)
    value_types: dict[str, str] = {
        value_id: parameter_value_type(expression.value)
        for value_id, expression in constants.items()
    }
    for binding in program_inputs.bindings.bindings:
        declaration = declared.get(binding.parameter_id)
        if declaration is None:
            raise ValueError("compiler binding references an unknown parameter")
        if declaration.value_type is not parameter_value_type(binding.value):
            raise ValueError("compiler binding type does not match its parameter")
        if binding.value.id in expressions:
            raise ValueError("compiler values must have one declared origin")
        expressions[binding.value.id] = ParameterRef(parameter_id=binding.parameter_id)
        value_types[binding.value.id] = declaration.value_type.value
    return CompilerInputContext(
        program_inputs=program_inputs,
        expressions_by_value_id=expressions,
        value_types_by_value_id=value_types,
    )
