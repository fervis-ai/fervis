"""Computed scalar pattern compiler."""

from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.model import FactFulfillment
from fervis.lookup.answer_program.operations import (
    ComputeBinary,
    ComputeBinaryOperator,
    ComputeNegation,
    ComputeSpec,
    Operation,
)
from fervis.lookup.answer_program.render_spec import RenderScalarOutput
from fervis.lookup.answer_program.values import ValueExpression
from fervis.lookup.source_binding import BoundSource
from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext

from .shared import (
    RelationBuilder,
    _dict,
    _pattern_output_relation_id,
    _required_strings,
    _scalar_output_spec,
    _text,
)
from .render_ids import _render_output_id
from .value_inputs import _scalar_inputs


def _compile_computed_scalar_answer(
    *,
    index: int,
    payload: dict[str, Any],
    namespace_render_outputs: bool,
    bound_sources: dict[str, BoundSource],
    input_context: CompilerInputContext,
    relation_builder: RelationBuilder,
) -> dict[str, Any]:
    del relation_builder
    output = _scalar_output_spec(_dict(payload.get("output"), "output"))
    render_output_id = _render_output_id(
        index,
        output["output_id"],
        namespace_render_outputs=namespace_render_outputs,
    )
    operation_id = f"{_pattern_output_relation_id(index)}_compute"
    scalar_inputs = _scalar_inputs(
        payload.get("scalar_inputs"),
        bound_sources=bound_sources,
    )
    answer_output_ids = _required_strings(
        payload.get("answer_output_ids"), "answer_output_ids"
    )
    return {
        "fulfillment": tuple(
            FactFulfillment(
                requested_fact_id=_text(payload.get("requested_fact_id")),
                answer_output_id=answer_output_id,
                render_output_id=render_output_id,
            )
            for answer_output_id in answer_output_ids
        ),
        "relations": (),
        "operations": (
            Operation(
                id=operation_id,
                spec=ComputeSpec(
                    expression=_compute_expression(
                        payload.get("expression"),
                        inputs={
                            item["input_id"]: input_context.expression_for_value(
                                item["value_id"]
                            )
                            for item in scalar_inputs
                        },
                    ),
                    output_scalar=output["scalar_id"],
                ),
            ),
        ),
        "relation_outputs": (),
        "scalar_outputs": (
            RenderScalarOutput(
                id=render_output_id,
                scalar_id=output["scalar_id"],
                label=output["label"] if namespace_render_outputs else "",
                role="answer_value",
            ),
        ),
    }


def _compute_expression(
    payload: object,
    *,
    inputs: dict[str, ValueExpression],
):
    if not isinstance(payload, list) or not payload:
        raise ValueError("computed scalar expression must be a non-empty token array")
    stack: list[object] = []
    for token in payload:
        if not isinstance(token, dict) or len(token) != 1:
            raise ValueError("computed scalar expression token is invalid")
        if "input_id" in token:
            input_id = _text(token["input_id"])
            if input_id not in inputs:
                raise ValueError("computed scalar expression input is not declared")
            stack.append(inputs[input_id])
            continue
        if str(token["operator"]) == "negate":
            if not stack:
                raise ValueError("computed scalar negation requires one operand")
            stack.append(ComputeNegation(operand=stack.pop()))
            continue
        operator = ComputeBinaryOperator(str(token.get("operator") or ""))
        if len(stack) < 2:
            raise ValueError("computed scalar operator requires two operands")
        right = stack.pop()
        left = stack.pop()
        stack.append(ComputeBinary(operator=operator, left=left, right=right))
    if len(stack) != 1:
        raise ValueError("computed scalar expression must produce one value")
    return stack[0]
