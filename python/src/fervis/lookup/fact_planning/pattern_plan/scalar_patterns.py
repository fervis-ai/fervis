"""Computed scalar pattern compiler."""

from __future__ import annotations

from typing import Any

from fervis.lookup.fact_plan.fact_plan import FactFulfillment
from fervis.lookup.fact_plan.operations import ComputeSpec, Operation
from fervis.lookup.fact_plan.render_spec import RenderScalarOutput
from fervis.lookup.fact_plan.values import ScalarInputUse, ValueUse
from fervis.lookup.source_binding import BoundSource

from .shared import (
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
) -> dict[str, Any]:
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
        "values": (),
        "value_uses": tuple(
            ValueUse(
                id=f"{operation_id}_{item['input_id']}",
                value_id=item["value_id"],
                target=ScalarInputUse(
                    operation_id=operation_id,
                    input_id=item["input_id"],
                ),
            )
            for item in scalar_inputs
        ),
        "relations": (),
        "operations": (
            Operation(
                id=operation_id,
                spec=ComputeSpec(
                    expression=_text(payload.get("expression")),
                    scalar_inputs=tuple(item["input_id"] for item in scalar_inputs),
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
