"""Deterministic source bindings for already-grounded endpoint inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    RelationInputOrigin,
)
from fervis.lookup.fact_plan.row_sources import RowSource
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.grounding.model import GroundedInputUse


@dataclass(frozen=True)
class _GroundedParam:
    param_id: str
    value_id: str
    value: Any
    value_component: str
    proof_refs: tuple[str, ...]


def grounded_param_bindings(
    *,
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[GroundedInputUse, ...],
    row_source: RowSource,
    requested_fact_id: str = "",
    excluded_param_ids: frozenset[str] = frozenset(),
) -> tuple[DraftEndpointParamBinding, ...]:
    """Compile grounded uses for one candidate into explicit source bindings."""

    values_by_id = {value.id: value for value in available_values}
    valid_param_ids = frozenset(param.id for param in row_source.params)
    grounded: dict[str, _GroundedParam] = {}
    for use in available_value_uses:
        if use.row_source_id != row_source.id:
            continue
        param_id = use.param_id
        if (
            not param_id
            or param_id not in valid_param_ids
            or param_id in excluded_param_ids
        ):
            continue
        value = values_by_id.get(use.value_id)
        if value is None or not _value_applies(value, requested_fact_id):
            continue
        component = use.value_component
        concrete_value = value_component(value, component)
        existing = grounded.get(param_id)
        if existing is not None and existing.value != concrete_value:
            raise ValueError(f"conflicting grounded values for source param {param_id}")
        grounded[param_id] = _GroundedParam(
            param_id=param_id,
            value_id=existing.value_id if existing is not None else value.id,
            value=concrete_value,
            value_component=(
                existing.value_component
                if existing is not None
                else component.value
            ),
            proof_refs=tuple(
                dict.fromkeys(
                    (*(() if existing is None else existing.proof_refs), *value.proof_refs)
                )
            ),
        )
    return tuple(
        DraftEndpointParamBinding(
            param_id=item.param_id,
            value=item.value,
            origin_kind=RelationInputOrigin.QUESTION_INPUT,
            value_id=item.value_id,
            value_component=item.value_component,
            proof_refs=item.proof_refs,
        )
        for item in grounded.values()
    )


def _value_applies(
    value: FactValue,
    requested_fact_id: str,
) -> bool:
    return (
        not requested_fact_id
        or not value.applies_to_requested_fact_ids
        or requested_fact_id in value.applies_to_requested_fact_ids
    )
