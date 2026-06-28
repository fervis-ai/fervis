"""Canonical endpoint-parameter values produced by grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.fact_plan.values import FactValue


@dataclass(frozen=True)
class GroundedParamValue:
    row_source_id: str
    param_id: str
    value_id: str
    value: Any
    field_id: str = ""
    proof_refs: tuple[str, ...] = ()


def unique_grounded_param_values(
    *,
    values: tuple[FactValue, ...],
    grounded_input_uses: tuple[object, ...],
) -> dict[tuple[str, str], GroundedParamValue]:
    values_by_id = {item.id: item for item in values}
    grouped: dict[tuple[str, str], dict[object, GroundedParamValue]] = {}
    for use in grounded_input_uses:
        row_source_id = str(getattr(use, "row_source_id", "") or "")
        param_id = str(getattr(use, "param_id", "") or "")
        field_id = str(getattr(use, "field_id", "") or "")
        if not row_source_id or not param_id:
            continue
        value = values_by_id.get(str(getattr(use, "value_id", "") or ""))
        if value is None:
            continue
        concrete_value = value_component(value, getattr(use, "value_component"))
        value_key = _value_key(concrete_value)
        group = grouped.setdefault((row_source_id, param_id), {})
        existing = group.get(value_key)
        proof_refs = tuple(
            dict.fromkeys(
                (
                    *(existing.proof_refs if existing is not None else ()),
                    *value.proof_refs,
                )
            )
        )
        group[value_key] = GroundedParamValue(
            row_source_id=row_source_id,
            param_id=param_id,
            field_id=field_id,
            value_id=existing.value_id if existing is not None else value.id,
            value=concrete_value,
            proof_refs=proof_refs,
        )
    return {
        key: next(iter(values_by_key.values()))
        for key, values_by_key in grouped.items()
        if len(values_by_key) == 1
    }


def unique_grounded_param_ids_by_row_source(
    *,
    values: tuple[FactValue, ...],
    grounded_input_uses: tuple[object, ...],
) -> dict[str, frozenset[str]]:
    output: dict[str, set[str]] = {}
    for row_source_id, param_id in unique_grounded_param_values(
        values=values,
        grounded_input_uses=grounded_input_uses,
    ):
        output.setdefault(row_source_id, set()).add(param_id)
    return {
        row_source_id: frozenset(param_ids)
        for row_source_id, param_ids in output.items()
    }


def _value_key(value: Any) -> object:
    if isinstance(value, dict):
        return tuple(sorted((key, _value_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_value_key(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_value_key(item) for item in value))
    return value
