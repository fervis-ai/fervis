"""Canonical endpoint-parameter values produced by grounding."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.answer_program.values import FactValue, ValueComponentValue
from fervis.lookup.grounding.model import GroundedInputUse


@dataclass(frozen=True)
class GroundedParamValue:
    row_source_id: str
    param_id: str
    value_id: str
    value: ValueComponentValue
    field_id: str = ""
    entity_kind: str = ""
    proof_refs: tuple[str, ...] = ()


def unique_grounded_param_values(
    *,
    values: tuple[FactValue, ...],
    grounded_input_uses: tuple[GroundedInputUse, ...],
) -> dict[tuple[str, str], GroundedParamValue]:
    values_by_id = {item.id: item for item in values}
    grouped: dict[tuple[str, str], dict[ValueComponentValue, GroundedParamValue]] = {}
    for use in grounded_input_uses:
        row_source_id = use.row_source_id
        param_id = use.param_id
        field_id = use.field_id
        entity_kind = use.entity_kind
        if not row_source_id or not param_id:
            continue
        value = values_by_id.get(use.value_id)
        if value is None:
            continue
        concrete_value = value_component(value, use.value_component)
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
            entity_kind=entity_kind,
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
    grounded_input_uses: tuple[GroundedInputUse, ...],
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


def _value_key(value: ValueComponentValue) -> ValueComponentValue:
    if isinstance(value, tuple):
        return tuple(sorted(value))
    if isinstance(value, Decimal):
        return value.normalize()
    return value
