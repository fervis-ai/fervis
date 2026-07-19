"""Concrete values for typed fact-plan value components."""

from __future__ import annotations

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentityValuePayload,
    TimeComponent,
    ValueComponent,
    ValueComponentValue,
)
from fervis.lookup.grounding.model import GroundedInputUse


def value_component(
    value: FactValue,
    component: ValueComponent | TimeComponent,
) -> ValueComponentValue:
    try:
        return value.payload.component_value(component)
    except ValueError as exc:
        raise VerificationError(f"{value.id} does not have {component.value}") from exc


def grounded_input_value(
    value: FactValue,
    use: GroundedInputUse,
) -> ValueComponentValue:
    if not use.key_component_id:
        return value_component(value, use.value_component)
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        raise VerificationError(
            f"{value.id} does not carry an entity-key component"
        )
    try:
        return str(payload.key.component_value(use.key_component_id))
    except KeyError as exc:
        raise VerificationError(
            f"{value.id} does not have key component {use.key_component_id}"
        ) from exc
