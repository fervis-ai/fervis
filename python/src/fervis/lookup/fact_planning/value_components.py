"""Concrete values for typed fact-plan value components."""

from __future__ import annotations

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.values import (
    FactValue,
    TimeComponent,
    ValueComponent,
    ValueComponentValue,
)


def value_component(
    value: FactValue,
    component: ValueComponent | TimeComponent,
) -> ValueComponentValue:
    try:
        return value.payload.component_value(component)
    except ValueError as exc:
        raise VerificationError(f"{value.id} does not have {component.value}") from exc
