"""Concrete values for typed fact-plan value components."""

from __future__ import annotations

from typing import Any

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.fact_plan.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralValuePayload,
    NamedValuePayload,
    TimeComponent,
    TimeValuePayload,
    ValueComponent,
    ValueKind,
)


def value_component(
    value: FactValue,
    component: ValueComponent | TimeComponent,
) -> Any:
    if value.kind == ValueKind.IDENTITY and isinstance(
        value.payload, IdentityValuePayload
    ):
        if component != ValueComponent.VALUE:
            raise VerificationError(f"{value.id} does not have {component.value}")
        return value.payload.value
    if value.kind == ValueKind.IDENTITY_SET and isinstance(
        value.payload, IdentitySetValuePayload
    ):
        if component != ValueComponent.VALUE:
            raise VerificationError(f"{value.id} does not have {component.value}")
        return value.payload.values
    if value.kind == ValueKind.NAMED and isinstance(value.payload, NamedValuePayload):
        if component != ValueComponent.VALUE:
            raise VerificationError(f"{value.id} does not have {component.value}")
        return value.payload.text
    if value.kind == ValueKind.LITERAL and isinstance(
        value.payload, LiteralValuePayload
    ):
        if component != ValueComponent.VALUE:
            raise VerificationError(f"{value.id} does not have {component.value}")
        return value.payload.value
    if value.kind == ValueKind.TIME and isinstance(value.payload, TimeValuePayload):
        if component == ValueComponent.VALUE:
            if (
                not value.payload.resolved_start
                or value.payload.resolved_start != value.payload.resolved_end
            ):
                raise VerificationError(f"{value.id} does not have instant")
            return value.payload.resolved_start
        if component == TimeComponent.START:
            if not value.payload.resolved_start:
                raise VerificationError(f"{value.id} does not have start")
            return value.payload.resolved_start
        if component == TimeComponent.END:
            if not value.payload.resolved_end:
                raise VerificationError(f"{value.id} does not have end")
            return value.payload.resolved_end
        if component == TimeComponent.INSTANT:
            if (
                not value.payload.resolved_start
                or value.payload.resolved_start != value.payload.resolved_end
            ):
                raise VerificationError(f"{value.id} does not have instant")
            return value.payload.resolved_start
        raise VerificationError(f"{value.id} requires explicit time component")
    raise VerificationError(f"value {value.id} has incomplete payload")
