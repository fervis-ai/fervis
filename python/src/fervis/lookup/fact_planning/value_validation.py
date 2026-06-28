"""Value contract validation."""

from __future__ import annotations

from fervis.lookup.grounding.time_resolution import validate_time_intent
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.fact_plan.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralValuePayload,
    NamedValuePayload,
    TimeValuePayload,
    ValueKind,
    ValueUse,
)


def verify_value_contract(
    *,
    values: tuple[FactValue, ...],
    value_uses: tuple[ValueUse, ...],
    available_values: tuple[FactValue, ...] = (),
) -> None:
    value_ids = set()
    for value in (*values, *available_values):
        if not value.id:
            raise VerificationError("value requires id")
        if value.id in value_ids:
            raise VerificationError(f"duplicate value {value.id}")
        value_ids.add(value.id)
        _verify_payload(value)
    value_use_ids: set[str] = set()
    for value_use in value_uses:
        if not value_use.id:
            raise VerificationError("value use requires id")
        if value_use.id in value_use_ids:
            raise VerificationError(f"duplicate value use {value_use.id}")
        value_use_ids.add(value_use.id)
        if not value_use.value_id:
            raise VerificationError("value use requires value id")
        if value_use.value_id not in value_ids:
            raise VerificationError(
                f"value use {value_use.id} references unknown value"
            )


def _verify_payload(value: FactValue) -> None:
    expected_types = {
        ValueKind.IDENTITY: IdentityValuePayload,
        ValueKind.IDENTITY_SET: IdentitySetValuePayload,
        ValueKind.NAMED: NamedValuePayload,
        ValueKind.TIME: TimeValuePayload,
        ValueKind.LITERAL: LiteralValuePayload,
    }
    if value.payload is None:
        raise VerificationError(
            f"value {value.id} is missing {value.kind.value} payload"
        )
    if not isinstance(value.payload, expected_types[value.kind]):
        raise VerificationError(
            f"value {value.id} has payload that does not match {value.kind.value}"
        )
    if value.kind == ValueKind.TIME and isinstance(value.payload, TimeValuePayload):
        if not value.payload.intent:
            return
        try:
            validate_time_intent(value.payload.intent)
        except ValueError as exc:
            raise VerificationError(
                f"value {value.id} has invalid time intent"
            ) from exc
