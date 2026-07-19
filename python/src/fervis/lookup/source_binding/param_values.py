"""Canonical source-binding parameter value helpers."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.relation_catalog.parameter_values import (
    parse_catalog_parameter_value,
)


def canonical_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def identity_key_component_ids(value: FactValue) -> tuple[str, ...]:
    """Return the complete key-component shape shared by all identity members."""

    payload = value.payload
    if isinstance(payload, IdentityValuePayload):
        return tuple(component.component_id for component in payload.key.components)
    if not isinstance(payload, IdentitySetValuePayload):
        return ()
    first_key = payload.keys[0]
    return tuple(
        component.component_id
        for component in first_key.components
        if all(
            component.component_id
            in {item.component_id for item in key.components}
            for key in payload.keys
        )
    )


def identity_value_matches_entity_target(
    value: FactValue,
    *,
    entity_kind: str,
    key_id: str,
    component_id: str,
) -> bool:
    """Check one declared entity target against an identity's complete key shape."""

    payload = value.payload
    if not isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload)):
        return False
    return (
        payload.entity_kind == entity_kind
        and payload.key_id == key_id
        and component_id in identity_key_component_ids(value)
    )


def identity_parameter_component_value(
    value: FactValue,
    *,
    component_id: str,
    type_name: str,
    choices: tuple[str, ...],
) -> RuntimeValue:
    """Project and validate one identity component for a declared parameter."""

    component_value = value.identity_key_component(component_id)
    if isinstance(component_value, tuple) and type_name not in {"array", "list"}:
        return tuple(
            parse_catalog_parameter_value(
                _parameter_wire_value(item),
                type_name=type_name,
                choices=choices,
            )
            for item in component_value
        )
    return parse_catalog_parameter_value(
        _parameter_wire_value(component_value),
        type_name=type_name,
        choices=choices,
    )


def compatible_identity_parameter_component_ids(
    value: FactValue,
    *,
    type_name: str,
    choices: tuple[str, ...],
) -> tuple[str, ...]:
    """Return identity components accepted by one declared parameter contract."""

    accepted: list[str] = []
    for component_id in identity_key_component_ids(value):
        try:
            identity_parameter_component_value(
                value,
                component_id=component_id,
                type_name=type_name,
                choices=choices,
            )
        except ValueError:
            continue
        accepted.append(component_id)
    return tuple(accepted)


def _parameter_wire_value(value: RuntimeValue) -> object:
    if isinstance(value, Decimal | UUID):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


__all__ = [
    "canonical_param_value",
    "compatible_identity_parameter_component_ids",
    "identity_key_component_ids",
    "identity_parameter_component_value",
    "identity_value_matches_entity_target",
]
