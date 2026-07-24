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
    TimeValuePayload,
    ValueProjectionKind,
    project_fact_value,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.relation_catalog.parameter_values import (
    parse_catalog_parameter_value,
)
from fervis.lookup.relation_catalog.model import EntityKeyComponentTarget


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

    return fact_value_parameter_projection(
        value,
        projection=ValueProjectionKind.IDENTITY_COMPONENT,
        component_id=component_id,
        type_name=type_name,
        choices=choices,
    )


def fact_value_parameter_projection(
    value: FactValue,
    *,
    projection: ValueProjectionKind,
    component_id: str | None,
    type_name: str,
    choices: tuple[str, ...],
) -> RuntimeValue:
    """Project and validate one value against a declared parameter contract."""

    projected = project_fact_value(
        value,
        projection=projection,
        component_id=component_id,
    )
    if isinstance(projected, tuple) and type_name not in {"array", "list"}:
        return tuple(
            parse_catalog_parameter_value(
                _parameter_wire_value(item),
                type_name=type_name,
                choices=choices,
            )
            for item in projected
        )
    return parse_catalog_parameter_value(
        _parameter_wire_value(projected),
        type_name=type_name,
        choices=choices,
    )


def compatible_fact_value_projections(
    value: FactValue,
    *,
    type_name: str,
    choices: tuple[str, ...],
    entity_target: EntityKeyComponentTarget | None,
) -> tuple[tuple[ValueProjectionKind, str | None], ...]:
    """Return intrinsic value projections accepted by one declared parameter."""

    payload = value.payload
    candidates: tuple[tuple[ValueProjectionKind, str | None], ...]
    if isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload)):
        if (
            entity_target is None
            or entity_target.entity_kind != payload.entity_kind
            or entity_target.key_id != payload.key_id
        ):
            return ()
        candidates = (
            (ValueProjectionKind.IDENTITY_COMPONENT, entity_target.component_id),
        )
    elif isinstance(payload, TimeValuePayload):
        candidates = (
            (ValueProjectionKind.TEMPORAL_START, None),
            (ValueProjectionKind.TEMPORAL_END, None),
        )
    else:
        candidates = ((ValueProjectionKind.WHOLE_VALUE, None),)
    compatible: list[tuple[ValueProjectionKind, str | None]] = []
    for projection, component_id in candidates:
        try:
            fact_value_parameter_projection(
                value,
                projection=projection,
                component_id=component_id,
                type_name=type_name,
                choices=choices,
            )
        except ValueError:
            continue
        compatible.append((projection, component_id))
    return tuple(compatible)


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
    "compatible_fact_value_projections",
    "fact_value_parameter_projection",
    "identity_key_component_ids",
    "identity_parameter_component_value",
    "identity_value_matches_entity_target",
]
