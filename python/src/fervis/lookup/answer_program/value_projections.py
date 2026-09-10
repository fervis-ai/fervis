"""Typed projections of certified values, shared by authoring and API verification."""
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID
from fervis.lookup.answer_program.values import (
    IdentityValuePayload, IdentitySetValuePayload, TimeValuePayload,
    LiteralValuePayload, NamedValuePayload, StringSetValuePayload,
)


def _projection_type(payload, component):
    if isinstance(payload, (IdentitySetValuePayload, StringSetValuePayload)):
        return "array"
    if isinstance(payload, IdentityValuePayload):
        keys = (
            (payload.key,)
            if isinstance(payload, IdentityValuePayload)
            else payload.keys
        )
        kinds = {
            _primitive_type(
                next(
                    item.value
                    for item in key.components
                    if item.component_id == component.split(":", 1)[1]
                )
            )
            for key in keys
        }
        return next(iter(kinds)) if len(kinds) == 1 else "unknown"
    if isinstance(payload, TimeValuePayload):
        return "datetime" if payload.granularity == "hour" else "date"
    if isinstance(payload, LiteralValuePayload):
        return payload.literal_type.value
    return "string"


def _primitive_type(value):
    if isinstance(value, UUID):
        return "uuid"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if isinstance(value, (Decimal, float)):
        return "number"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, str):
        return "string"
    return "unknown"


def identity_contract(payload):
    if isinstance(payload, IdentityValuePayload):
        key = payload.key
    elif isinstance(payload, IdentitySetValuePayload):
        key = payload.keys[0]
    else:
        return None
    return {
        "entity_kind": key.entity_kind,
        "key_id": key.key_id,
        "components": [item.component_id for item in key.components],
    }


def projection_description(value, component, item_index=None):
    payload = value.payload
    identity = identity_contract(payload)
    original_component = component
    if identity and component == "value" and len(identity["components"]) == 1:
        component = "key_component:" + identity["components"][0]
    kind = _projection_type(payload, component)
    if identity and original_component == "value":
        kind = "array" if isinstance(payload, IdentitySetValuePayload) else "string"
    if item_index is not None and isinstance(payload, IdentitySetValuePayload):
        kind = _primitive_type(
            payload.keys[item_index].component_value(component.split(":", 1)[1])
        )
    elif item_index is not None and isinstance(payload, StringSetValuePayload):
        kind = "string"
    projected_values = ()
    if identity and component.startswith('key_component:'):
        keys = (payload.key,) if isinstance(payload, IdentityValuePayload) else payload.keys
        if item_index is not None:
            keys = (keys[item_index],)
        projected_values = tuple(key.component_value(component.split(':', 1)[1]) for key in keys)
    return {
        "kind": value.kind.value,
        "identity": identity,
        **({'_projected_values': projected_values} if projected_values else {}),
        "projection": component,
        "value_type": kind,
        **({"value": payload.value} if isinstance(payload, LiteralValuePayload) else
           {"value": payload.text} if isinstance(payload, NamedValuePayload) else
           {"value": payload.values[item_index]} if isinstance(payload, StringSetValuePayload) and item_index is not None else {}),
    }


