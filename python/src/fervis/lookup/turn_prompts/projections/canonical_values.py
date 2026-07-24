"""Model-facing projection of certified input values."""

from __future__ import annotations

from decimal import Decimal

from fervis.lookup.answer_program.values import (
    IdentitySetValuePayload,
    IdentityValuePayload,
)
from fervis.lookup.canonical_data import EntityKeyValue
from fervis.lookup.grounding import CanonicalInputValue


def canonical_values_prompt_items(
    values: tuple[CanonicalInputValue, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_value_id": item.canonical_value_id,
            "input_ref": item.input_ref,
            "use_refs": list(item.use_refs),
            "kind": item.typed_value.kind.value,
            "label": item.typed_value.label,
            "value": prompt_value(item.typed_value.payload.canonical_value()),
            "identity": _identity_payload(item.typed_value.payload),
            "certification_refs": list(item.certification_refs),
        }
        for item in values
    )


def _identity_payload(payload) -> dict[str, object] | None:
    keys: tuple[EntityKeyValue, ...]
    if isinstance(payload, IdentityValuePayload):
        keys = (payload.key,)
    elif isinstance(payload, IdentitySetValuePayload):
        keys = payload.keys
    else:
        return None
    return {
        "entity_kind": keys[0].entity_kind,
        "key_id": keys[0].key_id,
        "components": [
            {
                "component_id": component.component_id,
                "values": [
                    prompt_value(key.component_value(component.component_id))
                    for key in keys
                ],
            }
            for component in keys[0].components
        ],
    }


def prompt_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: prompt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [prompt_value(item) for item in value]
    if hasattr(value, "component_values"):
        return prompt_value(value.component_values())
    return value


__all__ = ["canonical_values_prompt_items", "prompt_value"]
