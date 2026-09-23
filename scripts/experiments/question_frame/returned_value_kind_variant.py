"""Inject the proposed returned-value role into the production frame boundary."""

from __future__ import annotations

from copy import deepcopy


_OLD_INSTRUCTION = (
    "Each explicitly_requested_values item\n"
    "writes value_ref, meaning, and origin once."
)
_NEW_INSTRUCTION = (
    "Each explicitly_requested_values item writes value_ref, then value_kind_basis, "
    "value_kind, meaning, and origin once. value_kind_basis states whether the "
    "requested answer states a related entity or a value. related_entity means "
    "the answer states which related person, organization, place, product, or "
    "other entity is involved. value "
    "means the answer states an attribute, measurement, status, time, quantity, "
    "or computed value without identifying an entity."
)


def transform(boundary: dict[str, object]) -> dict[str, object]:
    transformed = deepcopy(boundary)
    prompt = transformed.get("prompt")
    if not isinstance(prompt, str) or prompt.count(_OLD_INSTRUCTION) != 1:
        raise ValueError("question-frame returned-value instruction drifted")
    transformed["prompt"] = prompt.replace(_OLD_INSTRUCTION, _NEW_INSTRUCTION)
    tool_specs = transformed.get("tool_specs")
    if not isinstance(tool_specs, list) or len(tool_specs) != 1:
        raise ValueError("question-frame boundary must expose one tool")
    tool_spec = tool_specs[0]
    if not isinstance(tool_spec, dict):
        raise ValueError("question-frame tool spec must be an object")
    count = _add_value_kind(tool_spec.get("input_schema"))
    if count != 3:
        raise ValueError(f"expected three projection branches, found {count}")
    return transformed


def _add_value_kind(value: object) -> int:
    if isinstance(value, list):
        return sum(_add_value_kind(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 0
    properties = value.get("properties")
    if isinstance(properties, dict):
        requested = properties.get("explicitly_requested_values")
        if isinstance(requested, dict):
            items = requested.get("items")
            if not isinstance(items, dict):
                raise ValueError("explicit requested values need an item schema")
            item_properties = items.get("properties")
            required = items.get("required")
            if not isinstance(item_properties, dict) or not isinstance(required, list):
                raise ValueError("returned-value item schema is not closed")
            items["properties"] = {
                "value_ref": item_properties["value_ref"],
                "value_kind_basis": {"type": "string", "minLength": 1},
                "value_kind": {
                    "enum": ["related_entity", "value"]
                },
                "meaning": item_properties["meaning"],
                "origin": item_properties["origin"],
            }
            items["required"] = [
                "value_ref",
                "value_kind_basis",
                "value_kind",
                "meaning",
                "origin",
            ]
            count += 1
    return count + sum(_add_value_kind(item) for item in value.values())


__all__ = ["transform"]
