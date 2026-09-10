"""Inject a graph-owned related-entity output into production."""

from __future__ import annotations

from copy import deepcopy


_OUTPUT_REF = "output_1"
_OLD_INSTRUCTION = (
    "Each requested_value_meaning is another value the answer must state. Write its\n"
    "expression in requested_value_outputs. A requested value may itself be an\n"
    "entity identity. Preserve the shown order within each output list."
)
_OLD_AUTHORING_ORDER = (
    "Within each item, write\n"
    "candidate_set and set_graph before grouping."
)
_NEW_AUTHORING_ORDER = (
    "Within each item, write outputs before candidate_set and set_graph. A "
    "related-entity output establishes its returned_entity_ref before set_graph "
    "copies that ref onto the set reached by the relationship."
)
_NEW_INSTRUCTION = (
    "Each requested_value_meaning is another value the answer must state. Each "
    "requested_value_outputs item writes output_kind_basis, then output_kind. "
    "related_entity means the answer states which related person, organization, "
    "place, product, or other entity is involved. It writes returned_entity_ref "
    "and origin; the one set_graph set carrying that same returned_entity_ref owns "
    "the relationship. value means the answer states an attribute, measurement, "
    "status, time, quantity, or computed value; it writes expression. Preserve "
    "the shown order within each output list."
)


def transform(boundary: dict[str, object]) -> dict[str, object]:
    transformed = deepcopy(boundary)
    prompt = transformed.get("prompt")
    if not isinstance(prompt, str) or prompt.count(_OLD_INSTRUCTION) != 1:
        raise ValueError("question-contract requested-output instruction drifted")
    transformed["prompt"] = prompt.replace(_OLD_INSTRUCTION, _NEW_INSTRUCTION)
    transformed_prompt = transformed["prompt"]
    if (
        not isinstance(transformed_prompt, str)
        or transformed_prompt.count(_OLD_AUTHORING_ORDER) != 1
    ):
        raise ValueError("question-contract authoring order drifted")
    transformed["prompt"] = transformed_prompt.replace(
        _OLD_AUTHORING_ORDER,
        _NEW_AUTHORING_ORDER,
    )
    tool_specs = transformed.get("tool_specs")
    if not isinstance(tool_specs, list) or len(tool_specs) != 1:
        raise ValueError("question-contract boundary must expose one tool")
    tool_spec = tool_specs[0]
    if not isinstance(tool_spec, dict):
        raise ValueError("question-contract tool spec must be an object")
    schema = tool_spec.get("input_schema")
    output_count = _replace_requested_output_items(schema)
    graph_count = _add_graph_output_owner(schema)
    if output_count != 1 or graph_count != 1:
        raise ValueError(
            f"expected one output and graph schema, found {output_count}/{graph_count}"
        )
    return transformed


def _replace_requested_output_items(value: object) -> int:
    if isinstance(value, list):
        return sum(_replace_requested_output_items(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 0
    properties = value.get("properties")
    if isinstance(properties, dict):
        requested = properties.get("requested_value_outputs")
        if isinstance(requested, dict):
            current = requested.get("items")
            if not isinstance(current, dict):
                raise ValueError("requested output items must be an object schema")
            expression = current.get("properties", {}).get("expression")
            if not isinstance(expression, dict):
                raise ValueError("requested output expression schema is missing")
            alternatives = expression.get("anyOf")
            if not isinstance(alternatives, list) or len(alternatives) != 4:
                raise ValueError("requested output expression alternatives drifted")
            requested["items"] = {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "output_kind_basis": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "output_kind": {"enum": ["related_entity"]},
                            "returned_entity_ref": {"enum": [_OUTPUT_REF]},
                            "origin": {"$ref": "#/$defs/source_origin"},
                        },
                        "required": [
                            "output_kind_basis",
                            "output_kind",
                            "returned_entity_ref",
                            "origin",
                        ],
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "output_kind_basis": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "output_kind": {"enum": ["value"]},
                            "expression": {"anyOf": deepcopy(alternatives[:2])},
                        },
                        "required": [
                            "output_kind_basis",
                            "output_kind",
                            "expression",
                        ],
                    },
                ]
            }
            count += 1
    return count + sum(_replace_requested_output_items(item) for item in value.values())


def _add_graph_output_owner(value: object) -> int:
    if isinstance(value, list):
        return sum(_add_graph_output_owner(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 0
    properties = value.get("properties")
    if isinstance(properties, dict):
        set_graph = properties.get("set_graph")
        outputs = properties.get("outputs")
        if isinstance(set_graph, dict) and isinstance(outputs, dict):
            value["properties"] = {
                key: properties[key]
                for key in (
                    "requested_fact_ref",
                    "origin",
                    "outputs",
                    "candidate_set",
                    "set_graph",
                    "grouping",
                    "qualification",
                    "ordering",
                    "selection",
                    "distinct_by",
                )
                if key in properties
            }
            required = value.get("required")
            if isinstance(required, list):
                value["required"] = [
                    key for key in value["properties"] if key in required
                ]
            count += 1
    set_schema = _related_set_declaration(value)
    if set_schema is not None:
        _allow_returned_entity_owner(set_schema)
    return count + sum(_add_graph_output_owner(item) for item in value.values())


def _related_set_declaration(value: dict[str, object]) -> dict[str, object] | None:
    properties = value.get("properties")
    if not isinstance(properties, dict):
        return None
    if set(properties) != {"association", "set", "related_sets"}:
        return None
    set_schema = properties.get("set")
    return set_schema if isinstance(set_schema, dict) else None


def _allow_returned_entity_owner(set_schema: dict[str, object]) -> None:
    original = deepcopy(set_schema)
    owner = deepcopy(set_schema)
    if "oneOf" in owner:
        raise ValueError("output-owner experiment expects an ordinary set schema")
    properties = owner.get("properties")
    required = owner.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("set declaration schema is not closed")
    owner["properties"] = {
        "id": properties["id"],
        "origin": properties["origin"],
        "returned_entity_ref": {"enum": [_OUTPUT_REF]},
    }
    owner["required"] = ["id", "origin", "returned_entity_ref"]
    set_schema.clear()
    set_schema["oneOf"] = [original, owner]


__all__ = ["transform"]
