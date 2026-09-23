"""Mechanical JSON Schema transforms."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def without_unreferenced_definitions(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Return the same schema without unreachable local ``$defs`` entries."""
    output = deepcopy(schema)
    definitions = output.get("$defs")
    if not isinstance(definitions, dict):
        return output

    reachable: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        reference = value.get("$ref")
        prefix = "#/$defs/"
        if isinstance(reference, str) and reference.startswith(prefix):
            name = reference.removeprefix(prefix).replace("~1", "/").replace("~0", "~")
            if name not in reachable:
                reachable.add(name)
                visit(definitions.get(name))
        for key, item in value.items():
            if key != "$defs":
                visit(item)

    visit(output)
    output["$defs"] = {
        name: value for name, value in definitions.items() if name in reachable
    }
    return output


__all__ = ["without_unreferenced_definitions"]
