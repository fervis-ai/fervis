from fervis.model_io.structured_output.schema import (
    without_unreferenced_definitions,
)


def test_unreferenced_definitions_are_removed_without_mutating_the_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/reachable"}},
        "$defs": {
            "reachable": {"$ref": "#/$defs/transitive"},
            "transitive": {"type": "string"},
            "unused": {"type": "integer"},
        },
    }

    pruned = without_unreferenced_definitions(schema)

    assert tuple(pruned["$defs"]) == ("reachable", "transitive")
    assert tuple(schema["$defs"]) == ("reachable", "transitive", "unused")
