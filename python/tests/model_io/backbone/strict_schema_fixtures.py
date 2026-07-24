from fervis.model_io.backbone.dto import ToolSpec


def nested_union_tool_spec() -> ToolSpec:
    """Return a domain-neutral strict nested union for adapter projections."""

    def branch(branch_id: str) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "branch_id": {"enum": [branch_id]},
                "mapping_basis": {"type": "string", "minLength": 1},
                "source_ref": {"type": "string", "minLength": 1},
            },
            "required": ["branch_id", "mapping_basis", "source_ref"],
        }

    return ToolSpec(
        name="submit_nested_union",
        description="Submit one strict nested-union value.",
        strict=True,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision_basis": {"type": "string", "minLength": 1},
                "binding": {"oneOf": [branch("branch_1"), branch("branch_2")]},
            },
            "required": ["decision_basis", "binding"],
        },
    )
