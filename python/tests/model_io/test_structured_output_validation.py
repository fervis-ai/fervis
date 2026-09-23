from jsonschema import validate

from fervis.model_io.structured_output.validation import strip_null_properties


def test_strip_null_properties_selects_array_branch_under_one_of():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "oneOf": [
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"enum": ["alpha"]},
                                "optional": {
                                    "type": "object",
                                    "properties": {"value": {"type": "string"}},
                                    "required": ["value"],
                                },
                            },
                            "required": ["kind"],
                        },
                    },
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"enum": ["beta"]},
                                "optional": {
                                    "type": "object",
                                    "properties": {"value": {"type": "string"}},
                                    "required": ["value"],
                                },
                            },
                            "required": ["kind"],
                        },
                    },
                ],
            },
        },
        "required": ["items"],
    }
    payload = {"items": [{"kind": "beta", "optional": None}]}

    normalized = strip_null_properties(payload, schema=schema)

    assert normalized == {"items": [{"kind": "beta"}]}
    validate(instance=normalized, schema=schema)


def test_required_nullable_enum_keeps_its_explicit_null_decision():
    schema = {'type': 'object', 'properties': {'source_ref': {'enum': ['source_1', None]}}, 'required': ['source_ref'], 'additionalProperties': False}
    assert strip_null_properties({'source_ref': None}, schema=schema) == {'source_ref': None}
