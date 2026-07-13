from __future__ import annotations

import json

from fervis.memory.conversation_context import (
    ConversationContextSource,
    ConversationMeaningAnchor,
)
from fervis.lookup.conversation_resolution import (
    ConversationResolutionRequest,
    generate_conversation_resolution,
)
from fervis.lookup.conversation_resolution.schema import (
    build_conversation_resolution_tool_schemas,
)


def _context_source() -> ConversationContextSource:
    return ConversationContextSource(
        source_id="prior_1",
        kind="prior_user_question",
        text="How much is that in total sales?",
        source_card_ids=("card_1",),
        source_memory_ids=("mem_1",),
        meaning_anchors=(
            ConversationMeaningAnchor(
                memory_id="mem_1",
                text="total sales",
                occurrence=1,
                kind="scalar_value",
                label="scalar value",
            ),
        ),
    )


def test_conversation_resolution_schemas_are_provider_compatible_root_objects():
    schemas = build_conversation_resolution_tool_schemas(
        context_sources=(_context_source(),),
    )

    violations = []
    for schema_name, schema in schemas.items():
        if schema["type"] != "object":
            violations.append(f"{schema_name}:root_not_object")
        for field_name, field_schema in schema.get("properties", {}).items():
            if "type" not in field_schema:
                violations.append(f"{schema_name}.{field_name}:missing_type")
            elif field_schema["type"] == "array" and "items" not in field_schema:
                violations.append(f"{schema_name}.{field_name}:missing_items")

    assert violations == []


def test_conversation_resolution_array_items_are_typed_for_provider_schemas():
    schemas = build_conversation_resolution_tool_schemas(
        context_sources=(_context_source(),),
    )

    assert [
        path
        for schema_name, schema in schemas.items()
        for path in _array_item_type_violations(schema, path=schema_name)
    ] == []


def _array_item_type_violations(schema: object, *, path: str) -> list[str]:
    violations = []
    if isinstance(schema, dict):
        if schema.get("type") == "array":
            items = schema.get("items")
            if not isinstance(items, dict) or not ("type" in items or "oneOf" in items):
                violations.append(path)
        for key, value in schema.items():
            violations.extend(_array_item_type_violations(value, path=f"{path}.{key}"))
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            violations.extend(
                _array_item_type_violations(item, path=f"{path}[{index}]")
            )
    return violations


class _ConversationResolutionModelPort:
    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode=None,
        tool_specs=(),
    ):
        del (
            provider,
            prompt,
            max_thinking_tokens,
            system_prompt,
            output_mode,
            tool_specs,
        )
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_conversation_resolution",
                    "arguments": {
                        "kind": "conversation_resolution",
                        "current_question_text": "What quantities were those?",
                        "outcome": {
                            "kind": "resolved",
                            "resolution_basis": (
                                "The prior rows supply the referent omitted by the "
                                "current question."
                            ),
                            "contextualized_question": (
                                "What quantities were in the prior rows?"
                            ),
                            "clauses": [
                                {
                                    "current_clause_text": (
                                        "What quantities were those?"
                                    ),
                                    "occurrence": 1,
                                    "resolved_text": (
                                        "What quantities were in the prior rows?"
                                    ),
                                    "retained_frame_parts": [],
                                    "values": [
                                        {
                                            "value_id": "prior_rows",
                                            "resolved_text": "prior rows",
                                            "frame_parameter": {"kind": "none"},
                                            "sources": [
                                                {
                                                    "kind": "current_span",
                                                    "text": "those",
                                                    "occurrence": 1,
                                                },
                                                {
                                                    "kind": "context_anchor",
                                                    "source_id": "prior_1",
                                                    "memory_id": "mem_1",
                                                    "source_text": "prior rows",
                                                },
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                }
            ),
            "usage": {"inputTokens": 1, "outputTokens": 1, "thinkingTokens": 0},
        }


def test_conversation_resolution_artifact_separates_submitted_and_derived_payloads():
    result = generate_conversation_resolution(
        request=ConversationResolutionRequest(
            question="What quantities were those?",
            conversation_context={},
            context_sources=(
                ConversationContextSource(
                    source_id="prior_1",
                    kind="prior_fervis_answer",
                    text="prior rows",
                    source_card_ids=("card_1",),
                    source_memory_ids=("mem_1",),
                    meaning_anchors=(
                        ConversationMeaningAnchor(
                            memory_id="mem_1",
                            text="prior rows",
                            occurrence=1,
                            kind="row_set",
                            label="row set",
                        ),
                    ),
                ),
            ),
        ),
        model_port=_ConversationResolutionModelPort(),
        provider="fake",
        model_key="FAKE",
        max_thinking_tokens=1,
    )

    assert {
        "submitted_has_derived_keys": any(
            key in result.artifact.submitted_payload
            for key in ("used_source_card_ids", "activated_memory_ids")
        ),
        "parsed_has_derived_keys": any(
            key in result.artifact.parsed_payload
            for key in ("used_source_card_ids", "activated_memory_ids")
        ),
        "derived_source_card_ids": result.artifact.derived_payload[
            "used_source_card_ids"
        ],
        "derived_memory_ids": result.artifact.derived_payload["activated_memory_ids"],
    } == {
        "submitted_has_derived_keys": False,
        "parsed_has_derived_keys": False,
        "derived_source_card_ids": ["card_1"],
        "derived_memory_ids": ["mem_1"],
    }
