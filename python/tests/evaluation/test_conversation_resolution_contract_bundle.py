from __future__ import annotations

from fervis.memory.conversation_context import (
    ConversationContextSource,
    ConversationMeaningAnchor,
)
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    ConversationResolutionRequest,
    ConversationResolutionTurnPrompt,
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


def test_conversation_resolution_contract_bundle_exposes_clause_resolution_contract():
    prompt = ConversationResolutionTurnPrompt(
        question="And how much did she make yesterday?",
        context_sources=(_context_source(),),
        conversation_context={},
    )
    invocation = prompt.to_model_invocation()
    schema_text = repr(invocation.provider_schema)

    assert {
        "prompt_terms_present": all(
            term in invocation.prompt_text
            for term in (
                "context_sources",
                "Available context frames:",
                "clause_resolutions",
                "requested_value_frame.context_frame_choices",
                "Return a conversation resolution result for the current user utterance",
                "context_frame_choices must include one choice item",
                "resolved_clause_text rewrites current_clause_text as a standalone clause",
                "current_clause_text is exact text copied from the current user question",
            )
        ),
        "retired_terms_absent": all(
            term not in invocation.prompt_text
            for term in ("context-light", "abstract conversation frame")
        ),
        "tool_names": set(invocation.provider_schema),
        "schema_terms_present": all(
            term in schema_text for term in ("clause_resolutions", "unresolved")
        ),
        "retired_integrated_question_absent": "integrated_question"
        not in invocation.provider_schema[CONVERSATION_RESOLUTION_TOOL_NAME][
            "properties"
        ],
    } == {
        "prompt_terms_present": True,
        "retired_terms_absent": True,
        "tool_names": {CONVERSATION_RESOLUTION_TOOL_NAME},
        "schema_terms_present": True,
        "retired_integrated_question_absent": True,
    }


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
            if not isinstance(items, dict) or not (
                "type" in items or "oneOf" in items
            ):
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
            "answer": (
                '{"tool": "submit_conversation_resolution", '
                '"arguments": {'
                '"kind": "conversation_resolution", '
                '"status": "resolved", '
                '"current_question_text": "What quantities were those?", '
                '"clause_resolutions": [{'
                '"current_clause_text": "What quantities were those", '
                '"occurrence": 1, '
                '"requested_value_frame": {'
                '"current_value_surface": {'
                '"text": "quantities", '
                '"kind": "self_sufficient_current_value"'
                "}, "
                '"context_frame_choices": []'
                "}, "
                '"dependencies": [{'
                '"anchor_text": "those", '
                '"occurrence": 1, '
                '"kind": "reference", '
                '"meaning_components": [{'
                '"kind": "other", '
                '"source_id": "prior_1", '
                '"source_text": "prior rows", '
                '"memory_id": "mem_1", '
                '"resolved_text": "prior rows"'
                "}], "
                '"resolved_text": "prior rows", '
                '"must_preserve_terms": ["prior rows"]'
                "}], "
                '"resolved_clause_text": "What quantities were in the prior rows?"'
                "}], "
                '"unresolved": {'
                '"unresolved_kind": "none", '
                '"why_unresolved": "", '
                '"candidate_interpretations": []'
                "}"
                "}}"
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
        "derived_memory_ids": result.artifact.derived_payload[
            "activated_memory_ids"
        ],
    } == {
        "submitted_has_derived_keys": False,
        "parsed_has_derived_keys": False,
        "derived_source_card_ids": ["card_1"],
        "derived_memory_ids": ["mem_1"],
    }
