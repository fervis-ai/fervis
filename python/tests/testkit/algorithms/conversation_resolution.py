from __future__ import annotations

from typing import Any

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
    ConversationMeaningAnchor,
)
from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    LiteralQuestionInputOverlay,
    NamedReferenceQuestionInputOverlay,
    ResolvedCanonicalValueOverlay,
    ResolvedQuestionInputOverlay,
    RowSetQuestionInputOverlay,
    parse_conversation_resolution,
)
from fervis.lookup.conversation_resolution.schema import (
    build_conversation_resolution_tool_schemas,
)

from tests.testkit.assertions import exact_mismatches, subset_mismatches


def run_conversation_resolution_parse_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload=dict(input_payload["payload"]),
        current_question=str(input_payload["current_question"]),
        context_sources=tuple(
            _context_source(item) for item in input_payload.get("context_sources") or ()
        ),
        context_frames=tuple(
            _context_frame(item) for item in input_payload.get("context_frames") or ()
        ),
    )
    actual = {
        "resolution": result.outcome.resolution.value,
        "current_question_text": result.outcome.current_question_text,
        "used_source_card_ids": list(result.outcome.used_source_card_ids),
        "used_memory_ids": list(result.outcome.used_memory_ids),
        "clause_resolutions": [
            {
                "current_clause_text": clause.current_clause_text,
                "resolved_clause_text": clause.resolved_clause_text,
                "dependencies": [
                    {
                        "anchor_text": dependency.anchor_text,
                        "resolved_text": dependency.resolved_text,
                        "meaning_components": [
                            {
                                "source_id": component.source_id,
                                "source_text": component.source_text,
                                "memory_id": component.memory_id,
                                "resolved_text": component.resolved_text,
                            }
                            for component in dependency.meaning_components
                        ],
                    }
                    for dependency in clause.dependencies
                ],
            }
            for clause in result.outcome.clause_resolutions
        ],
    }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_conversation_resolution_schema_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload.get("input") or {}
    schemas = build_conversation_resolution_tool_schemas(
        context_sources=tuple(
            _context_source(item) for item in input_payload.get("context_sources") or ()
        ),
        context_frames=tuple(
            _context_frame(item) for item in input_payload.get("context_frames") or ()
        ),
    )
    schema = schemas[CONVERSATION_RESOLUTION_TOOL_NAME]
    clause_schema = schema["properties"]["clause_resolutions"]["items"]
    dependency_schema = clause_schema["properties"]["dependencies"]["items"]
    no_context_dependencies_schema = build_conversation_resolution_tool_schemas()[
        CONVERSATION_RESOLUTION_TOOL_NAME
    ]["properties"]["clause_resolutions"]["items"]["properties"]["dependencies"]
    actual = {
        "tool_names": sorted(schemas),
        "status_values": schema["properties"]["status"]["enum"],
        "has_clause_resolutions": "clause_resolutions" in schema["properties"],
        "dependency_source_ids": dependency_schema["properties"][
            "meaning_components"
        ]["items"]["properties"]["source_id"]["enum"],
        "unresolved_evidence_source_ids": schema["properties"]["unresolved"][
            "properties"
        ]["candidate_interpretations"]["items"]["properties"]["supporting_evidence"][
            "items"
        ]["properties"]["source_id"]["enum"],
        "no_context_dependency_max_items": no_context_dependencies_schema["maxItems"],
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_conversation_resolution_overlay_case(payload: dict[str, Any]) -> list[str]:
    try:
        resolved_inputs = tuple(
            _resolved_question_input_overlay(item)
            for item in payload["input"].get("resolved_question_inputs") or ()
        )
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    actual = {
        "prompt_resolved_question_inputs": [
            item.to_prompt_payload() for item in resolved_inputs
        ],
        "backend_resolved_question_inputs": [
            item.to_backend_payload() for item in resolved_inputs
        ],
    }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _resolved_question_input_overlay(
    item: dict[str, Any],
) -> ResolvedQuestionInputOverlay:
    kind = str(item["kind"])
    if kind == "literal_text":
        return LiteralQuestionInputOverlay(
            source_text=str(item["source_text"]),
            occurrence=int(item.get("occurrence") or 1),
            resolved_input_ref=str(item["resolved_input_ref"]),
            resolved_value_text=str(item["resolved_value_text"]),
            value_meaning_hint=str(item.get("value_meaning_hint") or ""),
            field_label_text=str(item.get("field_label_text") or ""),
            role=str(item["role"]),
            evidence_refs=tuple(str(ref) for ref in item.get("evidence_refs") or ()),
            resolved_canonical_value=_resolved_canonical_value_overlay(
                item.get("resolved_canonical_value")
            ),
        )
    if kind == "row_set_reference":
        return RowSetQuestionInputOverlay(
            reference_text=str(item["reference_text"]),
            occurrence=int(item.get("occurrence") or 1),
            resolved_input_ref=str(item["resolved_input_ref"]),
            memory_ids=tuple(str(ref) for ref in item.get("memory_ids") or ()),
        )
    return NamedReferenceQuestionInputOverlay(
        reference_text=str(item["reference_text"]),
        occurrence=int(item.get("occurrence") or 1),
        target_meaning=str(item["target_meaning"]),
        lookup_text=str(item["lookup_text"]),
        resolved_input_ref=str(item.get("resolved_input_ref") or ""),
        memory_ids=tuple(str(ref) for ref in item.get("memory_ids") or ()),
    )


def _resolved_canonical_value_overlay(
    raw: object,
) -> ResolvedCanonicalValueOverlay | None:
    if not isinstance(raw, dict):
        return None
    return ResolvedCanonicalValueOverlay(
        kind=str(raw["kind"]),
        identity_type=str(raw["identity_type"]),
        identity_field=str(raw["identity_field"]),
        value=str(raw["value"]),
        proof_refs=tuple(str(ref) for ref in raw.get("proof_refs") or ()),
    )


def _context_source(payload: dict[str, Any]) -> ConversationContextSource:
    return ConversationContextSource(
        source_id=str(payload["source_id"]),
        kind=str(payload["kind"]),
        text=str(payload["text"]),
        source_card_ids=tuple(payload.get("source_card_ids") or ()),
        source_memory_ids=tuple(payload.get("source_memory_ids") or ()),
        meaning_anchors=tuple(
            ConversationMeaningAnchor(
                memory_id=str(anchor["memory_id"]),
                text=str(anchor["text"]),
                occurrence=int(anchor.get("occurrence") or 1),
                kind=str(anchor.get("kind") or "other"),
                label=str(anchor.get("label") or "prior meaning"),
            )
            for anchor in payload.get("meaning_anchors") or ()
        ),
    )


def _context_frame(payload: dict[str, Any]) -> ConversationContextFrame:
    return ConversationContextFrame(
        frame_id=str(payload["frame_id"]),
        source_ids=tuple(payload.get("source_ids") or ()),
        requested_frame=str(payload["requested_frame"]),
        prior_answer_fact=str(payload.get("prior_answer_fact") or ""),
    )
