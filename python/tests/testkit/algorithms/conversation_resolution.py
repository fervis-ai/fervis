from __future__ import annotations

from typing import Any

from fervis.lookup.conversation_resolution import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
    compile_conversation_resolution,
    parse_conversation_resolution,
    ResolvedCanonicalIdentity,
    ResolvedLiteralQuestionInput,
    ResolvedRowSetQuestionInput,
)
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
    CompiledResolvedClause,
    CompiledResolvedValue,
)
from fervis.lookup.question_inputs import LiteralInputRole
from fervis.lookup.conversation_resolution.schema import (
    build_conversation_resolution_tool_schemas,
)
from fervis.memory.conversation_context import (
    ConversationAnswerShape,
    ConversationCallableSignature,
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFrameParameter,
    ConversationFramePart,
    ConversationFramePartKind,
    ConversationMeaningAnchor,
    ConversationMemoryCardProjection,
)

from tests.testkit.assertions import exact_mismatches, subset_mismatches


def run_conversation_resolution_parse_case(payload: dict[str, Any]) -> list[str]:
    resolution, _projection = _parse(payload["input"])
    actual = {
        **resolution.to_model_dict(),
        "used_source_card_ids": list(resolution.used_source_card_ids),
        "used_memory_ids": list(resolution.used_memory_ids),
    }
    return _mismatches(actual, payload["expect"])


def run_conversation_resolution_compile_case(payload: dict[str, Any]) -> list[str]:
    resolution, projection = _parse(payload["input"])
    compiled = compile_conversation_resolution(
        resolution,
        memory_projection=projection,
    )
    actual = {
        "current_question_text": compiled.current_question_text,
        "recorded_contextualized_question": compiled.contextualized_question,
        "question_contract_context": compiled.to_prompt_payload(),
        "frame_call": (
            compiled.frame_call.to_model_dict()
            if compiled.frame_call is not None
            else {"kind": "none"}
        ),
        "uses_prior_context": compiled.uses_prior_context,
        "canonical_identity_inputs": [
            {
                "input_ref": item.input_ref,
                "entity_kind": item.canonical_identity.entity_kind,
                "key_id": item.canonical_identity.key_id,
                "key_component_id": item.canonical_identity.key_component_id,
                "value": item.canonical_identity.value,
                "authority_refs": list(item.canonical_identity.authority_refs),
            }
            for item in compiled.identity_inputs()
        ],
    }
    return _mismatches(actual, payload["expect"])


def run_conversation_resolution_schema_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload.get("input") or {}
    schema = build_conversation_resolution_tool_schemas(
        context_sources=tuple(
            _context_source(item) for item in input_payload.get("context_sources") or ()
        ),
        context_frames=tuple(
            _context_frame(item) for item in input_payload.get("context_frames") or ()
        ),
    )[CONVERSATION_RESOLUTION_TOOL_NAME]
    properties = schema["properties"]
    outcome_branches = properties["outcome"]["oneOf"]
    resolved = outcome_branches[0]
    clause = resolved["properties"]["clauses"]["items"]
    source_branches = clause["properties"]["values"]["items"]["properties"]["sources"][
        "items"
    ]["oneOf"]
    parameter_branches = clause["properties"]["values"]["items"]["properties"][
        "frame_parameter"
    ]["oneOf"]
    ambiguity = outcome_branches[1]
    ambiguity_candidate = ambiguity["properties"]["candidate_interpretations"]["items"]
    ambiguity_candidate_fields = ambiguity_candidate["properties"]
    evidence_field = next(
        field_id
        for field_id in ambiguity_candidate_fields
        if field_id != "contextualized_question"
    )
    evidence_source_ids = ambiguity_candidate_fields[evidence_field]["items"][
        "properties"
    ]["source_id"]["enum"]
    actual = {
        "top_level_fields": sorted(properties),
        "top_level_required": sorted(schema["required"]),
        "outcome_kinds": [
            branch["properties"]["kind"]["enum"][0] for branch in outcome_branches
        ],
        "resolved_fields": sorted(resolved["properties"]),
        "resolved_required": sorted(resolved["required"]),
        "source_kinds": [
            branch["properties"]["kind"]["enum"][0] for branch in source_branches
        ],
        "frame_parameter_kinds": [
            branch["properties"]["kind"]["enum"][0]
            for branch in parameter_branches
        ],
        "ambiguity_candidate_fields": sorted(ambiguity_candidate_fields),
        "ambiguity_evidence_source_ids": evidence_source_ids,
    }
    return _mismatches(actual, payload["expect"])


def _parse(input_payload: dict[str, Any]):
    sources = tuple(
        _context_source(item) for item in input_payload.get("context_sources") or ()
    )
    frames = tuple(
        _context_frame(item) for item in input_payload.get("context_frames") or ()
    )
    result = parse_conversation_resolution(
        tool_name=CONVERSATION_RESOLUTION_TOOL_NAME,
        payload=dict(input_payload["payload"]),
        current_question=str(input_payload["current_question"]),
        context_sources=sources,
        context_frames=frames,
    )
    private_cards = {
        str(item["memory_id"]): {
            "kind": str(item["kind"]),
            **dict(item.get("private") or {}),
        }
        for item in input_payload.get("memories") or ()
    }
    return result.outcome, ConversationMemoryCardProjection(
        context_sources=sources,
        context_frames=frames,
        private_cards=private_cards,
    )


def _context_source(payload: dict[str, Any]) -> ConversationContextSource:
    return ConversationContextSource(
        source_id=str(payload["source_id"]),
        kind=str(payload["kind"]),
        text=str(payload["text"]),
        source_card_ids=tuple(
            str(item) for item in payload.get("source_card_ids") or ()
        ),
        source_memory_ids=tuple(
            str(item) for item in payload.get("source_memory_ids") or ()
        ),
        meaning_anchors=tuple(
            ConversationMeaningAnchor(
                memory_id=str(anchor["memory_id"]),
                text=str(anchor["text"]),
                occurrence=int(anchor.get("occurrence") or 1),
                kind=str(anchor["kind"]),
                label=str(anchor.get("label") or anchor["kind"]),
            )
            for anchor in payload.get("meaning_anchors") or ()
        ),
    )


def _context_frame(payload: dict[str, Any]) -> ConversationContextFrame:
    callable_payload = payload.get("callable")
    return ConversationContextFrame(
        frame_id=str(payload["frame_id"]),
        source_ids=tuple(str(item) for item in payload.get("source_ids") or ()),
        answer_shape=ConversationAnswerShape(
            expression_family=str(payload["answer_shape"]["expression_family"]),
            output_roles=tuple(
                str(item) for item in payload["answer_shape"]["output_roles"]
            ),
        ),
        parts=tuple(
            ConversationFramePart(
                part_id=str(part["part_id"]),
                kind=ConversationFramePartKind(str(part["kind"])),
                text=str(part["text"]),
                source_ref=str(part.get("source_ref") or ""),
            )
            for part in payload.get("parts") or ()
        ),
        callable=(
            ConversationCallableSignature(
                base_run_id=str(callable_payload["base_run_id"]),
                requested_fact_id=str(callable_payload["requested_fact_id"]),
                parameters=tuple(
                    ConversationFrameParameter(
                        parameter_id=str(item["parameter_id"]),
                        part_id=str(item["part_id"]),
                        kind=ConversationFramePartKind(str(item["kind"])),
                        current_text=str(item["current_text"]),
                        resolved_text=str(item["resolved_text"]),
                        field_label_text=str(item.get("field_label_text") or ""),
                        value_meaning_hint=str(item.get("value_meaning_hint") or ""),
                    )
                    for item in callable_payload.get("parameters") or ()
                ),
            )
            if isinstance(callable_payload, dict)
            else None
        ),
    )


def _mismatches(actual: object, expect: dict[str, Any]) -> list[str]:
    if "result_equals" in expect:
        return exact_mismatches(actual=actual, expected=expect["result_equals"])
    return subset_mismatches(actual=actual, expected_subset=expect["result_contains"])


def compiled_conversation_resolution_from_payload(
    payload: dict[str, Any] | None,
) -> CompiledConversationResolution | None:
    if payload is None:
        return None
    question = str(payload["current_question_text"])
    return CompiledConversationResolution(
        current_question_text=question,
        contextualized_question=str(payload.get("contextualized_question") or question),
        clauses=tuple(
            CompiledResolvedClause(
                current_clause_text=str(item["current_clause_text"]),
                resolved_text=str(
                    item.get("resolved_text") or item["current_clause_text"]
                ),
                retained_frame_parts=(),
                values=tuple(
                    CompiledResolvedValue(
                        value_id=str(value["value_id"]),
                        resolved_text=str(value["resolved_text"]),
                        source_kinds=tuple(value.get("source_kinds") or ()),
                        sources=(),
                    )
                    for value in item.get("values") or ()
                ),
            )
            for item in payload.get("clauses") or ({"current_clause_text": question},)
        ),
        inputs=tuple(_compiled_input(item) for item in payload.get("inputs") or ()),
        frame_call=None,
        used_source_card_ids=(),
        used_memory_ids=(),
    )


def _compiled_input(payload: dict[str, Any]):
    if str(payload["kind"]) == "literal_text":
        canonical_identity_payload = payload.get("canonical_identity")
        canonical_identity = (
            ResolvedCanonicalIdentity(
                entity_kind=str(canonical_identity_payload["entity_kind"]),
                key_id=str(canonical_identity_payload["key_id"]),
                key_component_id=str(
                    canonical_identity_payload["key_component_id"]
                ),
                value=str(canonical_identity_payload["value"]),
                authority_refs=tuple(
                    str(item)
                    for item in canonical_identity_payload["authority_refs"]
                ),
                lineage_refs=tuple(
                    str(item)
                    for item in canonical_identity_payload["lineage_refs"]
                ),
            )
            if isinstance(canonical_identity_payload, dict)
            else None
        )
        return ResolvedLiteralQuestionInput(
            input_ref=str(payload["input_ref"]),
            value_source_text=str(payload["value_source_text"]),
            resolved_value_text=str(payload["resolved_value_text"]),
            role=LiteralInputRole(str(payload["role"])),
            occurrence=int(payload.get("occurrence") or 1),
            field_label_text=str(payload.get("field_label_text") or ""),
            value_meaning_hint=str(payload.get("value_meaning_hint") or ""),
            canonical_identity=canonical_identity,
        )
    return ResolvedRowSetQuestionInput(
        input_ref=str(payload["input_ref"]),
        reference_text=str(payload["reference_text"]),
        memory_ids=tuple(str(item) for item in payload["memory_ids"]),
        occurrence=int(payload.get("occurrence") or 1),
    )
