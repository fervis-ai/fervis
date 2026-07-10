"""Strict schema for conversation-resolution model output."""

from __future__ import annotations

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
)
from fervis.lookup.conversation_resolution import provider_contract as provider_output
from fervis.lookup.conversation_resolution.tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
)


def build_conversation_resolution_tool_schemas(
    *,
    context_sources: tuple[ConversationContextSource, ...] = (),
    context_frames: tuple[ConversationContextFrame, ...] = (),
) -> dict[str, dict[str, object]]:
    return {
        CONVERSATION_RESOLUTION_TOOL_NAME: _conversation_resolution_schema(
            context_sources=context_sources,
            context_frames=context_frames,
        )
    }


def _conversation_resolution_schema(
    *,
    context_sources: tuple[ConversationContextSource, ...],
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object]:
    source_id_schema = _source_id_schema(context_sources)
    context_source_id_schema = _context_source_id_schema(context_sources)
    frame_id_schema = _frame_id_schema(context_frames)
    replaceable_part_id_schema = _replaceable_part_id_schema(context_frames)
    return provider_output.ConversationResolutionOutput.schema(
        {
            "kind": {"type": "string", "enum": ["conversation_resolution"]},
            "current_question_text": {"type": "string", "minLength": 1},
            "clause_resolutions": {
                "type": "array",
                "items": _clause_resolution_schema(
                    source_id_schema=context_source_id_schema,
                    frame_id_schema=frame_id_schema,
                    replaceable_part_id_schema=replaceable_part_id_schema,
                    context_source_count=len(context_sources),
                    context_frame_count=len(context_frames),
                ),
            },
            "unresolved": _unresolved_schema(source_id_schema),
        },
    )


def _clause_resolution_schema(
    *,
    source_id_schema: dict[str, object],
    frame_id_schema: dict[str, object],
    replaceable_part_id_schema: dict[str, object] | None,
    context_source_count: int,
    context_frame_count: int,
) -> dict[str, object]:
    properties = _clause_resolution_properties(
        source_id_schema=source_id_schema,
        frame_id_schema=frame_id_schema,
        replaceable_part_id_schema=replaceable_part_id_schema,
        context_source_count=context_source_count,
        context_frame_count=context_frame_count,
    )
    return provider_output.ClauseResolutionOutput.schema(properties)


def _clause_resolution_properties(
    *,
    source_id_schema: dict[str, object],
    frame_id_schema: dict[str, object],
    replaceable_part_id_schema: dict[str, object] | None,
    context_source_count: int,
    context_frame_count: int,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "current_clause_text": {"type": "string", "minLength": 1},
        "occurrence": {"type": "integer", "minimum": 1},
        "requested_value_frame": _requested_value_frame_schema(
            frame_id_schema=frame_id_schema,
            context_frame_count=context_frame_count,
        ),
        "dependencies": {
            "type": "array",
            "items": _dependency_schema(source_id_schema),
            **({"maxItems": 0} if context_source_count == 0 else {}),
        },
        "resolved_clause_text": {"type": "string", "minLength": 1},
    }
    if replaceable_part_id_schema is not None:
        properties["continuation"] = _continuation_schema(
            frame_id_schema=frame_id_schema,
            replaceable_part_id_schema=replaceable_part_id_schema,
        )
    return properties


def _continuation_schema(
    *,
    frame_id_schema: dict[str, object],
    replaceable_part_id_schema: dict[str, object],
) -> dict[str, object]:
    return provider_output.ContinuationOutput.schema(
        {
            "kind": {
                "type": "string",
                "enum": ["continue_prior_question"],
            },
            "frame_id": frame_id_schema,
            "replacements": {
                "type": "array",
                "minItems": 1,
                "items": _continuation_replacement_schema(
                    replaceable_part_id_schema
                ),
            },
        },
    )


def _continuation_replacement_schema(
    replaceable_part_id_schema: dict[str, object],
) -> dict[str, object]:
    return provider_output.ContinuationReplacementOutput.schema(
        {
            "part_id": replaceable_part_id_schema,
            "current_text": {"type": "string", "minLength": 1},
        },
    )


def _requested_value_frame_schema(
    *,
    frame_id_schema: dict[str, object],
    context_frame_count: int,
) -> dict[str, object]:
    return provider_output.RequestedValueFrameOutput.schema(
        {
            "current_value_surface": _current_value_surface_schema(),
            "context_frame_choices": {
                "type": "array",
                "minItems": context_frame_count,
                "maxItems": context_frame_count,
                "items": _context_frame_choice_schema(frame_id_schema),
            },
        },
    )


def _current_value_surface_schema() -> dict[str, object]:
    return provider_output.CurrentValueSurfaceOutput.schema(
        {
            "text": {"type": "string", "minLength": 1},
            "kind": {
                "type": "string",
                "enum": [
                    "self_sufficient_current_value",
                    "broad_current_value",
                    "no_value_request",
                ],
            },
        },
    )


def _context_frame_choice_schema(
    frame_id_schema: dict[str, object],
) -> dict[str, object]:
    return provider_output.ContextFrameChoiceOutput.schema(
        {
            "frame_id": frame_id_schema,
            "current_conflict_quotes": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "choice": {
                "type": "string",
                "enum": [
                    "use_frame",
                    "current_text_names_different_value",
                    "not_for_this_clause",
                    "ambiguous",
                ],
            },
        },
    )


def _dependency_schema(source_id_schema: dict[str, object]) -> dict[str, object]:
    return provider_output.DependencyOutput.schema(
        {
            "anchor_text": {"type": "string", "minLength": 1},
            "occurrence": {"type": "integer", "minimum": 1},
            "meaning_components": {
                "type": "array",
                "minItems": 1,
                "items": _meaning_component_schema(source_id_schema),
            },
            "resolved_text": {"type": "string", "minLength": 1},
            "must_preserve_terms": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "kind": {"type": "string", "enum": ["reference", "scope"]},
        },
    )


def _meaning_component_schema(source_id_schema: dict[str, object]) -> dict[str, object]:
    return provider_output.MeaningComponentOutput.schema(
        {
            "source_id": source_id_schema,
            "source_text": {"type": "string", "minLength": 1},
            "memory_id": {"type": "string", "minLength": 1},
            "resolved_text": {"type": "string", "minLength": 1},
            "kind": {
                "type": "string",
                "enum": ["entity", "scope", "row_set", "value", "other"],
            },
        },
    )


def _unresolved_schema(source_id_schema: dict[str, object]) -> dict[str, object]:
    return provider_output.UnresolvedOutput.schema(
        {
            "why_unresolved": {"type": "string"},
            "candidate_interpretations": {
                "type": "array",
                "items": _candidate_interpretation_schema(source_id_schema),
            },
            "unresolved_kind": {
                "type": "string",
                "enum": ["none", "multiple_meanings", "missing_input"],
            },
        },
    )


def _candidate_interpretation_schema(
    source_id_schema: dict[str, object],
) -> dict[str, object]:
    return provider_output.CandidateInterpretationOutput.schema(
        {
            "integrated_question": {"type": "string", "minLength": 1},
            "supporting_evidence": {
                "type": "array",
                "minItems": 1,
                "items": _source_evidence_schema(source_id_schema),
            },
        },
    )


def _source_evidence_schema(source_id_schema: dict[str, object]) -> dict[str, object]:
    return provider_output.SourceEvidenceOutput.schema(
        {
            "source_id": source_id_schema,
            "exact_source_texts": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
    )


def _source_id_schema(
    context_sources: tuple[ConversationContextSource, ...],
) -> dict[str, object]:
    source_ids = ["current_question", *(item.source_id for item in context_sources)]
    return {"type": "string", "enum": source_ids}


def _context_source_id_schema(
    context_sources: tuple[ConversationContextSource, ...],
) -> dict[str, object]:
    source_ids = [item.source_id for item in context_sources]
    if not source_ids:
        return {"type": "string"}
    return {"type": "string", "enum": source_ids}


def _frame_id_schema(
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object]:
    frame_ids = [item.frame_id for item in context_frames]
    if not frame_ids:
        return {"type": "string"}
    return {"type": "string", "enum": frame_ids}


def _replaceable_part_id_schema(
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object] | None:
    part_ids = sorted(
        {
            part.part_id
            for frame in context_frames
            for part in frame.replaceable_parts
        }
    )
    if not part_ids:
        return None
    return {"type": "string", "enum": part_ids}
