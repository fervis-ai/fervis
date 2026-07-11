"""Strict schema for conversation-resolution model output."""

from __future__ import annotations

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFramePartKind,
)
from fervis.lookup.conversation_resolution import provider_contract as output
from fervis.lookup.conversation_resolution.tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
)


_VALUE_PART_KINDS = frozenset(
    {
        ConversationFramePartKind.ENTITY_IDENTITY,
        ConversationFramePartKind.TIME_SCOPE,
        ConversationFramePartKind.LIMIT,
    }
)
_FIXED_SHAPE_PART_KINDS = frozenset(ConversationFramePartKind) - _VALUE_PART_KINDS


def build_conversation_resolution_tool_schemas(
    *,
    context_sources: tuple[ConversationContextSource, ...] = (),
    context_frames: tuple[ConversationContextFrame, ...] = (),
) -> dict[str, dict[str, object]]:
    return {
        CONVERSATION_RESOLUTION_TOOL_NAME: output.ConversationResolutionOutput.schema(
            {
                "kind": {"type": "string", "enum": ["conversation_resolution"]},
                "current_question_text": {"type": "string", "minLength": 1},
                "outcome": {
                    "type": "object",
                    "oneOf": [
                        _resolved_outcome_schema(
                            context_sources=context_sources,
                            context_frames=context_frames,
                        ),
                        _multiple_meanings_schema(context_sources),
                        _missing_input_schema(context_sources),
                    ],
                },
            }
        )
    }


def _resolved_outcome_schema(
    *,
    context_sources: tuple[ConversationContextSource, ...],
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object]:
    return output.ResolvedOutcomeOutput.schema(
        {
            "kind": {"type": "string", "enum": ["resolved"]},
            "resolution_basis": {"type": "string", "minLength": 1},
            "contextualized_question": {"type": "string", "minLength": 1},
            "clauses": {
                "type": "array",
                "minItems": 1,
                "items": _resolved_clause_schema(
                    context_sources=context_sources,
                    context_frames=context_frames,
                ),
            },
            "frame_call": _frame_call_schema(context_frames),
        }
    )


def _resolved_clause_schema(
    *,
    context_sources: tuple[ConversationContextSource, ...],
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object]:
    return output.ResolvedClauseOutput.schema(
        {
            "current_clause_text": {"type": "string", "minLength": 1},
            "occurrence": {"type": "integer", "minimum": 1},
            "resolved_text": {"type": "string", "minLength": 1},
            "retained_frame_parts": _frame_part_references_schema(
                context_frames,
                allowed_kinds=_FIXED_SHAPE_PART_KINDS,
            ),
            "values": {
                "type": "array",
                "items": output.ResolvedValueOutput.schema(
                    {
                        "value_id": {"type": "string", "minLength": 1},
                        "resolved_text": {"type": "string", "minLength": 1},
                        "sources": {
                            "type": "array",
                            "minItems": 1,
                            "items": _resolution_source_schema(
                                context_sources=context_sources,
                                context_frames=context_frames,
                            ),
                        },
                    }
                ),
            },
        }
    )


def _resolution_source_schema(
    *,
    context_sources: tuple[ConversationContextSource, ...],
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object]:
    branches = [
        _current_span_source_schema(),
        *(
            _context_anchor_source_schema(source, anchor_index=index)
            for source in context_sources
            for index, _anchor in enumerate(source.meaning_anchors)
        ),
        *(
            _frame_part_source_schema(frame, allowed_kinds=_VALUE_PART_KINDS)
            for frame in context_frames
            if any(part.kind in _VALUE_PART_KINDS for part in frame.parts)
        ),
    ]
    return {"type": "object", "oneOf": branches}


def _current_span_source_schema() -> dict[str, object]:
    return output.CurrentSpanSourceOutput.schema(
        {
            "kind": {"type": "string", "enum": ["current_span"]},
            "text": {"type": "string", "minLength": 1},
            "occurrence": {"type": "integer", "minimum": 1},
        }
    )


def _context_anchor_source_schema(
    source: ConversationContextSource,
    *,
    anchor_index: int,
) -> dict[str, object]:
    anchor = source.meaning_anchors[anchor_index]
    return output.ContextAnchorSourceOutput.schema(
        {
            "kind": {"type": "string", "enum": ["context_anchor"]},
            "source_id": {"type": "string", "enum": [source.source_id]},
            "memory_id": {"type": "string", "enum": [anchor.memory_id]},
            "source_text": {"type": "string", "enum": [anchor.text]},
        }
    )


def _frame_part_source_schema(
    frame: ConversationContextFrame,
    *,
    allowed_kinds: frozenset[ConversationFramePartKind],
) -> dict[str, object]:
    return output.FramePartSourceOutput.schema(
        {
            "kind": {"type": "string", "enum": ["frame_part"]},
            "frame_id": {"type": "string", "enum": [frame.frame_id]},
            "part_id": {
                "type": "string",
                "enum": [
                    part.part_id for part in frame.parts if part.kind in allowed_kinds
                ],
            },
        }
    )


def _frame_part_references_schema(
    frames: tuple[ConversationContextFrame, ...],
    *,
    allowed_kinds: frozenset[ConversationFramePartKind],
) -> dict[str, object]:
    branches = [
        _frame_part_source_schema(frame, allowed_kinds=allowed_kinds)
        for frame in frames
        if any(part.kind in allowed_kinds for part in frame.parts)
    ]
    if not branches:
        return {
            "type": "array",
            "maxItems": 0,
            "items": output.FramePartSourceOutput.schema(
                {
                    "kind": {"type": "string", "enum": ["frame_part"]},
                    "frame_id": {"type": "string", "minLength": 1},
                    "part_id": {"type": "string", "minLength": 1},
                }
            ),
        }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "oneOf": branches,
        },
    }


def _frame_call_schema(
    context_frames: tuple[ConversationContextFrame, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "oneOf": [
            _no_frame_call_schema(),
            *(
                _callable_frame_schema(frame)
                for frame in context_frames
                if frame.callable is not None
            ),
        ],
    }


def _no_frame_call_schema() -> dict[str, object]:
    return output.NoFrameCallOutput.schema(
        {"kind": {"type": "string", "enum": ["none"]}}
    )


def _callable_frame_schema(
    frame: ConversationContextFrame,
) -> dict[str, object]:
    callable_signature = frame.callable
    if callable_signature is None:
        raise ValueError("frame is not callable")
    parameter_ids = [item.parameter_id for item in callable_signature.parameters]
    return output.FrameCallOutput.schema(
        {
            "kind": {"type": "string", "enum": ["call"]},
            "frame_id": {"type": "string", "enum": [frame.frame_id]},
            "arguments": {
                "type": "array",
                "minItems": len(parameter_ids),
                "maxItems": len(parameter_ids),
                "items": {
                    "type": "object",
                    "oneOf": [
                        output.CarriedFrameArgumentOutput.schema(
                            {
                                "kind": {"type": "string", "enum": ["carry"]},
                                "parameter_id": {
                                    "type": "string",
                                    "enum": parameter_ids,
                                },
                            }
                        ),
                        output.ResolvedValueFrameArgumentOutput.schema(
                            {
                                "kind": {
                                    "type": "string",
                                    "enum": ["resolved_value"],
                                },
                                "parameter_id": {
                                    "type": "string",
                                    "enum": parameter_ids,
                                },
                                "value_id": {"type": "string", "minLength": 1},
                            }
                        ),
                    ],
                },
            },
        }
    )


def _multiple_meanings_schema(
    context_sources: tuple[ConversationContextSource, ...],
) -> dict[str, object]:
    return _unresolved_outcome_schema(
        kind="multiple_meanings",
        context_sources=context_sources,
        minimum_candidates=2,
    )


def _missing_input_schema(
    context_sources: tuple[ConversationContextSource, ...],
) -> dict[str, object]:
    return _unresolved_outcome_schema(
        kind="missing_input",
        context_sources=context_sources,
        minimum_candidates=0,
    )


def _unresolved_outcome_schema(
    *,
    kind: str,
    context_sources: tuple[ConversationContextSource, ...],
    minimum_candidates: int,
) -> dict[str, object]:
    source_id_schema = {
        "type": "string",
        "enum": [item.source_id for item in context_sources],
    }
    return output.UnresolvedOutcomeOutput.schema(
        {
            "kind": {"type": "string", "enum": [kind]},
            "why_unresolved": {"type": "string", "minLength": 1},
            "candidate_interpretations": {
                "type": "array",
                "minItems": minimum_candidates,
                "items": output.CandidateInterpretationOutput.schema(
                    {
                        "contextualized_question": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "context_evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": output.SourceEvidenceOutput.schema(
                                {
                                    "source_id": source_id_schema,
                                    "exact_source_texts": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                }
                            ),
                        },
                    }
                ),
            },
        }
    )
