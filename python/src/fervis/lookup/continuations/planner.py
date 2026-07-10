"""Derive prior-question continuation plans from CR and memory."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.lookup.conversation_resolution.model import (
    ConversationResolution,
    PriorQuestionContinuation,
)
from fervis.lookup.continuations.model import (
    ContinuationCarriedInput,
    ContinuationPlan,
    ContinuationPlanKind,
    ContinuationReplacement,
)
from fervis.memory.prior_requests import PriorRequestMemory
from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationMemoryCardProjection,
)

_ANSWER_SUBJECT_PART_ID = "answer_subject"


@dataclass(frozen=True)
class _ContinuationContext:
    current_question: str
    resolved_request_text: str
    continuation: PriorQuestionContinuation
    frame: ConversationContextFrame
    prior_request: PriorRequestMemory | None


def derive_continuation_plan(
    *,
    resolution: ConversationResolution | None,
    memory_projection: ConversationMemoryCardProjection | None,
) -> ContinuationPlan:
    context = _continuation_context(
        resolution=resolution,
        memory_projection=memory_projection,
    )
    if context is None:
        return ContinuationPlan.none()

    replacements = _replacements(context)
    carried_inputs = _carried_inputs(context)
    kind = _plan_kind(replacements)
    return ContinuationPlan(
        kind=kind,
        current_question=context.current_question,
        resolved_request_text=context.resolved_request_text,
        frame_id=context.frame.frame_id,
        prior_answer_fact=context.frame.prior_answer_fact,
        replacements=replacements,
        carried_inputs=carried_inputs,
    )


def _continuation_context(
    *,
    resolution: ConversationResolution | None,
    memory_projection: ConversationMemoryCardProjection | None,
) -> _ContinuationContext | None:
    if resolution is None or memory_projection is None:
        return None
    clauses = tuple(resolution.clause_resolutions)
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    continuation = clause.continuation
    if continuation is None:
        return None
    frame = _frames_by_id(memory_projection).get(continuation.frame_id)
    if frame is None:
        return None
    return _ContinuationContext(
        current_question=resolution.current_question_text,
        resolved_request_text=clause.resolved_clause_text,
        continuation=continuation,
        frame=frame,
        prior_request=_prior_request_memory(
            frame=frame,
            memory_projection=memory_projection,
        ),
    )


def _frames_by_id(
    memory_projection: ConversationMemoryCardProjection,
) -> dict[str, ConversationContextFrame]:
    frames_by_id: dict[str, ConversationContextFrame] = {}
    for frame in memory_projection.context_frames:
        frames_by_id[frame.frame_id] = frame
    return frames_by_id


def _prior_request_memory(
    *,
    frame: ConversationContextFrame,
    memory_projection: ConversationMemoryCardProjection,
) -> PriorRequestMemory | None:
    card_ids = _frame_source_card_ids(frame, memory_projection=memory_projection)
    for card in memory_projection.cards:
        if card.card_id not in card_ids:
            continue
        if card.kind == "prior_answer_request":
            return memory_projection.prior_request(card.memory_id)
    return None


def _frame_source_card_ids(
    frame: ConversationContextFrame,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> frozenset[str]:
    frame_source_ids = frozenset(frame.source_ids)
    card_ids: set[str] = set()
    for source in memory_projection.context_sources:
        source_belongs_to_frame = source.source_id in frame_source_ids
        if source_belongs_to_frame:
            card_ids.update(source.source_card_ids)
    return frozenset(card_ids)


def _replacements(
    context: _ContinuationContext,
) -> tuple[ContinuationReplacement, ...]:
    parts_by_id = _parts_by_id(context.frame)
    replacements: list[ContinuationReplacement] = []
    for replacement in context.continuation.replacements:
        part = parts_by_id.get(replacement.part_id)
        if part is None:
            continue
        replacements.append(
            ContinuationReplacement(
                part=part,
                current_text=replacement.current_text,
            )
        )
    return tuple(replacements)


def _parts_by_id(frame: ConversationContextFrame) -> dict[str, object]:
    parts_by_id: dict[str, object] = {}
    for part in frame.replaceable_parts:
        parts_by_id[part.part_id] = part
    return parts_by_id


def _carried_inputs(
    context: _ContinuationContext,
) -> tuple[ContinuationCarriedInput, ...]:
    if context.prior_request is None:
        return ()
    replaced = frozenset(
        replacement.part_id for replacement in context.continuation.replacements
    )
    output: list[ContinuationCarriedInput] = []
    for part in context.frame.replaceable_parts:
        if _skip_carried_input_part(part.part_id, replaced_part_ids=replaced):
            continue
        slot = context.prior_request.slot(part.part_id)
        if slot is None:
            continue
        output.append(
            ContinuationCarriedInput(
                part=part,
                resolved_value_text=slot.resolved_value_text,
                field_label_text=slot.field_label_text,
                value_meaning_hint=slot.value_meaning_hint,
                binding=context.prior_request.binding(part.part_id),
            )
        )
    return tuple(output)


def _skip_carried_input_part(
    part_id: str,
    *,
    replaced_part_ids: frozenset[str],
) -> bool:
    part_was_replaced = part_id in replaced_part_ids
    part_is_answer_subject = part_id == _ANSWER_SUBJECT_PART_ID
    return part_was_replaced or part_is_answer_subject


def _plan_kind(
    replacements: tuple[ContinuationReplacement, ...],
) -> ContinuationPlanKind:
    replaces_answer_subject = any(
        item.part_id == _ANSWER_SUBJECT_PART_ID for item in replacements
    )
    if replaces_answer_subject:
        return ContinuationPlanKind.SHAPE_CHANGING
    return ContinuationPlanKind.SAME_FACT_INPUT_REPLACEMENT
