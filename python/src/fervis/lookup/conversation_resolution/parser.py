"""Parse provider-authored conversation resolutions into closed values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, assert_never

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFramePartKind,
)
from fervis.lookup.conversation_resolution import provider_contract as output
from fervis.lookup.conversation_resolution.model import (
    CandidateInterpretation,
    CarriedFrameArgument,
    ContextAnchorSource,
    ConversationFrameCall,
    ConversationResolution,
    ConversationResolutionResult,
    CurrentSpanSource,
    FrameArgument,
    FrameArgumentKind,
    FramePartSource,
    ResolutionSource,
    ResolutionSourceKind,
    ResolvedConversationClause,
    ResolvedConversationValue,
    ResolvedValueFrameArgument,
    SourceEvidence,
    UnresolvedResolution,
)
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


def parse_conversation_resolution(
    *,
    tool_name: str,
    payload: dict[str, Any],
    current_question: str,
    context_sources: tuple[ConversationContextSource, ...] = (),
    context_frames: tuple[ConversationContextFrame, ...] = (),
) -> ConversationResolutionResult:
    if tool_name != CONVERSATION_RESOLUTION_TOOL_NAME:
        raise ValueError("unknown conversation resolution tool")
    item = output.ConversationResolutionOutput.parse(payload)
    if item.kind != "conversation_resolution":
        raise ValueError("invalid conversation resolution kind")
    current_question_text = _required_string(
        item.current_question_text,
        path="current_question_text",
    )
    if current_question_text != current_question:
        raise ValueError("current_question_text must exactly match current question")
    context = _ParseContext.from_inputs(
        current_question=current_question_text,
        context_sources=context_sources,
        context_frames=context_frames,
    )
    outcome = _parse_outcome(item.outcome, context=context)
    used_sources = _used_sources(
        clauses=outcome.clauses,
        frame_call=outcome.frame_call,
        unresolved=outcome.unresolved,
        context=context,
    )
    return ConversationResolutionResult(
        outcome=ConversationResolution(
            current_question_text=current_question_text,
            resolution_basis=outcome.resolution_basis,
            contextualized_question=outcome.contextualized_question,
            clauses=outcome.clauses,
            frame_call=outcome.frame_call,
            unresolved=outcome.unresolved,
            used_source_card_ids=_source_card_ids(used_sources),
            used_memory_ids=_used_memory_ids(
                clauses=outcome.clauses,
                frame_call=outcome.frame_call,
                unresolved=outcome.unresolved,
                used_sources=used_sources,
                context=context,
            ),
        )
    )


class _OutcomeKind(StrEnum):
    RESOLVED = "resolved"
    MULTIPLE_MEANINGS = "multiple_meanings"
    MISSING_INPUT = "missing_input"


@dataclass(frozen=True)
class _ParsedOutcome:
    resolution_basis: str
    contextualized_question: str
    clauses: tuple[ResolvedConversationClause, ...]
    frame_call: ConversationFrameCall | None
    unresolved: UnresolvedResolution


def _parse_outcome(raw: object, *, context: _ParseContext) -> _ParsedOutcome:
    payload = _required_dict(raw, path="outcome")
    kind = _OutcomeKind(_required_string(payload.get("kind"), path="outcome.kind"))
    if kind is _OutcomeKind.RESOLVED:
        item = output.ResolvedOutcomeOutput.parse(payload)
        resolution_basis = _required_string(
            item.resolution_basis,
            path="outcome.resolution_basis",
        )
        contextualized_question = _required_string(
            item.contextualized_question,
            path="outcome.contextualized_question",
        )
        clauses = _resolved_clauses(
            item.clauses,
            contextualized_question=contextualized_question,
            context=context,
        )
        values_by_id = {
            value.value_id: value for clause in clauses for value in clause.values
        }
        frame_call = _frame_call(
            item.frame_call,
            values_by_id=values_by_id,
            context=context,
        )
        return _ParsedOutcome(
            resolution_basis=resolution_basis,
            contextualized_question=contextualized_question,
            clauses=clauses,
            frame_call=frame_call,
            unresolved=UnresolvedResolution(
                unresolved_kind="none",
                why_unresolved="",
                candidate_interpretations=(),
            ),
        )
    if kind in {_OutcomeKind.MULTIPLE_MEANINGS, _OutcomeKind.MISSING_INPUT}:
        return _ParsedOutcome(
            resolution_basis="",
            contextualized_question="",
            clauses=(),
            frame_call=None,
            unresolved=_unresolved_outcome(payload, kind=kind, context=context),
        )
    assert_never(kind)


@dataclass(frozen=True)
class _SourceText:
    source_id: str
    text: str
    source_card_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ParseContext:
    current_question: str
    sources: dict[str, _SourceText]
    source_contracts: dict[str, ConversationContextSource]
    frames: dict[str, ConversationContextFrame]

    @classmethod
    def from_inputs(
        cls,
        *,
        current_question: str,
        context_sources: tuple[ConversationContextSource, ...],
        context_frames: tuple[ConversationContextFrame, ...],
    ) -> "_ParseContext":
        source_contracts = _unique_by_id(
            context_sources,
            identity=lambda item: item.source_id,
            label="context source",
        )
        sources = {
            "current_question": _SourceText(
                source_id="current_question",
                text=current_question,
            ),
            **{
                source.source_id: _SourceText(
                    source_id=source.source_id,
                    text=source.text,
                    source_card_ids=source.source_card_ids,
                    source_memory_ids=source.source_memory_ids,
                )
                for source in context_sources
            },
        }
        frames = _unique_by_id(
            context_frames,
            identity=lambda item: item.frame_id,
            label="context frame",
        )
        for frame in context_frames:
            if any(source_id not in sources for source_id in frame.source_ids):
                raise ValueError("context frame references unavailable source")
        return cls(
            current_question=current_question,
            sources=sources,
            source_contracts=source_contracts,
            frames=frames,
        )


def _unique_by_id(items, *, identity, label: str):
    output = {}
    for item in items:
        item_id = identity(item)
        if item_id in output:
            raise ValueError(f"duplicate {label} id")
        output[item_id] = item
    return output


def _resolved_clauses(
    raw: object,
    *,
    contextualized_question: str,
    context: _ParseContext,
) -> tuple[ResolvedConversationClause, ...]:
    clauses = tuple(
        _resolved_clause(
            item,
            path=f"clauses[{index}]",
            contextualized_question=contextualized_question,
            context=context,
        )
        for index, item in enumerate(_required_dicts(raw, path="clauses"))
    )
    return clauses


def _resolved_clause(
    raw: object,
    *,
    path: str,
    contextualized_question: str,
    context: _ParseContext,
) -> ResolvedConversationClause:
    item = output.ResolvedClauseOutput.parse(raw)
    current_clause_text = _required_string(
        item.current_clause_text,
        path=f"{path}.current_clause_text",
    )
    occurrence = _positive_int(item.occurrence, path=f"{path}.occurrence")
    _require_occurrence(
        text=current_clause_text,
        occurrence=occurrence,
        source=context.current_question,
        path=f"{path}.current_clause_text",
    )
    resolved_text = _required_string(item.resolved_text, path=f"{path}.resolved_text")
    if resolved_text not in contextualized_question:
        raise ValueError(f"{path}.resolved_text must occur in contextualized_question")
    values = tuple(
        _resolved_value(
            value,
            path=f"{path}.values[{index}]",
            current_clause_text=current_clause_text,
            context=context,
        )
        for index, value in enumerate(_required_dicts(item.values, path=f"{path}.values"))
    )
    return ResolvedConversationClause(
        current_clause_text=current_clause_text,
        occurrence=occurrence,
        resolved_text=resolved_text,
        retained_frame_parts=tuple(
            _frame_part_source(
                retained,
                path=f"{path}.retained_frame_parts[{index}]",
                context=context,
                allowed_kinds=_FIXED_SHAPE_PART_KINDS,
            )
            for index, retained in enumerate(
                _required_dicts(
                    item.retained_frame_parts,
                    path=f"{path}.retained_frame_parts",
                )
            )
        ),
        values=values,
    )


def _resolved_value(
    raw: object,
    *,
    path: str,
    current_clause_text: str,
    context: _ParseContext,
) -> ResolvedConversationValue:
    item = output.ResolvedValueOutput.parse(raw)
    return ResolvedConversationValue(
        value_id=_required_string(item.value_id, path=f"{path}.value_id"),
        resolved_text=_required_string(
            item.resolved_text,
            path=f"{path}.resolved_text",
        ),
        sources=tuple(
            _resolution_source(
                source,
                path=f"{path}.sources[{index}]",
                current_clause_text=current_clause_text,
                context=context,
            )
            for index, source in enumerate(
                _required_dicts(item.sources, path=f"{path}.sources")
            )
        ),
    )


def _resolution_source(
    raw: object,
    *,
    path: str,
    current_clause_text: str,
    context: _ParseContext,
) -> ResolutionSource:
    source = _required_dict(raw, path=path)
    kind = ResolutionSourceKind(
        _required_string(source.get("kind"), path=f"{path}.kind")
    )
    if kind is ResolutionSourceKind.CURRENT_SPAN:
        item = output.CurrentSpanSourceOutput.parse(source)
        text = _required_string(item.text, path=f"{path}.text")
        occurrence = _positive_int(item.occurrence, path=f"{path}.occurrence")
        _require_occurrence(
            text=text,
            occurrence=occurrence,
            source=current_clause_text,
            path=f"{path}.text",
        )
        return CurrentSpanSource(text=text, occurrence=occurrence)
    if kind is ResolutionSourceKind.CONTEXT_ANCHOR:
        item = output.ContextAnchorSourceOutput.parse(source)
        source_id = _required_string(item.source_id, path=f"{path}.source_id")
        source_contract = context.source_contracts.get(source_id)
        if source_contract is None:
            raise ValueError(f"{path}.source_id is not available")
        memory_id = _required_string(item.memory_id, path=f"{path}.memory_id")
        source_text = _required_string(item.source_text, path=f"{path}.source_text")
        if not any(
            anchor.memory_id == memory_id and anchor.text == source_text
            for anchor in source_contract.meaning_anchors
        ):
            raise ValueError(f"{path} does not reference a visible context anchor")
        return ContextAnchorSource(
            source_id=source_id,
            memory_id=memory_id,
            source_text=source_text,
        )
    if kind is ResolutionSourceKind.FRAME_PART:
        return _frame_part_source(
            source,
            path=path,
            context=context,
            allowed_kinds=_VALUE_PART_KINDS,
        )
    assert_never(kind)


def _frame_part_source(
    raw: object,
    *,
    path: str,
    context: _ParseContext,
    allowed_kinds: frozenset[ConversationFramePartKind],
) -> FramePartSource:
    item = output.FramePartSourceOutput.parse(raw)
    frame_id = _required_string(item.frame_id, path=f"{path}.frame_id")
    frame = context.frames.get(frame_id)
    if frame is None:
        raise ValueError(f"{path}.frame_id is not available")
    part_id = _required_string(item.part_id, path=f"{path}.part_id")
    part = next((part for part in frame.parts if part.part_id == part_id), None)
    if part is None or part.kind not in allowed_kinds:
        raise ValueError(f"{path}.part_id is not available for this role")
    return FramePartSource(frame_id=frame_id, part_id=part_id)


def _frame_call(
    raw: object,
    *,
    values_by_id: dict[str, ResolvedConversationValue],
    context: _ParseContext,
) -> ConversationFrameCall | None:
    item = _required_dict(raw, path="frame_call")
    kind = _required_string(item.get("kind"), path="frame_call.kind")
    if kind == "none":
        output.NoFrameCallOutput.parse(item)
        return None
    if kind != "call":
        raise ValueError("unsupported frame_call.kind")
    call = output.FrameCallOutput.parse(item)
    frame_id = _required_string(call.frame_id, path="frame_call.frame_id")
    frame = context.frames.get(frame_id)
    if frame is None or frame.callable is None:
        raise ValueError("frame_call does not reference a callable frame")
    arguments = tuple(
        _frame_argument(
            argument,
            path=f"frame_call.arguments[{index}]",
            values_by_id=values_by_id,
        )
        for index, argument in enumerate(
            _required_dicts(call.arguments, path="frame_call.arguments")
        )
    )
    expected = {parameter.parameter_id for parameter in frame.callable.parameters}
    actual = {argument.parameter_id for argument in arguments}
    if len(actual) != len(arguments) or actual != expected:
        raise ValueError("frame_call must bind every callable parameter exactly once")
    return ConversationFrameCall(frame_id=frame_id, arguments=arguments)


def _frame_argument(
    raw: object,
    *,
    path: str,
    values_by_id: dict[str, ResolvedConversationValue],
) -> FrameArgument:
    item = _required_dict(raw, path=path)
    kind = FrameArgumentKind(
        _required_string(item.get("kind"), path=f"{path}.kind")
    )
    if kind is FrameArgumentKind.CARRY:
        value = output.CarriedFrameArgumentOutput.parse(item)
        return CarriedFrameArgument(
            parameter_id=_required_string(
                value.parameter_id,
                path=f"{path}.parameter_id",
            )
        )
    if kind is FrameArgumentKind.RESOLVED_VALUE:
        value = output.ResolvedValueFrameArgumentOutput.parse(item)
        value_id = _required_string(value.value_id, path=f"{path}.value_id")
        if value_id not in values_by_id:
            raise ValueError(f"{path}.value_id is not a resolved value")
        return ResolvedValueFrameArgument(
            parameter_id=_required_string(
                value.parameter_id,
                path=f"{path}.parameter_id",
            ),
            value_id=value_id,
        )
    assert_never(kind)


def _unresolved_outcome(
    raw: object,
    *,
    kind: _OutcomeKind,
    context: _ParseContext,
) -> UnresolvedResolution:
    item = output.UnresolvedOutcomeOutput.parse(raw)
    return UnresolvedResolution(
        unresolved_kind=kind.value,
        why_unresolved=_required_string(
            item.why_unresolved,
            path="outcome.why_unresolved",
        ),
        candidate_interpretations=tuple(
            _candidate_interpretation(
                candidate,
                path=f"outcome.candidate_interpretations[{index}]",
                context=context,
            )
            for index, candidate in enumerate(
                _required_dicts(
                    item.candidate_interpretations,
                    path="outcome.candidate_interpretations",
                )
            )
        ),
    )


def _candidate_interpretation(
    raw: object,
    *,
    path: str,
    context: _ParseContext,
) -> CandidateInterpretation:
    item = output.CandidateInterpretationOutput.parse(raw)
    return CandidateInterpretation(
        contextualized_question=_required_string(
            item.contextualized_question,
            path=f"{path}.contextualized_question",
        ),
        supporting_evidence=tuple(
            _source_evidence(
                evidence,
                path=f"{path}.supporting_evidence[{index}]",
                context=context,
            )
            for index, evidence in enumerate(
                _required_dicts(
                    item.supporting_evidence,
                    path=f"{path}.supporting_evidence",
                )
            )
        ),
    )


def _source_evidence(
    raw: object,
    *,
    path: str,
    context: _ParseContext,
) -> SourceEvidence:
    item = output.SourceEvidenceOutput.parse(raw)
    source_id = _required_string(item.source_id, path=f"{path}.source_id")
    source = context.sources.get(source_id)
    if source is None:
        raise ValueError(f"{path}.source_id is not available")
    exact_texts = _string_array(
        item.exact_source_texts,
        path=f"{path}.exact_source_texts",
    )
    for index, text in enumerate(exact_texts):
        _require_occurrence(
            text=text,
            occurrence=1,
            source=source.text,
            path=f"{path}.exact_source_texts[{index}]",
        )
    return SourceEvidence(source_id=source_id, exact_source_texts=exact_texts)


def _used_sources(
    *,
    clauses: tuple[ResolvedConversationClause, ...],
    frame_call: ConversationFrameCall | None,
    unresolved: UnresolvedResolution,
    context: _ParseContext,
) -> tuple[_SourceText, ...]:
    source_ids: list[str] = []
    for clause in clauses:
        for retained in clause.retained_frame_parts:
            for frame_source_id in context.frames[retained.frame_id].source_ids:
                _append_unique(source_ids, frame_source_id)
        for value in clause.values:
            for source in value.sources:
                if source.kind is ResolutionSourceKind.CONTEXT_ANCHOR:
                    _append_unique(source_ids, source.source_id)
                elif source.kind is ResolutionSourceKind.FRAME_PART:
                    for frame_source_id in context.frames[source.frame_id].source_ids:
                        _append_unique(source_ids, frame_source_id)
    if frame_call is not None:
        for source_id in context.frames[frame_call.frame_id].source_ids:
            _append_unique(source_ids, source_id)
    for interpretation in unresolved.candidate_interpretations:
        for evidence in interpretation.supporting_evidence:
            _append_unique(source_ids, evidence.source_id)
    return tuple(
        context.sources[source_id]
        for source_id in source_ids
        if source_id != "current_question"
    )


def _source_card_ids(sources: tuple[_SourceText, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(card_id for source in sources for card_id in source.source_card_ids)
    )


def _used_memory_ids(
    *,
    clauses: tuple[ResolvedConversationClause, ...],
    frame_call: ConversationFrameCall | None,
    unresolved: UnresolvedResolution,
    used_sources: tuple[_SourceText, ...],
    context: _ParseContext,
) -> tuple[str, ...]:
    del unresolved
    memory_ids = [
        source.memory_id
        for clause in clauses
        for value in clause.values
        for source in value.sources
        if source.kind is ResolutionSourceKind.CONTEXT_ANCHOR
    ]
    frame_ids = {
        source.frame_id
        for clause in clauses
        for value in clause.values
        for source in value.sources
        if source.kind is ResolutionSourceKind.FRAME_PART
    }
    frame_ids.update(
        retained.frame_id
        for clause in clauses
        for retained in clause.retained_frame_parts
    )
    if frame_call is not None:
        frame_ids.add(frame_call.frame_id)
    source_ids = {
        source_id
        for frame_id in frame_ids
        for source_id in context.frames[frame_id].source_ids
    }
    for source in used_sources:
        if source.source_id in source_ids:
            memory_ids.extend(source.source_memory_ids)
    return tuple(dict.fromkeys(memory_ids))


def _append_unique(items: list[str], value: str) -> None:
    if value != "current_question" and value not in items:
        items.append(value)


def _require_occurrence(
    *,
    text: str,
    occurrence: int,
    source: str,
    path: str,
) -> None:
    if _occurrence_count(source, text) < occurrence:
        raise ValueError(f"{path} occurrence does not appear in source text")


def _occurrence_count(source: str, text: str) -> int:
    count = 0
    start = 0
    while True:
        index = source.find(text, start)
        if index < 0:
            return count
        count += 1
        start = index + len(text)


def _required_dict(raw: object, *, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object")
    return raw


def _required_dicts(raw: object, *, path: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be an array")
    return tuple(
        _required_dict(item, path=f"{path}[{index}]")
        for index, item in enumerate(raw)
    )


def _required_string(raw: object, *, path: str, allow_empty: bool = False) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{path} must be a string")
    if not allow_empty and not raw.strip():
        raise ValueError(f"{path} must not be empty")
    return raw


def _positive_int(raw: object, *, path: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{path} must be a positive integer")
    return raw


def _string_array(raw: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be an array")
    return tuple(
        _required_string(item, path=f"{path}[{index}]")
        for index, item in enumerate(raw)
    )
