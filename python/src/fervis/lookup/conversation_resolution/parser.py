"""Parse provider-authored conversation resolutions into closed values."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Callable, TypeVar
from typing_extensions import assert_never

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
    ConversationFramePartKind,
)
from fervis.lookup.conversation_resolution import provider_contract as output
from fervis.lookup.conversation_resolution.model import (
    CandidateInterpretation,
    ContextAnchorSource,
    ConversationFrameCall,
    ConversationResolution,
    ConversationResolutionResult,
    CurrentSpanSource,
    FrameParameterRef,
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
from fervis.lookup.provider_contract import ProviderObject


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
    payload: dict[str, object],
    current_question: str,
    context_sources: tuple[ConversationContextSource, ...] = (),
    context_frames: tuple[ConversationContextFrame, ...] = (),
) -> ConversationResolutionResult:
    if tool_name != CONVERSATION_RESOLUTION_TOOL_NAME:
        raise ValueError("unknown conversation resolution tool")
    item = output.ConversationResolutionOutput.parse(payload)
    if _required_text(item.kind) != "conversation_resolution":
        raise ValueError("invalid conversation resolution kind")
    current_question_text = _required_text(item.current_question_text)
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


def _parse_outcome(value: ProviderObject, *, context: _ParseContext) -> _ParsedOutcome:
    kind = _OutcomeKind(value.discriminator("kind"))
    if kind is _OutcomeKind.RESOLVED:
        item = value.parse_as(output.ResolvedOutcomeOutput)
        resolution_basis = _required_text(item.resolution_basis)
        contextualized_question = _required_text(item.contextualized_question)
        clauses = _resolved_clauses(
            item.clauses,
            contextualized_question=contextualized_question,
            context=context,
        )
        frame_call = _derived_frame_call(clauses, context=context)
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
    if kind is _OutcomeKind.MULTIPLE_MEANINGS:
        return _ParsedOutcome(
            resolution_basis="",
            contextualized_question="",
            clauses=(),
            frame_call=None,
            unresolved=_unresolved_outcome(value, kind=kind, context=context),
        )
    if kind is _OutcomeKind.MISSING_INPUT:
        return _ParsedOutcome(
            resolution_basis="",
            contextualized_question="",
            clauses=(),
            frame_call=None,
            unresolved=_unresolved_outcome(value, kind=kind, context=context),
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


_ItemT = TypeVar("_ItemT")


def _unique_by_id(
    items: tuple[_ItemT, ...],
    *,
    identity: Callable[[_ItemT], str],
    label: str,
) -> dict[str, _ItemT]:
    output: dict[str, _ItemT] = {}
    for item in items:
        item_id = identity(item)
        if item_id in output:
            raise ValueError(f"duplicate {label} id")
        output[item_id] = item
    return output


def _resolved_clauses(
    items: tuple[output.ResolvedClauseOutput, ...],
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
        for index, item in enumerate(items)
    )
    return clauses


def _resolved_clause(
    item: output.ResolvedClauseOutput,
    *,
    path: str,
    contextualized_question: str,
    context: _ParseContext,
) -> ResolvedConversationClause:
    current_clause_text = _required_text(item.current_clause_text)
    occurrence = item.occurrence
    if occurrence < 1:
        raise ValueError(f"{path}.occurrence must be a positive integer")
    _require_occurrence(
        text=current_clause_text,
        occurrence=occurrence,
        source=context.current_question,
        path=f"{path}.current_clause_text",
    )
    resolved_text = _required_text(item.resolved_text)
    if resolved_text not in contextualized_question:
        raise ValueError(f"{path}.resolved_text must occur in contextualized_question")
    values = tuple(
        _resolved_value(
            value,
            path=f"{path}.values[{index}]",
            current_clause_text=current_clause_text,
            context=context,
        )
        for index, value in enumerate(item.values)
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
            for index, retained in enumerate(item.retained_frame_parts)
        ),
        values=values,
    )


def _resolved_value(
    item: output.ResolvedValueOutput,
    *,
    path: str,
    current_clause_text: str,
    context: _ParseContext,
) -> ResolvedConversationValue:
    return ResolvedConversationValue(
        value_id=_required_text(item.value_id),
        resolved_text=_required_text(item.resolved_text),
        frame_parameter=_frame_parameter_ref(item.frame_parameter, context=context),
        sources=tuple(
            _resolution_source(
                source,
                path=f"{path}.sources[{index}]",
                current_clause_text=current_clause_text,
                context=context,
            )
            for index, source in enumerate(item.sources)
        ),
    )


def _resolution_source(
    value: ProviderObject,
    *,
    path: str,
    current_clause_text: str,
    context: _ParseContext,
) -> ResolutionSource:
    kind = ResolutionSourceKind(value.discriminator("kind"))
    if kind is ResolutionSourceKind.CURRENT_SPAN:
        current_span = value.parse_as(output.CurrentSpanSourceOutput)
        text = _required_text(current_span.text)
        occurrence = current_span.occurrence
        if occurrence < 1:
            raise ValueError(f"{path}.occurrence must be a positive integer")
        _require_occurrence(
            text=text,
            occurrence=occurrence,
            source=current_clause_text,
            path=f"{path}.text",
        )
        return CurrentSpanSource(text=text, occurrence=occurrence)
    if kind is ResolutionSourceKind.CONTEXT_ANCHOR:
        context_anchor = value.parse_as(output.ContextAnchorSourceOutput)
        source_id = _required_text(context_anchor.source_id)
        source_contract = context.source_contracts.get(source_id)
        if source_contract is None:
            raise ValueError(f"{path}.source_id is not available")
        anchor_id = _required_text(context_anchor.anchor_id)
        matching_anchors = tuple(
            anchor
            for anchor in source_contract.meaning_anchors
            if anchor.anchor_id == anchor_id
        )
        if len(matching_anchors) != 1:
            raise ValueError(f"{path} does not reference a visible context anchor")
        source_text = matching_anchors[0].text
        return ContextAnchorSource(
            source_id=source_id,
            anchor_id=anchor_id,
            source_text=source_text,
            memory_ids=(
                (anchor_id,)
                if anchor_id in source_contract.source_memory_ids
                else ()
            ),
        )
    if kind is ResolutionSourceKind.FRAME_PART:
        return _frame_part_source(
            value.parse_as(output.FramePartSourceOutput),
            path=path,
            context=context,
            allowed_kinds=_VALUE_PART_KINDS,
        )
    assert_never(kind)


def _frame_part_source(
    item: output.FramePartSourceOutput,
    *,
    path: str,
    context: _ParseContext,
    allowed_kinds: frozenset[ConversationFramePartKind],
) -> FramePartSource:
    frame_id = _required_text(item.frame_id)
    frame = context.frames.get(frame_id)
    if frame is None:
        raise ValueError(f"{path}.frame_id is not available")
    part_id = _required_text(item.part_id)
    part = next((part for part in frame.parts if part.part_id == part_id), None)
    if part is None or part.kind not in allowed_kinds:
        raise ValueError(f"{path}.part_id is not available for this role")
    return FramePartSource(frame_id=frame_id, part_id=part_id)


def _frame_parameter_ref(
    value: ProviderObject,
    *,
    context: _ParseContext,
) -> FrameParameterRef | None:
    kind = value.discriminator("kind")
    if kind == "none":
        value.parse_as(output.NoFrameParameterOutput)
        return None
    if kind != "parameter":
        raise ValueError("unsupported frame_parameter.kind")
    parameter = value.parse_as(output.FrameParameterOutput)
    frame_id = _required_text(parameter.frame_id)
    frame = context.frames.get(frame_id)
    if frame is None or frame.callable is None:
        raise ValueError("frame parameter does not reference a callable frame")
    parameter_id = _required_text(parameter.parameter_id)
    available_ids = {
        item.parameter_id for item in frame.callable.parameters
    }
    if parameter_id not in available_ids:
        raise ValueError("frame parameter is not available")
    return FrameParameterRef(frame_id=frame_id, parameter_id=parameter_id)


def _derived_frame_call(
    clauses: tuple[ResolvedConversationClause, ...],
    *,
    context: _ParseContext,
) -> ConversationFrameCall | None:
    values = tuple(value for clause in clauses for value in clause.values)
    retained_parts = {
        (part.frame_id, part.part_id)
        for clause in clauses
        for part in clause.retained_frame_parts
    }
    calls = tuple(
        call
        for frame in context.frames.values()
        if (
            call := _complete_frame_call(
                frame,
                values=values,
                retained_parts=retained_parts,
            )
        )
        is not None
    )
    if len(calls) > 1:
        raise ValueError("resolved values match more than one callable frame")
    return calls[0] if calls else None


def _complete_frame_call(
    frame: ConversationContextFrame,
    *,
    values: tuple[ResolvedConversationValue, ...],
    retained_parts: set[tuple[str, str]],
) -> ConversationFrameCall | None:
    signature = frame.callable
    if signature is None:
        return None
    fixed_part_ids = {
        part.part_id for part in frame.parts if part.kind in _FIXED_SHAPE_PART_KINDS
    }
    retained_part_ids = {
        part_id for frame_id, part_id in retained_parts if frame_id == frame.frame_id
    }
    if not fixed_part_ids.issubset(retained_part_ids):
        return None
    bindings = tuple(
        (value, value.frame_parameter)
        for value in values
        if value.frame_parameter is not None
        and value.frame_parameter.frame_id == frame.frame_id
    )
    if len(bindings) != len(values):
        return None
    expected_ids = {parameter.parameter_id for parameter in signature.parameters}
    actual_ids = {parameter.parameter_id for _, parameter in bindings}
    if len(actual_ids) != len(bindings) or actual_ids != expected_ids:
        return None
    values_by_parameter_id = {
        parameter.parameter_id: value for value, parameter in bindings
    }
    arguments = tuple(
        ResolvedValueFrameArgument(
            parameter_id=parameter.parameter_id,
            value_id=values_by_parameter_id[parameter.parameter_id].value_id,
        )
        for parameter in signature.parameters
    )
    return ConversationFrameCall(frame_id=frame.frame_id, arguments=arguments)


def _unresolved_outcome(
    value: ProviderObject,
    *,
    kind: _OutcomeKind,
    context: _ParseContext,
) -> UnresolvedResolution:
    item = value.parse_as(output.UnresolvedOutcomeOutput)
    resolution = UnresolvedResolution(
        unresolved_kind=kind.value,
        why_unresolved=_required_text(item.why_unresolved),
        candidate_interpretations=tuple(
            _candidate_interpretation(
                candidate,
                path=f"outcome.candidate_interpretations[{index}]",
                context=context,
            )
            for index, candidate in enumerate(item.candidate_interpretations)
        ),
    )
    if kind is _OutcomeKind.MULTIPLE_MEANINGS:
        _require_distinct_context_evidence(resolution.candidate_interpretations)
    return resolution


def _candidate_interpretation(
    item: output.CandidateInterpretationOutput,
    *,
    path: str,
    context: _ParseContext,
) -> CandidateInterpretation:
    context_evidence = tuple(
        _source_evidence(
            evidence,
            path=f"{path}.context_evidence[{index}]",
            context=context,
        )
        for index, evidence in enumerate(item.context_evidence)
    )
    if any(evidence.source_id == "current_question" for evidence in context_evidence):
        raise ValueError(f"{path}.context_evidence must cite prior context")
    return CandidateInterpretation(
        contextualized_question=_required_text(item.contextualized_question),
        context_evidence=context_evidence,
    )


def _require_distinct_context_evidence(
    candidates: tuple[CandidateInterpretation, ...],
) -> None:
    signatures = tuple(
        frozenset(
            (evidence.source_id, evidence.exact_source_texts)
            for evidence in candidate.context_evidence
        )
        for candidate in candidates
    )
    if len(signatures) != len(set(signatures)):
        raise ValueError("multiple meanings require distinct context evidence")


def _source_evidence(
    item: output.SourceEvidenceOutput,
    *,
    path: str,
    context: _ParseContext,
) -> SourceEvidence:
    source_id = _required_text(item.source_id)
    source = context.sources.get(source_id)
    if source is None:
        raise ValueError(f"{path}.source_id is not available")
    exact_texts = tuple(_required_text(text) for text in item.exact_source_texts)
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
                for source_id in source.context_source_references():
                    _append_unique(source_ids, source_id)
                for frame_id, _ in source.frame_part_references():
                    for frame_source_id in context.frames[frame_id].source_ids:
                        _append_unique(source_ids, frame_source_id)
    if frame_call is not None:
        for source_id in context.frames[frame_call.frame_id].source_ids:
            _append_unique(source_ids, source_id)
    for interpretation in unresolved.candidate_interpretations:
        for evidence in interpretation.context_evidence:
            _append_unique(source_ids, evidence.source_id)
    return tuple(
        context.sources[source_id]
        for source_id in source_ids
        if source_id != "current_question"
    )


def _source_card_ids(sources: tuple[_SourceText, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            card_id for source in sources for card_id in source.source_card_ids
        )
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
        memory_id
        for clause in clauses
        for value in clause.values
        for source in value.sources
        for memory_id in source.memory_references()
    ]
    frame_ids = {
        frame_id
        for clause in clauses
        for value in clause.values
        for source in value.sources
        for frame_id, _ in source.frame_part_references()
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


def _required_text(value: str) -> str:
    if not value.strip():
        raise ValueError("conversation resolution requires non-empty text")
    return value
