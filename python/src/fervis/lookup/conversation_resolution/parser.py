"""Parse provider-authored conversation-resolution decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
    ConversationMeaningAnchor,
)
from fervis.lookup.conversation_resolution.model import (
    CandidateInterpretation,
    ClauseDependency,
    ClauseResolution,
    ContextFrameChoice,
    ContextFrameChoiceKind,
    ConversationResolution,
    ConversationResolutionKind,
    ConversationResolutionResult,
    CurrentValueSurface,
    CurrentValueSurfaceKind,
    DependencyKind,
    MeaningComponent,
    MeaningComponentKind,
    RequestedValueFrame,
    SelectedFrameStatus,
    SourceEvidence,
    UnresolvedResolution,
)
from fervis.lookup.conversation_resolution.tools import (
    CONVERSATION_RESOLUTION_TOOL_NAME,
)


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
    _reject_unexpected_keys(
        payload,
        {
            "kind",
            "status",
            "current_question_text",
            "clause_resolutions",
            "unresolved",
        },
        "conversation_resolution",
    )
    if payload.get("kind") != "conversation_resolution":
        raise ValueError("invalid conversation resolution kind")
    resolution = _status(payload.get("status"))
    current_question_text = _current_question_text(
        payload=payload,
        current_question=current_question,
    )
    context = _ParseContext.from_inputs(
        current_question=current_question_text,
        context_sources=context_sources,
        context_frames=context_frames,
    )
    clause_resolutions = _clause_resolutions(
        payload.get("clause_resolutions"),
        context=context,
    )
    unresolved = _unresolved(
        payload.get("unresolved"),
        context=context,
    )
    used_sources = _used_sources(
        clause_resolutions=clause_resolutions,
        unresolved=unresolved,
        context=context,
    )
    return ConversationResolutionResult(
        outcome=ConversationResolution(
            resolution=resolution,
            current_question_text=current_question_text,
            clause_resolutions=clause_resolutions,
            unresolved=unresolved,
            used_source_card_ids=_source_card_ids(used_sources),
            used_memory_ids=_used_memory_ids(
                clause_resolutions=clause_resolutions,
                unresolved=unresolved,
                used_sources=used_sources,
            ),
        )
    )


@dataclass(frozen=True)
class _ParseContext:
    current_question: str
    sources: dict[str, "_SourceText"]
    frames: dict[str, ConversationContextFrame]

    @classmethod
    def from_inputs(
        cls,
        *,
        current_question: str,
        context_sources: tuple[ConversationContextSource, ...],
        context_frames: tuple[ConversationContextFrame, ...],
    ) -> "_ParseContext":
        sources = {
            "current_question": _SourceText(
                source_id="current_question",
                text=current_question,
            ),
            **{
                item.source_id: _source_text_from_context(item)
                for item in context_sources
            },
        }
        frames = _context_frames_by_id(
            context_frames=context_frames,
            sources=sources,
        )
        return cls(
            current_question=current_question,
            sources=sources,
            frames=frames,
        )


@dataclass(frozen=True)
class _SourceText:
    source_id: str
    text: str
    source_card_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    meaning_anchors: tuple[ConversationMeaningAnchor, ...] = ()


def _context_frames_by_id(
    *,
    context_frames: tuple[ConversationContextFrame, ...],
    sources: dict[str, _SourceText],
) -> dict[str, ConversationContextFrame]:
    output: dict[str, ConversationContextFrame] = {}
    for frame in context_frames:
        if frame.frame_id in output:
            raise ValueError("duplicate context frame id")
        for source_id in frame.source_ids:
            if source_id not in sources:
                raise ValueError("context frame references unavailable source")
        output[frame.frame_id] = frame
    return output


def _source_text_from_context(source: ConversationContextSource) -> _SourceText:
    return _SourceText(
        source_id=source.source_id,
        text=source.text,
        source_card_ids=source.source_card_ids,
        source_memory_ids=source.source_memory_ids,
        meaning_anchors=source.meaning_anchors,
    )


def _current_question_text(
    *,
    payload: dict[str, Any],
    current_question: str,
) -> str:
    text = _required_string(
        payload.get("current_question_text"),
        path="current_question_text",
    )
    expected = _required_string(current_question, path="current_question")
    if text != expected:
        raise ValueError("current_question_text must exactly match current question")
    return text


def _status(raw: Any) -> ConversationResolutionKind:
    try:
        return ConversationResolutionKind(str(raw))
    except ValueError as exc:
        raise ValueError("conversation resolution status is invalid") from exc


def _clause_resolutions(
    raw: Any,
    *,
    context: _ParseContext,
) -> tuple[ClauseResolution, ...]:
    return tuple(
        _clause_resolution(
            item,
            path=f"clause_resolutions[{index}]",
            context=context,
        )
        for index, item in enumerate(_required_dicts(raw, "clause_resolutions"))
    )


def _clause_resolution(
    item: dict[str, Any],
    *,
    path: str,
    context: _ParseContext,
) -> ClauseResolution:
    _reject_unexpected_keys(
        item,
        {
            "current_clause_text",
            "occurrence",
            "requested_value_frame",
            "dependencies",
            "resolved_clause_text",
        },
        path,
    )
    current_clause_text = _required_string(
        item.get("current_clause_text"),
        path=f"{path}.current_clause_text",
    )
    occurrence = _positive_int(item.get("occurrence"), path=f"{path}.occurrence")
    _require_occurrence(
        text=current_clause_text,
        occurrence=occurrence,
        source=context.current_question,
        path=f"{path}.current_clause_text",
    )
    resolved_clause_text = _required_string(
        item.get("resolved_clause_text"),
        path=f"{path}.resolved_clause_text",
    )
    requested_value_frame = _requested_value_frame(
        item.get("requested_value_frame"),
        context=context,
        path=f"{path}.requested_value_frame",
        current_clause_text=current_clause_text,
        resolved_clause_text=resolved_clause_text,
    )
    dependencies = _dependencies(
        item.get("dependencies"),
        context=context,
        current_clause_text=current_clause_text,
        resolved_clause_text=resolved_clause_text,
        path=f"{path}.dependencies",
    )
    return ClauseResolution(
        current_clause_text=current_clause_text,
        occurrence=occurrence,
        requested_value_frame=requested_value_frame,
        dependencies=dependencies,
        resolved_clause_text=resolved_clause_text,
    )


def _requested_value_frame(
    raw: Any,
    *,
    context: _ParseContext,
    path: str,
    current_clause_text: str,
    resolved_clause_text: str,
) -> RequestedValueFrame:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(
        item,
        {
            "current_value_surface",
            "context_frame_choices",
        },
        path,
    )
    current_value_surface = _current_value_surface(
        item.get("current_value_surface"),
        current_clause_text=current_clause_text,
        path=f"{path}.current_value_surface",
    )
    choices = _context_frame_choices(
        item.get("context_frame_choices"),
        context=context,
        current_clause_text=current_clause_text,
        current_value_surface=current_value_surface,
        path=f"{path}.context_frame_choices",
    )
    selected_status, selected_frame_id = _derive_selected_frame(
        choices=choices,
        path=path,
    )
    resolved_frame_text, preserve_terms = _derive_frame_text(
        current_value_surface=current_value_surface,
        selected_status=selected_status,
        selected_frame_id=selected_frame_id,
        context=context,
        path=path,
    )
    _require_terms(
        terms=preserve_terms,
        source=resolved_frame_text,
        resolved_clause_text=resolved_clause_text,
        path=f"{path}.must_preserve_terms",
    )
    return RequestedValueFrame(
        current_value_surface=current_value_surface,
        context_frame_choices=choices,
        selected_frame_status=selected_status,
        selected_context_frame_id=selected_frame_id,
        resolved_frame_text=resolved_frame_text,
        must_preserve_terms=preserve_terms,
    )


def _current_value_surface(
    raw: Any,
    *,
    current_clause_text: str,
    path: str,
) -> CurrentValueSurface:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(item, {"text", "kind"}, path)
    text = _required_string(item.get("text"), path=f"{path}.text")
    _require_occurrence(
        text=text,
        occurrence=1,
        source=current_clause_text,
        path=f"{path}.text",
    )
    return CurrentValueSurface(
        text=text,
        kind=CurrentValueSurfaceKind(
            _required_string(item.get("kind"), path=f"{path}.kind")
        ),
    )


def _context_frame_choices(
    raw: Any,
    *,
    context: _ParseContext,
    current_clause_text: str,
    current_value_surface: CurrentValueSurface,
    path: str,
) -> tuple[ContextFrameChoice, ...]:
    choices: list[ContextFrameChoice] = []
    seen: set[str] = set()
    for index, item in enumerate(_required_dicts(raw, path)):
        item_path = f"{path}[{index}]"
        _reject_unexpected_keys(
            item,
            {"frame_id", "choice", "current_conflict_quotes"},
            item_path,
        )
        frame_id = _required_string(item.get("frame_id"), path=f"{item_path}.frame_id")
        if frame_id not in context.frames:
            raise ValueError(f"{item_path}.frame_id is not an available frame")
        if frame_id in seen:
            raise ValueError(f"{item_path}.frame_id is duplicated")
        seen.add(frame_id)
        choice = ContextFrameChoiceKind(
            _required_string(item.get("choice"), path=f"{item_path}.choice")
        )
        conflict_quotes = _string_array(
            item.get("current_conflict_quotes"),
            path=f"{item_path}.current_conflict_quotes",
        )
        for quote_index, quote in enumerate(conflict_quotes):
            _require_occurrence(
                text=quote,
                occurrence=1,
                source=current_clause_text,
                path=f"{item_path}.current_conflict_quotes[{quote_index}]",
            )
        if (
            current_value_surface.kind == CurrentValueSurfaceKind.BROAD_CURRENT_VALUE
            and choice == ContextFrameChoiceKind.CURRENT_TEXT_NAMES_DIFFERENT_VALUE
        ):
            raise ValueError(
                f"{item_path}.choice cannot reject with broad current value text"
            )
        choices.append(
            ContextFrameChoice(
                frame_id=frame_id,
                choice=choice,
                current_conflict_quotes=conflict_quotes,
            )
        )
    expected = set(context.frames)
    if seen != expected:
        raise ValueError(f"{path} must decide every available context frame")
    return tuple(choices)


def _derive_selected_frame(
    *,
    choices: tuple[ContextFrameChoice, ...],
    path: str,
) -> tuple[SelectedFrameStatus, str | None]:
    used_ids = tuple(
        item.frame_id
        for item in choices
        if item.choice == ContextFrameChoiceKind.USE_FRAME
    )
    if not used_ids:
        return SelectedFrameStatus.LITERAL, None
    if len(used_ids) > 1:
        raise ValueError(f"{path} contextual frame must use exactly one frame")
    return SelectedFrameStatus.CONTEXTUAL, used_ids[0]


def _derive_frame_text(
    *,
    current_value_surface: CurrentValueSurface,
    selected_status: SelectedFrameStatus,
    selected_frame_id: str | None,
    context: _ParseContext,
    path: str,
) -> tuple[str, tuple[str, ...]]:
    if selected_status == SelectedFrameStatus.LITERAL:
        if selected_frame_id is not None:
            raise ValueError(f"{path} literal frame cannot select context frame")
        return current_value_surface.text, ()
    if selected_frame_id is None:
        raise ValueError(f"{path} contextual frame must select a used frame")
    selected_frame = context.frames[selected_frame_id]
    return selected_frame.requested_frame, (selected_frame.requested_frame,)


def _dependencies(
    raw: Any,
    *,
    context: _ParseContext,
    current_clause_text: str,
    resolved_clause_text: str,
    path: str,
) -> tuple[ClauseDependency, ...]:
    return tuple(
        _dependency(
            item,
            path=f"{path}[{index}]",
            context=context,
            current_clause_text=current_clause_text,
            resolved_clause_text=resolved_clause_text,
        )
        for index, item in enumerate(_required_dicts(raw, path))
    )


def _dependency(
    item: dict[str, Any],
    *,
    path: str,
    context: _ParseContext,
    current_clause_text: str,
    resolved_clause_text: str,
) -> ClauseDependency:
    _reject_unexpected_keys(
        item,
        {
            "anchor_text",
            "occurrence",
            "kind",
            "meaning_components",
            "resolved_text",
            "must_preserve_terms",
        },
        path,
    )
    anchor_text = _required_string(item.get("anchor_text"), path=f"{path}.anchor_text")
    occurrence = _positive_int(item.get("occurrence"), path=f"{path}.occurrence")
    _require_occurrence(
        text=anchor_text,
        occurrence=occurrence,
        source=current_clause_text,
        path=f"{path}.anchor_text",
    )
    resolved_text = _required_string(
        item.get("resolved_text"),
        path=f"{path}.resolved_text",
    )
    preserve_terms = _string_array(
        item.get("must_preserve_terms"),
        path=f"{path}.must_preserve_terms",
    )
    return ClauseDependency(
        anchor_text=anchor_text,
        occurrence=occurrence,
        kind=DependencyKind(_required_string(item.get("kind"), path=f"{path}.kind")),
        meaning_components=_meaning_components(
            item.get("meaning_components"),
            context=context,
            path=f"{path}.meaning_components",
        ),
        resolved_text=resolved_text,
        must_preserve_terms=preserve_terms,
    )


def _meaning_components(
    raw: Any,
    *,
    context: _ParseContext,
    path: str,
) -> tuple[MeaningComponent, ...]:
    output: list[MeaningComponent] = []
    for index, item in enumerate(_required_dicts(raw, path)):
        item_path = f"{path}[{index}]"
        _reject_unexpected_keys(
            item,
            {"kind", "source_id", "source_text", "memory_id", "resolved_text"},
            item_path,
        )
        source_id = _required_string(
            item.get("source_id"),
            path=f"{item_path}.source_id",
        )
        if source_id == "current_question":
            raise ValueError(
                f"{item_path}.source_id must reference a prior context source"
            )
        source = context.sources.get(source_id)
        if source is None:
            raise ValueError(f"{item_path}.source_id is not an available source")
        source_text = _required_string(
            item.get("source_text"),
            path=f"{item_path}.source_text",
        )
        if not _anchors_source_text(text=source_text, source_text=source.text):
            raise ValueError(f"{item_path}.source_text must appear in declared source")
        memory_id = _required_string(
            item.get("memory_id"),
            path=f"{item_path}.memory_id",
        )
        _require_meaning_anchor(
            source=source,
            memory_id=memory_id,
            source_text=source_text,
            path=item_path,
        )
        output.append(
            MeaningComponent(
                kind=MeaningComponentKind(
                    _required_string(item.get("kind"), path=f"{item_path}.kind")
                ),
                source_id=source_id,
                source_text=source_text,
                memory_id=memory_id,
                resolved_text=_required_string(
                    item.get("resolved_text"),
                    path=f"{item_path}.resolved_text",
                ),
            )
        )
    return tuple(output)


def _require_meaning_anchor(
    *,
    source: _SourceText,
    memory_id: str,
    source_text: str,
    path: str,
) -> None:
    del source_text
    for anchor in source.meaning_anchors:
        if anchor.memory_id == memory_id:
            return
    raise ValueError(f"{path}.memory_id must reference a meaning anchor on source_id")


def _unresolved(
    raw: Any,
    *,
    context: _ParseContext,
) -> UnresolvedResolution:
    item = _required_dict(raw, "unresolved")
    _reject_unexpected_keys(
        item,
        {"unresolved_kind", "why_unresolved", "candidate_interpretations"},
        "unresolved",
    )
    return UnresolvedResolution(
        unresolved_kind=_required_string(
            item.get("unresolved_kind"),
            path="unresolved.unresolved_kind",
        ),
        why_unresolved=_required_string(
            item.get("why_unresolved"),
            path="unresolved.why_unresolved",
            allow_empty=True,
        ),
        candidate_interpretations=_candidate_interpretations(
            item.get("candidate_interpretations"),
            context=context,
            path="unresolved.candidate_interpretations",
        ),
    )


def _candidate_interpretations(
    raw: Any,
    *,
    context: _ParseContext,
    path: str,
) -> tuple[CandidateInterpretation, ...]:
    output: list[CandidateInterpretation] = []
    for index, item in enumerate(_required_dicts(raw, path)):
        item_path = f"{path}[{index}]"
        _reject_unexpected_keys(
            item,
            {"integrated_question", "supporting_evidence"},
            item_path,
        )
        output.append(
            CandidateInterpretation(
                integrated_question=_required_string(
                    item.get("integrated_question"),
                    path=f"{item_path}.integrated_question",
                ),
                supporting_evidence=_source_evidence(
                    item.get("supporting_evidence"),
                    context=context,
                    path=f"{item_path}.supporting_evidence",
                ),
            )
        )
    return tuple(output)


def _source_evidence(
    raw: Any,
    *,
    context: _ParseContext,
    path: str,
) -> tuple[SourceEvidence, ...]:
    output: list[SourceEvidence] = []
    for index, item in enumerate(_required_dicts(raw, path)):
        item_path = f"{path}[{index}]"
        _reject_unexpected_keys(item, {"source_id", "exact_source_texts"}, item_path)
        source_id = _required_string(
            item.get("source_id"),
            path=f"{item_path}.source_id",
        )
        source = context.sources.get(source_id)
        if source is None:
            raise ValueError(f"{item_path}.source_id is not an available source")
        exact_source_texts = _string_array(
            item.get("exact_source_texts"),
            path=f"{item_path}.exact_source_texts",
        )
        for text_index, exact_text in enumerate(exact_source_texts):
            if not _anchors_source_text(text=exact_text, source_text=source.text):
                raise ValueError(
                    f"{item_path}.exact_source_texts[{text_index}] "
                    "must appear in declared source"
                )
        output.append(
            SourceEvidence(
                source_id=source_id,
                exact_source_texts=exact_source_texts,
            )
        )
    return tuple(output)


def _used_sources(
    *,
    clause_resolutions: tuple[ClauseResolution, ...],
    unresolved: UnresolvedResolution,
    context: _ParseContext,
) -> tuple[_SourceText, ...]:
    source_ids: list[str] = []
    for clause in clause_resolutions:
        for source_id in _selected_frame_source_ids(
            clause.requested_value_frame,
            context=context,
        ):
            _append_source_id(source_ids, source_id)
        for dependency in clause.dependencies:
            for component in dependency.meaning_components:
                _append_source_id(source_ids, component.source_id)
    for interpretation in unresolved.candidate_interpretations:
        for evidence in interpretation.supporting_evidence:
            _append_source_id(source_ids, evidence.source_id)
    return tuple(context.sources[source_id] for source_id in source_ids)


def _selected_frame_source_ids(
    value_frame: RequestedValueFrame,
    *,
    context: _ParseContext,
) -> tuple[str, ...]:
    if value_frame.selected_frame_status != SelectedFrameStatus.CONTEXTUAL:
        return ()
    selected_id = value_frame.selected_context_frame_id
    if selected_id is None:
        return ()
    return context.frames[selected_id].source_ids


def _append_source_id(source_ids: list[str], source_id: str) -> None:
    if source_id != "current_question" and source_id not in source_ids:
        source_ids.append(source_id)


def _source_card_ids(sources: tuple[_SourceText, ...]) -> tuple[str, ...]:
    output: list[str] = []
    for source in sources:
        for card_id in source.source_card_ids:
            if card_id not in output:
                output.append(card_id)
    return tuple(output)


def _used_memory_ids(
    *,
    clause_resolutions: tuple[ClauseResolution, ...],
    unresolved: UnresolvedResolution,
    used_sources: tuple[_SourceText, ...],
) -> tuple[str, ...]:
    output: list[str] = []
    for clause in clause_resolutions:
        if (
            clause.requested_value_frame.selected_frame_status
            == SelectedFrameStatus.CONTEXTUAL
        ):
            for source in used_sources:
                for memory_id in source.source_memory_ids:
                    if memory_id not in output:
                        output.append(memory_id)
        for dependency in clause.dependencies:
            for component in dependency.meaning_components:
                if component.memory_id not in output:
                    output.append(component.memory_id)
    for interpretation in unresolved.candidate_interpretations:
        for evidence in interpretation.supporting_evidence:
            for source in used_sources:
                if source.source_id != evidence.source_id:
                    continue
                for memory_id in source.source_memory_ids:
                    if memory_id not in output:
                        output.append(memory_id)
    return tuple(output)


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


def _require_terms(
    *,
    terms: tuple[str, ...],
    source: str,
    resolved_clause_text: str,
    path: str,
) -> None:
    for index, term in enumerate(terms):
        term_path = f"{path}[{index}]"
        if not _anchors_source_text(text=term, source_text=source):
            raise ValueError(f"{term_path} must appear in declared source text")


def _anchors_source_text(*, text: str, source_text: str) -> bool:
    text_anchor = _anchor_text(text)
    source_anchor = _anchor_text(source_text)
    if not text_anchor or not source_anchor:
        return False
    if text_anchor in source_anchor:
        return True
    text_first_word = text_anchor.split()[0]
    source_first_word = source_anchor.split()[0]
    return text_anchor.startswith(source_first_word) or source_anchor.startswith(
        text_first_word
    )


def _anchor_text(value: str) -> str:
    stripped = value.strip().rstrip("?.!").strip().lower()
    if not stripped:
        return ""
    return " ".join(stripped.split())


def _required_dict(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object")
    return raw


def _required_dicts(raw: Any, path: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be an array")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        output.append(item)
    return tuple(output)


def _required_string(raw: Any, *, path: str, allow_empty: bool = False) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{path} must be a string")
    if not allow_empty and not raw.strip():
        raise ValueError(f"{path} must not be empty")
    return raw


def _positive_int(raw: Any, *, path: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{path} must be a positive integer")
    return raw


def _string_array(raw: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be an array")
    return tuple(
        _required_string(item, path=f"{path}[{index}]")
        for index, item in enumerate(raw)
    )


def _reject_unexpected_keys(
    raw: dict[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(f"{path} contains unexpected keys: {', '.join(unexpected)}")
