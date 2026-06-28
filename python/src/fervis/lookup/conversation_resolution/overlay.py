"""Conversation-resolution context overlay projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.memory.conversation_context import ConversationMemoryCardProjection
from fervis.lookup.conversation_resolution.model import (
    ClauseDependency,
    ConversationResolution,
    ContextFrameChoiceKind,
    DependencyKind,
)


@dataclass(frozen=True)
class ResolvedQuestionInputOverlay:
    kind: str
    reference_text: str
    occurrence: int
    target_meaning: str = ""
    lookup_text: str = ""
    resolved_input_ref: str = ""
    memory_ids: tuple[str, ...] = ()

    def to_prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "reference_text": self.reference_text,
            "occurrence": self.occurrence,
        }
        if self.target_meaning:
            payload["target_meaning"] = self.target_meaning
        if self.lookup_text:
            payload["lookup_text"] = self.lookup_text
        if self.resolved_input_ref:
            payload["resolved_input_ref"] = self.resolved_input_ref
        return payload


@dataclass(frozen=True)
class ConversationValueFrameOverlay:
    current_clause_text: str
    current_value_text: str
    current_value_kind: str
    resolved_frame_text: str
    must_preserve_terms: tuple[str, ...]
    used_context_frame_ids: tuple[str, ...]

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "current_clause_text": self.current_clause_text,
            "current_value_text": self.current_value_text,
            "current_value_kind": self.current_value_kind,
            "resolved_frame_text": self.resolved_frame_text,
            "must_preserve_terms": list(self.must_preserve_terms),
            "used_context_frame_ids": list(self.used_context_frame_ids),
        }


@dataclass(frozen=True)
class ConversationDependencyOverlay:
    current_clause_text: str
    anchor_text: str
    occurrence: int
    resolved_text: str
    must_preserve_terms: tuple[str, ...]
    source_ids: tuple[str, ...]
    memory_ids: tuple[str, ...] = ()

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "current_clause_text": self.current_clause_text,
            "anchor_text": self.anchor_text,
            "occurrence": self.occurrence,
            "resolved_text": self.resolved_text,
            "must_preserve_terms": list(self.must_preserve_terms),
            "source_ids": list(self.source_ids),
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class ConversationResolutionOverlay:
    current_question: str
    value_frames: tuple[ConversationValueFrameOverlay, ...]
    references: tuple[ConversationDependencyOverlay, ...]
    scopes: tuple[ConversationDependencyOverlay, ...]
    activated_memory_ids: tuple[str, ...]
    used_source_card_ids: tuple[str, ...]
    resolved_question_inputs: tuple[ResolvedQuestionInputOverlay, ...] = ()

    def to_prompt_payload(self) -> dict[str, Any]:
        payload = {
            "current_question": self.current_question,
            "value_frames": [item.to_prompt_payload() for item in self.value_frames],
            "references": [item.to_prompt_payload() for item in self.references],
            "scopes": [item.to_prompt_payload() for item in self.scopes],
            "activated_memory_ids": list(self.activated_memory_ids),
            "used_source_card_ids": list(self.used_source_card_ids),
        }
        if self.resolved_question_inputs:
            payload["resolved_question_inputs"] = [
                item.to_prompt_payload() for item in self.resolved_question_inputs
            ]
        return payload


def conversation_resolution_overlay_from(
    resolution: ConversationResolution,
    *,
    memory_projection: ConversationMemoryCardProjection | None = None,
) -> ConversationResolutionOverlay:
    value_frames: list[ConversationValueFrameOverlay] = []
    references: list[ConversationDependencyOverlay] = []
    scopes: list[ConversationDependencyOverlay] = []
    dependencies: list[ConversationDependencyOverlay] = []
    for clause in resolution.clause_resolutions:
        frame = clause.requested_value_frame
        value_frames.append(
            ConversationValueFrameOverlay(
                current_clause_text=clause.current_clause_text,
                current_value_text=frame.current_value_surface.text,
                current_value_kind=frame.current_value_surface.kind.value,
                resolved_frame_text=frame.resolved_frame_text,
                must_preserve_terms=frame.must_preserve_terms,
                used_context_frame_ids=tuple(
                    choice.frame_id
                    for choice in frame.context_frame_choices
                    if choice.choice == ContextFrameChoiceKind.USE_FRAME
                ),
            )
        )
        for dependency in clause.dependencies:
            overlay = _dependency_overlay(
                dependency,
                current_clause_text=clause.current_clause_text,
            )
            dependencies.append(overlay)
            if dependency.kind == DependencyKind.REFERENCE:
                references.append(overlay)
            elif dependency.kind == DependencyKind.SCOPE:
                scopes.append(overlay)
    return ConversationResolutionOverlay(
        current_question=resolution.current_question_text,
        value_frames=tuple(value_frames),
        references=tuple(references),
        scopes=tuple(scopes),
        activated_memory_ids=resolution.used_memory_ids,
        used_source_card_ids=resolution.used_source_card_ids,
        resolved_question_inputs=_resolved_question_inputs(
            dependencies=tuple(dependencies),
            references=tuple(references),
            memory_projection=memory_projection,
        ),
    )


def conversation_resolution_question_contract_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> dict[str, Any] | None:
    if overlay is None:
        return None

    payload = _value_frame_prompt_payload(overlay)
    if payload is None:
        return None

    if overlay.resolved_question_inputs:
        payload["resolved_question_inputs"] = [
            item.to_prompt_payload() for item in overlay.resolved_question_inputs
        ]

    return payload


def conversation_resolution_query_enrichment_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> dict[str, Any] | None:
    return _value_frame_prompt_payload(overlay)


def conversation_resolution_source_binding_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> dict[str, Any] | None:
    return _value_frame_prompt_payload(overlay)


def conversation_resolution_source_binding_evidence_texts(
    overlay: ConversationResolutionOverlay | None,
) -> dict[str, tuple[str, ...]]:
    if overlay is None:
        return {}
    return {
        f"value_frame_{index}": _value_frame_evidence_texts(item)
        for index, item in enumerate(overlay.value_frames, start=1)
    }


def conversation_resolution_question_contract_context_texts(
    overlay: ConversationResolutionOverlay | None,
) -> tuple[str, ...]:
    if overlay is None:
        return ()
    output: list[str] = []
    for item in overlay.value_frames:
        output.extend(_value_frame_evidence_texts(item))
    for item in overlay.resolved_question_inputs:
        _append_text(output, item.lookup_text)
        _append_text(output, item.reference_text)
    return tuple(dict.fromkeys(output))


def conversation_resolution_value_frame_instruction_lines() -> tuple[str, ...]:
    return (
        "If conversation resolution annotations are present, the current question "
        "contained conversation-dependent wording.",
        "Use those annotations as authoritative resolved meaning for the current question.",
        "current_value_text shows the user's surface wording; resolved_frame_text "
        "shows what that wording means in this conversation.",
        "value_frames determine the resolved requested value frame.",
    )


def _value_frame_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> dict[str, Any] | None:
    if overlay is None:
        return None
    return {
        "current_question": overlay.current_question,
        "value_frames": [
            {
                "annotation_id": f"value_frame_{index}",
                **item.to_prompt_payload(),
            }
            for index, item in enumerate(overlay.value_frames, start=1)
        ],
    }


def _value_frame_evidence_texts(
    item: ConversationValueFrameOverlay,
) -> tuple[str, ...]:
    values = (
        item.current_clause_text,
        item.current_value_text,
        item.current_value_kind,
        item.resolved_frame_text,
        *item.must_preserve_terms,
        *item.used_context_frame_ids,
    )
    return tuple(
        dict.fromkeys(text for value in values if (text := str(value).strip()))
    )


def _append_text(output: list[str], value: str) -> None:
    text = str(value or "").strip()
    if text:
        output.append(text)


def _dependency_overlay(
    dependency: ClauseDependency,
    *,
    current_clause_text: str,
) -> ConversationDependencyOverlay:
    return ConversationDependencyOverlay(
        current_clause_text=current_clause_text,
        anchor_text=dependency.anchor_text,
        occurrence=dependency.occurrence,
        resolved_text=dependency.resolved_text,
        must_preserve_terms=dependency.must_preserve_terms,
        source_ids=tuple(
            dict.fromkeys(
                component.source_id for component in dependency.meaning_components
            )
        ),
        memory_ids=tuple(
            dict.fromkeys(
                component.memory_id for component in dependency.meaning_components
            )
        ),
    )


def _resolved_question_inputs(
    *,
    dependencies: tuple[ConversationDependencyOverlay, ...],
    references: tuple[ConversationDependencyOverlay, ...],
    memory_projection: ConversationMemoryCardProjection | None,
) -> tuple[ResolvedQuestionInputOverlay, ...]:
    if memory_projection is None:
        return ()
    output: list[ResolvedQuestionInputOverlay] = []
    seen: set[tuple[str, int, str]] = set()
    for reference in dependencies:
        row_set_memory_id = _single_memory_id_by_kind(
            reference.memory_ids,
            kind="row_set",
            memory_projection=memory_projection,
        )
        if row_set_memory_id:
            key = (reference.anchor_text, reference.occurrence, row_set_memory_id)
            if key not in seen:
                seen.add(key)
                output.append(
                    ResolvedQuestionInputOverlay(
                        kind="row_set_reference",
                        reference_text=reference.anchor_text,
                        occurrence=reference.occurrence,
                        resolved_input_ref=f"cr_input_{len(output) + 1}",
                        memory_ids=(row_set_memory_id,),
                    )
                )
            continue
    for reference in references:
        entity_memory_id = _single_entity_memory_id(
            reference.memory_ids,
            memory_projection=memory_projection,
        )
        if not entity_memory_id:
            continue
        if reference.resolved_text not in reference.must_preserve_terms:
            continue
        key = (reference.anchor_text, reference.occurrence, reference.resolved_text)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            ResolvedQuestionInputOverlay(
                kind="named_reference_text",
                reference_text=reference.anchor_text,
                occurrence=reference.occurrence,
                lookup_text=_entity_lookup_text(
                    entity_memory_id,
                    default_text=reference.resolved_text,
                    memory_projection=memory_projection,
                ),
                target_meaning=_entity_target_meaning(
                    entity_memory_id,
                    memory_projection=memory_projection,
                ),
                memory_ids=(entity_memory_id,),
            )
        )
    return tuple(output)


def _single_memory_id_by_kind(
    memory_ids: tuple[str, ...],
    *,
    kind: str,
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    matches = tuple(
        memory_id
        for memory_id in memory_ids
        if _private_card(memory_id, memory_projection=memory_projection).get("kind")
        == kind
    )
    if len(matches) != 1:
        return ""
    return matches[0]


def _single_entity_memory_id(
    memory_ids: tuple[str, ...],
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    entity_ids = _entity_memory_ids(memory_projection)
    matches = tuple(memory_id for memory_id in memory_ids if memory_id in entity_ids)
    if len(matches) != 1:
        return ""
    return matches[0]


def _entity_target_meaning(
    memory_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    for card in memory_projection.cards:
        if card.memory_id != memory_id or card.kind != "entity_identity":
            continue
        private = {}
        try:
            private = memory_projection.private_card(card.memory_id)
        except KeyError:
            pass
        identity_type = str(private.get("identity_type") or "").strip()
        if identity_type:
            return f"{identity_type} identity"
    private = _private_card(memory_id, memory_projection=memory_projection)
    identity_type = str(private.get("identity_type") or "").strip()
    if identity_type:
        return f"{identity_type} identity"
    return "entity_identity"


def _entity_lookup_text(
    memory_id: str,
    *,
    default_text: str,
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    private = _private_card(memory_id, memory_projection=memory_projection)
    display = str(private.get("display") or "").strip()
    if display:
        return display
    for card in memory_projection.cards:
        if card.memory_id == memory_id and card.display.strip():
            return card.display
    return default_text


def _private_card(
    memory_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> dict[str, Any]:
    try:
        return memory_projection.private_card(memory_id)
    except KeyError:
        return {}


def _entity_memory_ids(
    memory_projection: ConversationMemoryCardProjection,
) -> set[str]:
    output = {
        card.memory_id
        for card in memory_projection.cards
        if card.kind == "entity_identity"
    }
    private_cards = memory_projection.private_cards or {}
    output.update(
        memory_id
        for memory_id, private in private_cards.items()
        if isinstance(private, dict) and private.get("kind") == "entity_identity"
    )
    return output
