"""Conversation-resolution context overlay projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypeAlias, TypedDict

from fervis.memory.conversation_context import ConversationMemoryCardProjection
from fervis.lookup.conversation_resolution.model import (
    ClauseDependency,
    ConversationResolution,
    ContextFrameChoiceKind,
    DependencyKind,
)
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole


class ResolvedCanonicalIdentityPayload(TypedDict):
    kind: Literal["identity"]
    identity_type: str
    identity_field: str
    value: str
    authority_refs: list[str]
    lineage_refs: list[str]


class LiteralQuestionInputPromptPayload(TypedDict):
    kind: Literal["literal_text"]
    occurrence: int
    source_text: str
    resolved_input_ref: str
    resolved_value_text: str
    role: str
    value_meaning_hint: NotRequired[str]
    field_label_text: NotRequired[str]


class LiteralQuestionInputBackendPayload(LiteralQuestionInputPromptPayload):
    evidence_refs: NotRequired[list[str]]
    resolved_canonical_identity: NotRequired[ResolvedCanonicalIdentityPayload]


class RowSetQuestionInputPromptPayload(TypedDict):
    kind: Literal["row_set_reference"]
    reference_text: str
    occurrence: int
    resolved_input_ref: str


class RowSetQuestionInputBackendPayload(RowSetQuestionInputPromptPayload):
    memory_ids: NotRequired[list[str]]


ResolvedQuestionInputPromptPayload: TypeAlias = (
    LiteralQuestionInputPromptPayload | RowSetQuestionInputPromptPayload
)
ResolvedQuestionInputBackendPayload: TypeAlias = (
    LiteralQuestionInputBackendPayload | RowSetQuestionInputBackendPayload
)


class ConversationValueFramePayload(TypedDict):
    current_clause_text: str
    current_value_text: str
    current_value_kind: str
    resolved_frame_text: str
    must_preserve_terms: list[str]
    used_context_frame_ids: list[str]
    annotation_id: NotRequired[str]


class ConversationDependencyPayload(TypedDict):
    current_clause_text: str
    anchor_text: str
    occurrence: int
    resolved_text: str
    must_preserve_terms: list[str]
    source_ids: list[str]
    memory_ids: list[str]


class ConversationResolutionPromptPayload(TypedDict):
    current_question: str
    value_frames: list[ConversationValueFramePayload]
    references: list[ConversationDependencyPayload]
    scopes: list[ConversationDependencyPayload]
    activated_memory_ids: list[str]
    used_source_card_ids: list[str]
    resolved_question_inputs: NotRequired[list[ResolvedQuestionInputPromptPayload]]


class ConversationResolutionBackendPayload(ConversationResolutionPromptPayload):
    resolved_question_inputs: NotRequired[list[ResolvedQuestionInputBackendPayload]]


class ValueFramePromptPayload(TypedDict):
    current_question: str
    value_frames: list[ConversationValueFramePayload]


@dataclass(frozen=True)
class ResolvedCanonicalIdentityOverlay:
    identity_type: str
    identity_field: str
    value: str
    authority_refs: tuple[str, ...]
    lineage_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.identity_type.strip():
            raise ValueError("resolved canonical identity requires identity type")
        if not self.identity_field.strip():
            raise ValueError("resolved canonical identity requires identity field")
        if not self.value.strip():
            raise ValueError("resolved canonical identity requires value")
        if not self.authority_refs:
            raise ValueError("resolved canonical identity requires authority refs")
        if not all(_is_identity_authority_ref(ref) for ref in self.authority_refs):
            raise ValueError("resolved canonical identity authority refs are invalid")
        if not self.lineage_refs:
            raise ValueError("resolved canonical identity requires lineage refs")

    def to_payload(self) -> ResolvedCanonicalIdentityPayload:
        return {
            "kind": "identity",
            "identity_type": self.identity_type,
            "identity_field": self.identity_field,
            "value": self.value,
            "authority_refs": list(self.authority_refs),
            "lineage_refs": list(self.lineage_refs),
        }


@dataclass(frozen=True)
class LiteralQuestionInputOverlay:
    source_text: str
    resolved_input_ref: str
    resolved_value_text: str
    role: LiteralInputRole
    occurrence: int = 1
    value_meaning_hint: str = ""
    field_label_text: str = ""
    evidence_refs: tuple[str, ...] = ()
    resolved_canonical_identity: ResolvedCanonicalIdentityOverlay | None = None

    @property
    def kind(self) -> KnownInputKind:
        return KnownInputKind.LITERAL

    def __post_init__(self) -> None:
        if self.occurrence < 1:
            raise ValueError("resolved question input occurrence must be positive")
        if not self.source_text.strip():
            raise ValueError("literal resolved question input requires source text")
        if not self.resolved_input_ref.strip():
            raise ValueError("literal resolved question input requires resolved ref")
        if not self.resolved_value_text.strip():
            raise ValueError("literal resolved question input requires resolved value")

    def to_prompt_payload(self) -> LiteralQuestionInputPromptPayload:
        payload: LiteralQuestionInputPromptPayload = {
            "kind": self.kind.value,
            "occurrence": self.occurrence,
            "source_text": self.source_text,
            "resolved_input_ref": self.resolved_input_ref,
            "resolved_value_text": self.resolved_value_text,
            "role": self.role.value,
        }
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        return payload

    def to_backend_payload(self) -> LiteralQuestionInputBackendPayload:
        payload: LiteralQuestionInputBackendPayload = {**self.to_prompt_payload()}
        if self.evidence_refs:
            payload["evidence_refs"] = list(self.evidence_refs)
        if self.resolved_canonical_identity:
            payload["resolved_canonical_identity"] = (
                self.resolved_canonical_identity.to_payload()
            )
        return payload


@dataclass(frozen=True)
class RowSetQuestionInputOverlay:
    reference_text: str
    resolved_input_ref: str
    occurrence: int = 1
    memory_ids: tuple[str, ...] = ()

    @property
    def kind(self) -> KnownInputKind:
        return KnownInputKind.ROW_SET_REFERENCE

    def __post_init__(self) -> None:
        if self.occurrence < 1:
            raise ValueError("resolved question input occurrence must be positive")
        if not self.reference_text.strip():
            raise ValueError("row-set resolved question input requires reference text")
        if not self.resolved_input_ref.strip():
            raise ValueError("row-set resolved question input requires resolved ref")

    def to_prompt_payload(self) -> RowSetQuestionInputPromptPayload:
        return {
            "kind": self.kind.value,
            "reference_text": self.reference_text,
            "occurrence": self.occurrence,
            "resolved_input_ref": self.resolved_input_ref,
        }

    def to_backend_payload(self) -> RowSetQuestionInputBackendPayload:
        payload: RowSetQuestionInputBackendPayload = {**self.to_prompt_payload()}
        if self.memory_ids:
            payload["memory_ids"] = list(self.memory_ids)
        return payload


ResolvedQuestionInputOverlay: TypeAlias = (
    LiteralQuestionInputOverlay | RowSetQuestionInputOverlay
)


@dataclass(frozen=True)
class _CanonicalIdentity:
    field: str
    value: str


@dataclass(frozen=True)
class _EntityIdentityMemory:
    memory_id: str
    identity_type: str
    display: str
    canonical_identity: _CanonicalIdentity | None
    authority_refs: tuple[str, ...]

    @property
    def value_meaning_hint(self) -> str:
        return f"{self.identity_type} identity"

    def resolved_canonical_identity(self) -> ResolvedCanonicalIdentityOverlay | None:
        if self.canonical_identity is None or not self.authority_refs:
            return None
        return ResolvedCanonicalIdentityOverlay(
            identity_type=self.identity_type,
            identity_field=self.canonical_identity.field,
            value=self.canonical_identity.value,
            authority_refs=self.authority_refs,
            lineage_refs=(f"memory:{self.memory_id}",),
        )


@dataclass(frozen=True)
class _TimeScopeMemory:
    expression: str


@dataclass(frozen=True)
class ConversationValueFrameOverlay:
    current_clause_text: str
    current_value_text: str
    current_value_kind: str
    resolved_frame_text: str
    must_preserve_terms: tuple[str, ...]
    used_context_frame_ids: tuple[str, ...]

    def to_prompt_payload(self) -> ConversationValueFramePayload:
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

    def to_prompt_payload(self) -> ConversationDependencyPayload:
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

    def to_prompt_payload(self) -> ConversationResolutionPromptPayload:
        payload: ConversationResolutionPromptPayload = {
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

    def to_backend_payload(self) -> ConversationResolutionBackendPayload:
        payload: ConversationResolutionBackendPayload = {
            **self.to_prompt_payload(),
        }
        if self.resolved_question_inputs:
            payload["resolved_question_inputs"] = [
                item.to_backend_payload() for item in self.resolved_question_inputs
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
            scopes=tuple(scopes),
            memory_projection=memory_projection,
        ),
    )


def conversation_resolution_question_contract_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> ValueFramePromptPayload | None:
    return _value_frame_prompt_payload(overlay)


def conversation_resolution_query_enrichment_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> ValueFramePromptPayload | None:
    return _value_frame_prompt_payload(overlay)


def conversation_resolution_source_binding_prompt_payload(
    overlay: ConversationResolutionOverlay | None,
) -> ValueFramePromptPayload | None:
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
) -> ValueFramePromptPayload | None:
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
    scopes: tuple[ConversationDependencyOverlay, ...],
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
                    RowSetQuestionInputOverlay(
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
        key = (reference.anchor_text, reference.occurrence, reference.resolved_text)
        if key in seen:
            continue
        seen.add(key)
        entity_memory = _entity_identity_memory(
            entity_memory_id,
            memory_projection=memory_projection,
        )
        output.append(
            LiteralQuestionInputOverlay(
                source_text=reference.anchor_text,
                occurrence=reference.occurrence,
                resolved_input_ref=f"cr_input_{len(output) + 1}",
                resolved_value_text=entity_memory.display,
                value_meaning_hint=entity_memory.value_meaning_hint,
                role=LiteralInputRole.REFERENCE_VALUE,
                evidence_refs=(entity_memory_id,),
                resolved_canonical_identity=entity_memory.resolved_canonical_identity(),
            )
        )
    for scope in scopes:
        time_scope_memory_id = _single_memory_id_by_kind(
            scope.memory_ids,
            kind="time_scope",
            memory_projection=memory_projection,
        )
        if not time_scope_memory_id:
            continue
        key = (scope.anchor_text, scope.occurrence, scope.resolved_text)
        if key in seen:
            continue
        seen.add(key)
        time_scope_memory = _time_scope_memory(
            time_scope_memory_id,
            memory_projection=memory_projection,
        )
        output.append(
            LiteralQuestionInputOverlay(
                source_text=scope.anchor_text,
                occurrence=scope.occurrence,
                resolved_input_ref=f"cr_input_{len(output) + 1}",
                resolved_value_text=time_scope_memory.expression,
                value_meaning_hint="time scope",
                role=LiteralInputRole.TIME_VALUE,
                evidence_refs=(time_scope_memory_id,),
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
        if _memory_kind(memory_id, memory_projection=memory_projection) == kind
    )
    if len(matches) != 1:
        return ""
    return matches[0]


def _memory_kind(
    memory_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> str:
    for card in memory_projection.cards:
        if card.memory_id == memory_id:
            return card.kind
    private = memory_projection.private_card(memory_id)
    return str(private.get("kind") or "").strip()


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


def _entity_identity_memory(
    memory_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> _EntityIdentityMemory:
    private = memory_projection.private_card(memory_id)
    return _EntityIdentityMemory(
        memory_id=memory_id,
        identity_type=_required_memory_text(
            private.get("identity_type"),
            "entity memory input requires identity type",
        ),
        display=_required_memory_text(
            private.get("display"),
            "entity memory input requires display",
        ),
        canonical_identity=_canonical_identity(private.get("canonical_values")),
        authority_refs=_authority_refs(private.get("proof_refs")),
    )


def _time_scope_memory(
    memory_id: str,
    *,
    memory_projection: ConversationMemoryCardProjection,
) -> _TimeScopeMemory:
    private = memory_projection.private_card(memory_id)
    return _TimeScopeMemory(
        expression=_required_memory_text(
            private.get("expression"),
            "time scope memory input requires expression",
        ),
    )


def _required_memory_text(raw_value: object, error_message: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError(error_message)
    return text


def _canonical_identity(raw_value: object) -> _CanonicalIdentity | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("entity memory canonical values must be an object")
    canonical_values = tuple(
        _CanonicalIdentity(field=field, value=value)
        for raw_field, raw_item in raw_value.items()
        if (field := str(raw_field or "").strip())
        and (value := str(raw_item or "").strip())
    )
    if len(canonical_values) != 1:
        return None
    return canonical_values[0]


def _authority_refs(raw_value: object) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str) or not isinstance(raw_value, (list, tuple)):
        raise ValueError("entity memory proof refs must be a sequence")
    return tuple(
        ref
        for raw_item in raw_value
        if (ref := str(raw_item or "").strip()) and _is_identity_authority_ref(ref)
    )


def _is_identity_authority_ref(ref: str) -> bool:
    return ref.startswith("source_read:") or ref.startswith("prior_source_read:")


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
