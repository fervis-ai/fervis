"""Conversation-resolution contract models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
)
from fervis.lookup.turn_prompts.context import HostPromptContext


class ConversationResolutionKind(StrEnum):
    STANDALONE = "standalone"
    RESOLVED = "resolved"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    exact_source_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source evidence requires source_id")
        if not self.exact_source_texts:
            raise ValueError("source evidence requires exact_source_texts")
        if any(not text.strip() for text in self.exact_source_texts):
            raise ValueError("source evidence exact_source_texts must be non-empty")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exact_source_texts": list(self.exact_source_texts),
        }


class SelectedFrameStatus(StrEnum):
    LITERAL = "literal"
    CONTEXTUAL = "contextual"


class DependencyKind(StrEnum):
    REFERENCE = "reference"
    SCOPE = "scope"


class MeaningComponentKind(StrEnum):
    ENTITY = "entity"
    SCOPE = "scope"
    ROW_SET = "row_set"
    VALUE = "value"
    OTHER = "other"


class CurrentValueSurfaceKind(StrEnum):
    SELF_SUFFICIENT_CURRENT_VALUE = "self_sufficient_current_value"
    BROAD_CURRENT_VALUE = "broad_current_value"
    NO_VALUE_REQUEST = "no_value_request"


class ContextFrameChoiceKind(StrEnum):
    USE_FRAME = "use_frame"
    CURRENT_TEXT_NAMES_DIFFERENT_VALUE = "current_text_names_different_value"
    NOT_FOR_THIS_CLAUSE = "not_for_this_clause"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CurrentValueSurface:
    text: str
    kind: CurrentValueSurfaceKind

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("current value surface requires text")
        if not isinstance(self.kind, CurrentValueSurfaceKind):
            object.__setattr__(self, "kind", CurrentValueSurfaceKind(str(self.kind)))

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class ContextFrameChoice:
    frame_id: str
    choice: ContextFrameChoiceKind
    current_conflict_quotes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("context frame choice requires frame_id")
        if not isinstance(self.choice, ContextFrameChoiceKind):
            object.__setattr__(self, "choice", ContextFrameChoiceKind(str(self.choice)))
        if (
            self.choice == ContextFrameChoiceKind.CURRENT_TEXT_NAMES_DIFFERENT_VALUE
            and not self.current_conflict_quotes
        ):
            raise ValueError("current-text value rejection requires conflict quotes")
        if (
            self.choice != ContextFrameChoiceKind.CURRENT_TEXT_NAMES_DIFFERENT_VALUE
            and self.current_conflict_quotes
        ):
            raise ValueError("only current-text value rejection can include quotes")
        if any(not text.strip() for text in self.current_conflict_quotes):
            raise ValueError("current conflict quotes must be non-empty")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "choice": self.choice.value,
            "current_conflict_quotes": list(self.current_conflict_quotes),
        }


@dataclass(frozen=True)
class RequestedValueFrame:
    current_value_surface: CurrentValueSurface
    context_frame_choices: tuple[ContextFrameChoice, ...]
    selected_frame_status: SelectedFrameStatus
    selected_context_frame_id: str | None
    resolved_frame_text: str
    must_preserve_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.selected_frame_status, SelectedFrameStatus):
            object.__setattr__(
                self,
                "selected_frame_status",
                SelectedFrameStatus(str(self.selected_frame_status)),
            )
        if self.selected_frame_status == SelectedFrameStatus.CONTEXTUAL:
            if not str(self.selected_context_frame_id or "").strip():
                raise ValueError("contextual value frame requires selected frame")
            if not self.resolved_frame_text.strip():
                raise ValueError("contextual value frame requires resolved text")
            if not self.must_preserve_terms:
                raise ValueError("contextual value frame requires preserve terms")
        if self.selected_frame_status == SelectedFrameStatus.LITERAL:
            if self.selected_context_frame_id is not None:
                raise ValueError("literal value frame cannot select context frame")
            if self.resolved_frame_text != self.current_value_surface.text:
                raise ValueError(
                    "literal value frame resolved text must match current value text"
                )
            if self.must_preserve_terms:
                raise ValueError("literal value frame cannot require preserve terms")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "current_value_surface": self.current_value_surface.to_model_dict(),
            "context_frame_choices": [
                item.to_model_dict() for item in self.context_frame_choices
            ],
        }


@dataclass(frozen=True)
class MeaningComponent:
    kind: MeaningComponentKind
    source_id: str
    source_text: str
    memory_id: str
    resolved_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MeaningComponentKind):
            object.__setattr__(self, "kind", MeaningComponentKind(str(self.kind)))
        if not self.source_id.strip():
            raise ValueError("meaning component requires source_id")
        if not self.source_text.strip():
            raise ValueError("meaning component requires source_text")
        if not self.memory_id.strip():
            raise ValueError("meaning component requires memory_id")
        if not self.resolved_text.strip():
            raise ValueError("meaning component requires resolved_text")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source_id": self.source_id,
            "source_text": self.source_text,
            "memory_id": self.memory_id,
            "resolved_text": self.resolved_text,
        }


@dataclass(frozen=True)
class ClauseDependency:
    anchor_text: str
    occurrence: int
    kind: DependencyKind
    meaning_components: tuple[MeaningComponent, ...]
    resolved_text: str
    must_preserve_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.anchor_text.strip():
            raise ValueError("dependency requires anchor_text")
        if self.occurrence < 1:
            raise ValueError("dependency occurrence must be positive")
        if not isinstance(self.kind, DependencyKind):
            object.__setattr__(self, "kind", DependencyKind(str(self.kind)))
        if not self.meaning_components:
            raise ValueError("dependency requires meaning_components")
        if not self.resolved_text.strip():
            raise ValueError("dependency requires resolved_text")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "anchor_text": self.anchor_text,
            "occurrence": self.occurrence,
            "kind": self.kind.value,
            "meaning_components": [
                item.to_model_dict() for item in self.meaning_components
            ],
            "resolved_text": self.resolved_text,
            "must_preserve_terms": list(self.must_preserve_terms),
        }


@dataclass(frozen=True)
class ClauseResolution:
    current_clause_text: str
    occurrence: int
    requested_value_frame: RequestedValueFrame
    dependencies: tuple[ClauseDependency, ...]
    resolved_clause_text: str

    def __post_init__(self) -> None:
        if not self.current_clause_text.strip():
            raise ValueError("clause resolution requires current_clause_text")
        if self.occurrence < 1:
            raise ValueError("clause resolution occurrence must be positive")
        if not self.resolved_clause_text.strip():
            raise ValueError("clause resolution requires resolved_clause_text")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "current_clause_text": self.current_clause_text,
            "occurrence": self.occurrence,
            "requested_value_frame": self.requested_value_frame.to_model_dict(),
            "dependencies": [item.to_model_dict() for item in self.dependencies],
            "resolved_clause_text": self.resolved_clause_text,
        }


@dataclass(frozen=True)
class CandidateInterpretation:
    integrated_question: str
    supporting_evidence: tuple[SourceEvidence, ...]

    def __post_init__(self) -> None:
        if not self.integrated_question.strip():
            raise ValueError("candidate interpretation requires integrated_question")
        if not self.supporting_evidence:
            raise ValueError("candidate interpretation requires supporting_evidence")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "integrated_question": self.integrated_question,
            "supporting_evidence": [
                item.to_model_dict() for item in self.supporting_evidence
            ],
        }


@dataclass(frozen=True)
class UnresolvedResolution:
    unresolved_kind: str
    why_unresolved: str
    candidate_interpretations: tuple[CandidateInterpretation, ...]

    def __post_init__(self) -> None:
        if self.unresolved_kind not in {"none", "multiple_meanings", "missing_input"}:
            raise ValueError("unsupported unresolved kind")
        if self.unresolved_kind == "none":
            if self.why_unresolved or self.candidate_interpretations:
                raise ValueError("resolved unresolved payload must be empty")
            return
        if not self.why_unresolved.strip():
            raise ValueError("unresolved meaning requires why_unresolved")
        if (
            self.unresolved_kind == "multiple_meanings"
            and len(self.candidate_interpretations) < 2
        ):
            raise ValueError(
                "multiple meanings require at least two candidate interpretations"
            )

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "unresolved_kind": self.unresolved_kind,
            "why_unresolved": self.why_unresolved,
            "candidate_interpretations": [
                item.to_model_dict() for item in self.candidate_interpretations
            ],
        }


@dataclass(frozen=True)
class ConversationResolution:
    resolution: ConversationResolutionKind
    current_question_text: str
    clause_resolutions: tuple[ClauseResolution, ...] = ()
    unresolved: UnresolvedResolution = UnresolvedResolution(
        unresolved_kind="none",
        why_unresolved="",
        candidate_interpretations=(),
    )
    used_source_card_ids: tuple[str, ...] = ()
    used_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, ConversationResolutionKind):
            object.__setattr__(
                self,
                "resolution",
                ConversationResolutionKind(str(self.resolution)),
            )
        if not self.current_question_text.strip():
            raise ValueError("conversation resolution requires current_question_text")
        if self.resolution == ConversationResolutionKind.STANDALONE:
            if self.clause_resolutions:
                raise ValueError("standalone resolution cannot include clauses")
            if self.unresolved.unresolved_kind != "none":
                raise ValueError("standalone resolution cannot include unresolved")
        elif self.resolution == ConversationResolutionKind.RESOLVED:
            if not self.clause_resolutions:
                raise ValueError("resolved conversation requires clause_resolutions")
            if self.unresolved.unresolved_kind != "none":
                raise ValueError("resolved conversation cannot include unresolved")
        elif self.resolution == ConversationResolutionKind.NEEDS_CLARIFICATION:
            if self.clause_resolutions:
                raise ValueError(
                    "needs_clarification cannot include clause_resolutions"
                )
            if self.unresolved.unresolved_kind == "none":
                raise ValueError("needs_clarification requires unresolved meanings")
        else:
            raise ValueError("unsupported conversation resolution")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "kind": "conversation_resolution",
            "status": self.resolution.value,
            "current_question_text": self.current_question_text,
            "clause_resolutions": [
                item.to_model_dict() for item in self.clause_resolutions
            ],
            "unresolved": self.unresolved.to_model_dict(),
        }

    def activation_payload(self) -> dict[str, Any]:
        if not self.used_memory_ids:
            return {}
        return {
            "used_source_card_ids": list(self.used_source_card_ids),
            "activated_memory_ids": list(self.used_memory_ids),
        }


@dataclass(frozen=True)
class ConversationResolutionResult:
    outcome: ConversationResolution


@dataclass(frozen=True)
class ConversationResolutionRequest:
    question: str
    conversation_context: dict[str, Any]
    host: HostPromptContext = field(default_factory=HostPromptContext)
    context_sources: tuple[ConversationContextSource, ...] = ()
    context_frames: tuple[ConversationContextFrame, ...] = ()
