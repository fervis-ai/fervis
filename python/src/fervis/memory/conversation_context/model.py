"""Conversation-resolution attribution input and activation handles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fervis.memory.prior_requests import PriorRequestMemory

VALID_CONTEXT_SOURCE_KINDS = frozenset(
    {
        "prior_user_question",
        "prior_fervis_answer",
        "active_clarification",
    }
)


@dataclass(frozen=True)
class ConversationMeaningAnchor:
    memory_id: str
    text: str
    occurrence: int
    kind: str
    label: str

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("meaning anchor requires memory_id")
        if not self.text.strip():
            raise ValueError("meaning anchor requires text")
        if self.occurrence < 1:
            raise ValueError("meaning anchor occurrence must be positive")
        if not self.kind.strip():
            raise ValueError("meaning anchor requires kind")
        if not self.label.strip():
            raise ValueError("meaning anchor requires label")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "occurrence": self.occurrence,
            "kind": self.kind,
            "label": self.label,
        }


@dataclass(frozen=True)
class ConversationContextSource:
    source_id: str
    kind: str
    text: str
    source_card_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    meaning_anchors: tuple[ConversationMeaningAnchor, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("context source requires source_id")
        if self.kind not in VALID_CONTEXT_SOURCE_KINDS:
            raise ValueError(f"unsupported context source kind: {self.kind}")
        if not self.text.strip():
            raise ValueError("context source requires text")
        if any(not card_id.strip() for card_id in self.source_card_ids):
            raise ValueError("context source_card_ids must be non-empty")
        if any(not memory_id.strip() for memory_id in self.source_memory_ids):
            raise ValueError("context source_memory_ids must be non-empty")
        seen: set[tuple[str, str, int]] = set()
        for anchor in self.meaning_anchors:
            key = (anchor.memory_id, anchor.text, anchor.occurrence)
            if key in seen:
                raise ValueError("duplicate meaning anchor")
            seen.add(key)

    def to_model_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "kind": self.kind,
            "text": self.text,
        }
        if self.meaning_anchors:
            payload["meaning_anchors"] = [
                anchor.to_model_dict() for anchor in self.meaning_anchors
            ]
        return payload


@dataclass(frozen=True)
class ConversationReplaceablePart:
    part_id: str
    kind: str
    text: str

    def __post_init__(self) -> None:
        if not self.part_id.strip():
            raise ValueError("replaceable part requires part_id")
        if not self.kind.strip():
            raise ValueError("replaceable part requires kind")
        if not self.text.strip():
            raise ValueError("replaceable part requires text")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "kind": self.kind,
            "text": self.text,
        }


@dataclass(frozen=True)
class ConversationContextFrame:
    frame_id: str
    source_ids: tuple[str, ...]
    requested_frame: str
    prior_answer_fact: str
    replaceable_parts: tuple[ConversationReplaceablePart, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("context frame requires frame_id")
        if not self.source_ids:
            raise ValueError("context frame requires source_ids")
        if any(not source_id.strip() for source_id in self.source_ids):
            raise ValueError("context frame source_ids must be non-empty")
        if not self.requested_frame.strip():
            raise ValueError("context frame requires requested_frame")
        if not self.prior_answer_fact.strip():
            raise ValueError("context frame requires prior_answer_fact")
        seen: set[str] = set()
        for part in self.replaceable_parts:
            if part.part_id in seen:
                raise ValueError("duplicate context frame replaceable part")
            seen.add(part.part_id)

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "source_ids": list(self.source_ids),
            "requested_frame": self.requested_frame,
            "prior_answer_fact": self.prior_answer_fact,
            "replaceable_parts": [
                part.to_model_dict() for part in self.replaceable_parts
            ],
        }


@dataclass(frozen=True)
class ConversationMemoryCard:
    card_id: str
    memory_id: str
    kind: str
    display: str
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.card_id.strip():
            raise ValueError("memory card requires card_id")
        if not self.memory_id.strip():
            raise ValueError("memory card requires memory_id")
        if not self.kind.strip():
            raise ValueError("memory card requires kind")
        if not self.display.strip():
            raise ValueError("memory card requires display")

    def to_model_dict(self, *, include_memory_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "card_id": self.card_id,
            "kind": self.kind,
            "display": self.display,
        }
        if include_memory_id:
            payload["memory_id"] = self.memory_id
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class ConversationMemoryActivationKind(StrEnum):
    PRIOR_REQUEST = "prior_answer_request"
    ROW_SET = "row_set"
    ENTITY_IDENTITY = "entity_identity"
    SCALAR_VALUE = "scalar_value"
    TIME_SCOPE = "time_scope"
    CLARIFICATION_ANSWER = "clarification_answer"


@dataclass(frozen=True)
class ConversationMemoryActivation:
    card: ConversationMemoryCard
    kind: ConversationMemoryActivationKind
    artifact_id: str
    address_id: str = ""
    prior_request: PriorRequestMemory | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("memory activation requires artifact identity")
        if self.card.kind != self.kind.value:
            raise ValueError("memory activation kind does not match its card")
        if self.kind is ConversationMemoryActivationKind.PRIOR_REQUEST:
            if (
                self.prior_request is None
                or self.address_id
                or self.prior_request.memory_id != self.card.memory_id
                or self.prior_request.artifact_id != self.artifact_id
            ):
                raise ValueError("prior-request activation contract is inconsistent")
            return
        if self.prior_request is not None or not self.address_id:
            raise ValueError("address activation contract is inconsistent")

    @property
    def memory_id(self) -> str:
        return self.card.memory_id


@dataclass(frozen=True)
class ConversationMemoryCardProjection:
    context_sources: tuple[ConversationContextSource, ...] = ()
    context_frames: tuple[ConversationContextFrame, ...] = ()
    cards: tuple[ConversationMemoryCard, ...] = ()
    activations: tuple[ConversationMemoryActivation, ...] = ()
    prior_requests: tuple[PriorRequestMemory, ...] = ()
    private_cards: dict[str, dict[str, Any]] | None = None
    omitted_counts_by_kind: dict[str, int] | None = None

    def private_card(self, memory_id: str) -> dict[str, Any]:
        private_cards = self.private_cards or {}
        if memory_id not in private_cards:
            raise KeyError(memory_id)
        return dict(private_cards[memory_id])

    def prior_request(self, memory_id: str) -> PriorRequestMemory:
        for request in self.prior_requests:
            if request.memory_id == memory_id:
                return request
        raise KeyError(memory_id)
