"""Conversation-resolution attribution input and activation handles."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Any

VALID_CONTEXT_SOURCE_KINDS = frozenset(
    {
        "prior_user_question",
        "prior_fervis_answer",
        "active_clarification",
    }
)


@dataclass(frozen=True)
class ConversationMeaningAnchor:
    anchor_id: str
    text: str
    occurrence: int
    kind: str
    label: str

    def __post_init__(self) -> None:
        if not self.anchor_id.strip():
            raise ValueError("meaning anchor requires anchor_id")
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
            "anchor_id": self.anchor_id,
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
            key = (anchor.anchor_id, anchor.text, anchor.occurrence)
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


class ConversationFramePartKind(StrEnum):
    SUBJECT = "subject"
    QUALIFICATION = "qualification"
    GROUPING = "grouping"
    REQUESTED_OUTPUT = "requested_output"
    ORDERING = "ordering"
    SELECTION = "selection"
    CANONICAL_OUTPUT_IDENTITY = "canonical_output_identity"
    INPUT = "input"


@dataclass(frozen=True)
class ConversationFramePart:
    part_id: str
    kind: ConversationFramePartKind
    text: str
    source_ref: str = ""
    value_type: str = ""

    def __post_init__(self) -> None:
        if not self.part_id.strip():
            raise ValueError("frame part requires part_id")
        if not isinstance(self.kind, ConversationFramePartKind):
            raise ValueError("frame part requires typed kind")
        if not self.text.strip():
            raise ValueError("frame part requires text")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "kind": self.kind.value,
            "text": self.text,
            **({"value_type": self.value_type} if self.value_type else {}),
        }


@dataclass(frozen=True)
class ConversationCallableParameter:
    parameter_id: str
    part_id: str
    value_type: str
    input_ref: str
    input_use_refs: tuple[str, ...]
    current_text: str

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.part_id.strip():
            raise ValueError("frame parameter requires stable identity")
        if not self.value_type or not self.input_ref or not self.input_use_refs:
            raise ValueError("frame parameter requires its typed input signature")
        if len(self.input_use_refs) != len(set(self.input_use_refs)):
            raise ValueError("frame parameter input uses must be unique")
        if not self.current_text.strip():
            raise ValueError("frame parameter requires current display text")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "part_id": self.part_id,
            "value_type": self.value_type,
            "input_ref": self.input_ref,
            "input_use_refs": list(self.input_use_refs),
            "current_text": self.current_text,
        }


@dataclass(frozen=True)
class ConversationCallableSignature:
    base_invocation_id: str
    program_id: str
    requested_fact_ref: str
    requested_fact_fingerprint: str
    parameters: tuple[ConversationCallableParameter, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.base_invocation_id.strip(),
                self.program_id.strip(),
                self.requested_fact_ref.strip(),
                self.requested_fact_fingerprint.strip(),
            )
        ):
            raise ValueError("callable signature requires canonical persisted identity")
        parameter_ids = tuple(item.parameter_id for item in self.parameters)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("callable signature contains duplicate parameters")
        part_ids = tuple(item.part_id for item in self.parameters)
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("callable signature binds one frame part more than once")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "base_invocation_id": self.base_invocation_id,
            "program_id": self.program_id,
            "requested_fact_ref": self.requested_fact_ref,
            "requested_fact_fingerprint": self.requested_fact_fingerprint,
            "parameters": [item.to_model_dict() for item in self.parameters],
        }


@dataclass(frozen=True)
class ConversationContextFrame:
    frame_id: str
    source_ids: tuple[str, ...]
    parts: tuple[ConversationFramePart, ...]
    callable: ConversationCallableSignature | None = None

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("context frame requires frame_id")
        if not self.source_ids:
            raise ValueError("context frame requires source_ids")
        if any(not source_id.strip() for source_id in self.source_ids):
            raise ValueError("context frame source_ids must be non-empty")
        seen: set[str] = set()
        for part in self.parts:
            if part.part_id in seen:
                raise ValueError("duplicate context frame part")
            seen.add(part.part_id)
        if self.callable is not None:
            available_part_ids = set(seen)
            if any(
                parameter.part_id not in available_part_ids
                for parameter in self.callable.parameters
            ):
                raise ValueError("callable parameter references unavailable frame part")

    def to_model_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_id": self.frame_id,
            "source_ids": list(self.source_ids),
            "parts": [part.to_model_dict() for part in self.parts],
        }
        if self.callable is not None:
            payload["callable"] = self.callable.to_model_dict()
        return payload

    def control_key(self) -> tuple[object, ...]:
        return (
            tuple(
                (part.part_id, part.kind.value, part.value_type)
                for part in self.parts
            ),
            tuple(
                (
                    parameter.parameter_id,
                    parameter.part_id,
                    parameter.value_type,
                    parameter.input_ref,
                    parameter.input_use_refs,
                )
                for parameter in (
                    self.callable.parameters if self.callable is not None else ()
                )
            ),
        )

    def control_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "parts": [
                {
                    "part_id": part.part_id,
                    "kind": part.kind.value,
                    **({"value_type": part.value_type} if part.value_type else {}),
                }
                for part in self.parts
            ],
        }
        if self.callable is not None:
            payload["parameters"] = [
                {
                    "parameter_id": parameter.parameter_id,
                    "part_id": parameter.part_id,
                    "value_type": parameter.value_type,
                    "input_ref": parameter.input_ref,
                    "input_use_refs": list(parameter.input_use_refs),
                }
                for parameter in self.callable.parameters
            ]
        return payload


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


@dataclass(frozen=True)
class ConversationMemoryActivation:
    card: ConversationMemoryCard
    kind: ConversationMemoryActivationKind
    artifact_id: str
    address_id: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("memory activation requires artifact identity")
        if self.card.kind != self.kind.value:
            raise ValueError("memory activation kind does not match its card")
        if self.kind is ConversationMemoryActivationKind.PRIOR_REQUEST:
            if self.address_id:
                raise ValueError("prior-request activation contract is inconsistent")
            return
        if not self.address_id:
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
    private_cards: dict[str, dict[str, Any]] | None = None
    omitted_counts_by_kind: dict[str, int] | None = None

    def private_card(self, memory_id: str) -> dict[str, Any]:
        private_cards = self.private_cards or {}
        if memory_id not in private_cards:
            raise KeyError(memory_id)
        return dict(private_cards[memory_id])

    def frame(self, frame_id: str) -> ConversationContextFrame:
        for frame in self.context_frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(frame_id)
