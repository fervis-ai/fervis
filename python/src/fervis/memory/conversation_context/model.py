"""Conversation-resolution attribution input and activation handles."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Any

from fervis.memory.prior_requests import PriorRequestMemory, PriorRequestSlotBinding

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


class ConversationFramePartKind(StrEnum):
    ANSWER_SUBJECT = "answer_subject"
    ANSWER_OUTPUT = "answer_output"
    ENTITY_IDENTITY = "entity_identity"
    TIME_SCOPE = "time_scope"
    LIMIT = "limit"
    POPULATION_CONSTRAINT = "population_constraint"
    GROUPING = "grouping"


@dataclass(frozen=True)
class ConversationFramePart:
    part_id: str
    kind: ConversationFramePartKind
    text: str
    source_ref: str = ""

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
        }


@dataclass(frozen=True)
class ConversationFrameParameter:
    parameter_id: str
    part_id: str
    kind: ConversationFramePartKind
    current_text: str
    resolved_text: str
    field_label_text: str = ""
    value_meaning_hint: str = ""
    binding: PriorRequestSlotBinding | None = None

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.part_id.strip():
            raise ValueError("frame parameter requires stable identity")
        if self.kind not in {
            ConversationFramePartKind.ENTITY_IDENTITY,
            ConversationFramePartKind.TIME_SCOPE,
            ConversationFramePartKind.LIMIT,
        }:
            raise ValueError("frame parameter requires a bindable part kind")
        if not self.current_text.strip() or not self.resolved_text.strip():
            raise ValueError("frame parameter requires current and resolved values")
        if self.binding is not None and self.binding.kind.value != self.kind.value:
            raise ValueError("frame parameter binding kind does not match")

    def to_model_dict(self) -> dict[str, str]:
        payload = {
            "parameter_id": self.parameter_id,
            "part_id": self.part_id,
            "kind": self.kind.value,
            "current_text": self.current_text,
            "resolved_text": self.resolved_text,
        }
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        return payload


@dataclass(frozen=True)
class ConversationCallableSignature:
    base_run_id: str
    requested_fact_id: str
    parameters: tuple[ConversationFrameParameter, ...]

    def __post_init__(self) -> None:
        if not self.base_run_id.strip():
            raise ValueError("callable signature requires base_run_id")
        if not self.requested_fact_id.strip():
            raise ValueError("callable signature requires requested_fact_id")
        parameter_ids = tuple(item.parameter_id for item in self.parameters)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("callable signature contains duplicate parameters")
        part_ids = tuple(item.part_id for item in self.parameters)
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("callable signature binds one frame part more than once")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "parameters": [item.to_model_dict() for item in self.parameters],
        }


@dataclass(frozen=True)
class ConversationAnswerShape:
    expression_family: str
    output_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.expression_family.strip() or not self.output_roles:
            raise ValueError("conversation frame requires typed answer shape")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "expression_family": self.expression_family,
            "output_roles": list(self.output_roles),
        }


@dataclass(frozen=True)
class ConversationContextFrame:
    frame_id: str
    source_ids: tuple[str, ...]
    answer_shape: ConversationAnswerShape
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
            "answer_shape": self.answer_shape.to_model_dict(),
            "parts": [part.to_model_dict() for part in self.parts],
        }
        if self.callable is not None:
            payload["callable"] = self.callable.to_model_dict()
        return payload

    def control_key(self) -> tuple[object, ...]:
        return (
            self.answer_shape.expression_family,
            self.answer_shape.output_roles,
            tuple((part.part_id, part.kind.value) for part in self.parts),
            tuple(
                (
                    parameter.parameter_id,
                    parameter.part_id,
                    parameter.kind.value,
                )
                for parameter in (
                    self.callable.parameters if self.callable is not None else ()
                )
            ),
        )

    def control_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "answer_shape": self.answer_shape.to_model_dict(),
            "parts": [
                {
                    "part_id": part.part_id,
                    "kind": part.kind.value,
                }
                for part in self.parts
            ],
        }
        if self.callable is not None:
            payload["parameters"] = [
                {
                    "parameter_id": parameter.parameter_id,
                    "part_id": parameter.part_id,
                    "kind": parameter.kind.value,
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

    def frame(self, frame_id: str) -> ConversationContextFrame:
        for frame in self.context_frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(frame_id)
