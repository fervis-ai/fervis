"""Typed prior-question continuation plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fervis.memory.prior_requests import PriorRequestSlotBinding
from fervis.memory.conversation_context import ConversationReplaceablePart


class ContinuationPlanKind(StrEnum):
    NONE = "none"
    SHAPE_CHANGING = "shape_changing"
    SAME_FACT_INPUT_REPLACEMENT = "same_fact_input_replacement"


@dataclass(frozen=True)
class ContinuationReplacement:
    part: ConversationReplaceablePart
    current_text: str

    def __post_init__(self) -> None:
        if not self.current_text.strip():
            raise ValueError("continuation replacement requires current_text")

    @property
    def part_id(self) -> str:
        return self.part.part_id

    def to_payload(self) -> dict[str, object]:
        payload = self.part.to_model_dict()
        payload["prior_text"] = self.part.text
        payload["current_text"] = self.current_text
        return payload


@dataclass(frozen=True)
class ContinuationCarriedInput:
    part: ConversationReplaceablePart
    resolved_value_text: str = ""
    field_label_text: str = ""
    value_meaning_hint: str = ""
    binding: PriorRequestSlotBinding | None = None

    @property
    def part_id(self) -> str:
        return self.part.part_id

    def to_payload(self) -> dict[str, object]:
        payload = self.part.to_model_dict()
        if self.resolved_value_text:
            payload["resolved_value_text"] = self.resolved_value_text
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        if self.binding is not None:
            payload["binding"] = self.binding.to_payload()
        return payload


@dataclass(frozen=True)
class ContinuationPlan:
    kind: ContinuationPlanKind
    current_question: str = ""
    resolved_request_text: str = ""
    frame_id: str = ""
    prior_answer_fact: str = ""
    replacements: tuple[ContinuationReplacement, ...] = ()
    carried_inputs: tuple[ContinuationCarriedInput, ...] = ()

    @classmethod
    def none(cls) -> "ContinuationPlan":
        return cls(kind=ContinuationPlanKind.NONE)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ContinuationPlanKind):
            object.__setattr__(self, "kind", ContinuationPlanKind(str(self.kind)))
        if self.kind == ContinuationPlanKind.NONE:
            return
        if not self.current_question.strip():
            raise ValueError("continuation plan requires current_question")
        if not self.resolved_request_text.strip():
            raise ValueError("continuation plan requires resolved_request_text")
        if not self.frame_id.strip():
            raise ValueError("continuation plan requires frame_id")
        if not self.replacements:
            raise ValueError("continuation plan requires replacements")

    @property
    def has_continuation(self) -> bool:
        return self.kind != ContinuationPlanKind.NONE

    def to_inspection_payload(self) -> dict[str, Any]:
        if not self.has_continuation:
            return {"kind": self.kind.value}
        replacement_payloads = [item.to_payload() for item in self.replacements]
        carried_input_payloads = [item.to_payload() for item in self.carried_inputs]
        return {
            "kind": self.kind.value,
            "current_question": self.current_question,
            "resolved_request_text": self.resolved_request_text,
            "frame_id": self.frame_id,
            "prior_answer_fact": self.prior_answer_fact,
            "replacements": replacement_payloads,
            "carried_inputs": carried_input_payloads,
        }
