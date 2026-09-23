"""Typed incomplete-question outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.types.enums import StrEnum


class IncompleteFactualRequestKind(StrEnum):
    UNRESOLVED_PRIOR_TURN_REFERENCE = "unresolved_prior_turn_reference"
    MISSING_REQUESTED_FACT = "missing_requested_fact"


@dataclass(frozen=True)
class IncompleteFactualRequestItem:
    missing_kind: IncompleteFactualRequestKind
    source_text: str
    why_question_is_incomplete: str
    target_label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.missing_kind, IncompleteFactualRequestKind):
            raise ValueError("incomplete factual request requires structured kind")
        if not self.source_text.strip():
            raise ValueError("incomplete factual request requires source text")
        if not self.why_question_is_incomplete.strip():
            raise ValueError("incomplete factual request requires a reason")
        if (
            self.missing_kind
            is IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
            and not self.target_label.strip()
        ):
            raise ValueError("missing target reference requires target label")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "missing_kind": self.missing_kind.value,
            "source_text": self.source_text,
            "target_label": self.target_label or None,
            "why_question_is_incomplete": self.why_question_is_incomplete,
        }


@dataclass(frozen=True)
class QuestionContractNeedsClarification:
    missing: tuple[IncompleteFactualRequestItem, ...]

    def __post_init__(self) -> None:
        if not self.missing:
            raise ValueError("question-contract clarification requires missing inputs")
        kinds = {item.missing_kind for item in self.missing}
        if len(kinds) != 1:
            raise ValueError("question-contract clarification cannot mix missing kinds")
        if (
            IncompleteFactualRequestKind.MISSING_REQUESTED_FACT in kinds
            and len(self.missing) != 1
        ):
            raise ValueError("missing requested fact must be singular")

    def to_model_dict(self) -> dict[str, object]:
        if all(
            item.missing_kind
            is IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
            for item in self.missing
        ):
            return {
                "kind": "unresolved_prior_turn_references",
                "references": [
                    {
                        "source_text": item.source_text,
                        "target_label": item.target_label,
                        "why_question_is_incomplete": (item.why_question_is_incomplete),
                    }
                    for item in self.missing
                ],
            }
        item = self.missing[0]
        return {
            "kind": "missing_requested_fact",
            "source_text": item.source_text,
            "why_question_is_incomplete": item.why_question_is_incomplete,
        }


__all__ = [
    "IncompleteFactualRequestItem",
    "IncompleteFactualRequestKind",
    "QuestionContractNeedsClarification",
]
