"""Closed conversation-resolution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from typing import Any, TypeAlias

from fervis.memory.conversation_context import (
    ConversationContextFrame,
    ConversationContextSource,
)
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lookup.clarification.model import (
    ConversationResolutionResponse,
)


class ResolutionSourceKind(StrEnum):
    CURRENT_SPAN = "current_span"
    CONTEXT_ANCHOR = "context_anchor"
    FRAME_PART = "frame_part"


@dataclass(frozen=True)
class CurrentSpanSource:
    text: str
    occurrence: int
    kind: ResolutionSourceKind = ResolutionSourceKind.CURRENT_SPAN

    def __post_init__(self) -> None:
        if not self.text.strip() or self.occurrence < 1:
            raise ValueError("current-span source requires one copied occurrence")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "occurrence": self.occurrence,
        }

    def memory_references(self) -> tuple[str, ...]:
        return ()

    def frame_part_references(self) -> tuple[tuple[str, str], ...]:
        return ()

    def uses_prior_context(self) -> bool:
        return False

    def context_source_references(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class ContextAnchorSource:
    source_id: str
    anchor_id: str
    source_text: str
    memory_ids: tuple[str, ...] = ()
    kind: ResolutionSourceKind = ResolutionSourceKind.CONTEXT_ANCHOR

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.anchor_id.strip():
            raise ValueError("context-anchor source requires stable identity")
        if not self.source_text.strip():
            raise ValueError("context-anchor source requires copied source text")
        if any(not memory_id.strip() for memory_id in self.memory_ids):
            raise ValueError("context-anchor memory ids must be non-empty")

    def to_model_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "source_id": self.source_id,
            "anchor_id": self.anchor_id,
            "source_text": self.source_text,
        }

    def memory_references(self) -> tuple[str, ...]:
        return self.memory_ids

    def frame_part_references(self) -> tuple[tuple[str, str], ...]:
        return ()

    def uses_prior_context(self) -> bool:
        return True

    def context_source_references(self) -> tuple[str, ...]:
        return (self.source_id,)


@dataclass(frozen=True)
class FramePartSource:
    frame_id: str
    part_id: str
    kind: ResolutionSourceKind = ResolutionSourceKind.FRAME_PART

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.part_id.strip():
            raise ValueError("frame-part source requires stable identity")

    def to_model_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "frame_id": self.frame_id,
            "part_id": self.part_id,
        }

    def memory_references(self) -> tuple[str, ...]:
        return ()

    def frame_part_references(self) -> tuple[tuple[str, str], ...]:
        return ((self.frame_id, self.part_id),)

    def uses_prior_context(self) -> bool:
        return True

    def context_source_references(self) -> tuple[str, ...]:
        return ()


ResolutionSource: TypeAlias = CurrentSpanSource | ContextAnchorSource | FramePartSource


@dataclass(frozen=True)
class FrameParameterRef:
    frame_id: str
    parameter_id: str

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.parameter_id.strip():
            raise ValueError("frame parameter reference requires stable identity")

    def to_model_dict(self) -> dict[str, str]:
        return {
            "kind": "parameter",
            "frame_id": self.frame_id,
            "parameter_id": self.parameter_id,
        }


@dataclass(frozen=True)
class ResolvedConversationValue:
    value_id: str
    resolved_text: str
    sources: tuple[ResolutionSource, ...]
    frame_parameter: FrameParameterRef | None = None

    def __post_init__(self) -> None:
        if not self.value_id.strip() or not self.resolved_text.strip():
            raise ValueError("resolved value requires identity and standalone text")
        if not self.sources:
            raise ValueError("resolved value requires attributed sources")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "resolved_text": self.resolved_text,
            "frame_parameter": (
                self.frame_parameter.to_model_dict()
                if self.frame_parameter is not None
                else {"kind": "none"}
            ),
            "sources": [source.to_model_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class ResolvedConversationClause:
    current_clause_text: str
    occurrence: int
    resolved_text: str
    retained_frame_parts: tuple[FramePartSource, ...]
    values: tuple[ResolvedConversationValue, ...]

    def __post_init__(self) -> None:
        if not self.current_clause_text.strip() or self.occurrence < 1:
            raise ValueError("resolved clause requires one current-clause occurrence")
        if not self.resolved_text.strip():
            raise ValueError("resolved clause requires complete standalone text")
        value_ids = tuple(value.value_id for value in self.values)
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("resolved clause contains duplicate value ids")
        retained_refs = tuple(
            (part.frame_id, part.part_id) for part in self.retained_frame_parts
        )
        if len(retained_refs) != len(set(retained_refs)):
            raise ValueError("resolved clause contains duplicate retained frame parts")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "current_clause_text": self.current_clause_text,
            "occurrence": self.occurrence,
            "resolved_text": self.resolved_text,
            "retained_frame_parts": [
                part.to_model_dict() for part in self.retained_frame_parts
            ],
            "values": [value.to_model_dict() for value in self.values],
        }


class FrameArgumentKind(StrEnum):
    CARRY = "carry"
    RESOLVED_VALUE = "resolved_value"


@dataclass(frozen=True)
class CarriedFrameArgument:
    parameter_id: str
    kind: FrameArgumentKind = FrameArgumentKind.CARRY

    def __post_init__(self) -> None:
        if not self.parameter_id.strip():
            raise ValueError("carried frame argument requires parameter identity")

    def to_model_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "parameter_id": self.parameter_id}

    def resolved_value_ref(self) -> str:
        return ""


@dataclass(frozen=True)
class ResolvedValueFrameArgument:
    parameter_id: str
    value_id: str
    kind: FrameArgumentKind = FrameArgumentKind.RESOLVED_VALUE

    def __post_init__(self) -> None:
        if not self.parameter_id.strip() or not self.value_id.strip():
            raise ValueError("resolved frame argument requires parameter and value")

    def to_model_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "parameter_id": self.parameter_id,
            "value_id": self.value_id,
        }

    def resolved_value_ref(self) -> str:
        return self.value_id


FrameArgument: TypeAlias = CarriedFrameArgument | ResolvedValueFrameArgument


@dataclass(frozen=True)
class ConversationFrameCall:
    frame_id: str
    arguments: tuple[FrameArgument, ...]

    def __post_init__(self) -> None:
        if not self.frame_id.strip():
            raise ValueError("frame call requires frame identity")
        parameter_ids = tuple(argument.parameter_id for argument in self.arguments)
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("frame call contains duplicate arguments")

    def to_model_dict(self) -> dict[str, object]:
        return {
            "kind": "call",
            "frame_id": self.frame_id,
            "arguments": [argument.to_model_dict() for argument in self.arguments],
        }


@dataclass(frozen=True)
class SourceEvidence:
    source_id: str
    exact_source_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.exact_source_texts:
            raise ValueError("source evidence requires identity and copied text")
        if any(not text.strip() for text in self.exact_source_texts):
            raise ValueError("source evidence text must be non-empty")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exact_source_texts": list(self.exact_source_texts),
        }


@dataclass(frozen=True)
class CandidateInterpretation:
    contextualized_question: str
    context_evidence: tuple[SourceEvidence, ...]

    def __post_init__(self) -> None:
        if not self.contextualized_question.strip() or not self.context_evidence:
            raise ValueError("candidate interpretation requires text and evidence")

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "contextualized_question": self.contextualized_question,
            "context_evidence": [
                item.to_model_dict() for item in self.context_evidence
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
            raise ValueError("multiple meanings require at least two candidates")

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
    current_question_text: str
    resolution_basis: str
    contextualized_question: str
    clauses: tuple[ResolvedConversationClause, ...]
    frame_call: ConversationFrameCall | None = None
    unresolved: UnresolvedResolution = UnresolvedResolution(
        unresolved_kind="none",
        why_unresolved="",
        candidate_interpretations=(),
    )
    used_source_card_ids: tuple[str, ...] = ()
    used_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.current_question_text.strip():
            raise ValueError("conversation resolution requires current question")
        if self.unresolved.unresolved_kind == "none":
            if (
                not self.resolution_basis.strip()
                or not self.contextualized_question.strip()
                or not self.clauses
            ):
                raise ValueError("resolved conversation requires complete clauses")
        elif (
            self.resolution_basis
            or self.contextualized_question
            or self.clauses
            or self.frame_call is not None
        ):
            raise ValueError("unresolved conversation cannot contain a resolution")
        value_ids = tuple(
            value.value_id for clause in self.clauses for value in clause.values
        )
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("resolved question contains duplicate value ids")
        if self.frame_call is not None:
            referenced_values = {
                value_id
                for argument in self.frame_call.arguments
                for value_id in (argument.resolved_value_ref(),)
                if value_id
            }
            if not referenced_values.issubset(value_ids):
                raise ValueError("frame call references an unknown resolved value")
        if not self.needs_clarification and not self.uses_prior_context:
            if self.contextualized_question != self.current_question_text or any(
                clause.resolved_text != clause.current_clause_text
                for clause in self.clauses
            ):
                raise ValueError(
                    "a context-free question must pass through without rewriting"
                )

    @property
    def needs_clarification(self) -> bool:
        return self.unresolved.unresolved_kind != "none"

    @property
    def uses_prior_context(self) -> bool:
        return self.frame_call is not None or any(
            source.uses_prior_context()
            for clause in self.clauses
            for source in (
                *clause.retained_frame_parts,
                *(source for value in clause.values for source in value.sources),
            )
        )

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "kind": "conversation_resolution",
            "current_question_text": self.current_question_text,
            "resolution_basis": self.resolution_basis,
            "contextualized_question": self.contextualized_question,
            "clauses": [clause.to_model_dict() for clause in self.clauses],
            "frame_call": (
                self.frame_call.to_model_dict()
                if self.frame_call is not None
                else {"kind": "none"}
            ),
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
    clarification_responses: tuple[ConversationResolutionResponse, ...] = ()
