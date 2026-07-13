"""Typed provider-output contracts for conversation resolution."""

from dataclasses import dataclass

from fervis.lookup.provider_contract import ProviderObject, ProviderOutput


@dataclass(frozen=True)
class FramePartSourceOutput(ProviderOutput):
    kind: str
    frame_id: str
    part_id: str


@dataclass(frozen=True)
class CurrentSpanSourceOutput(ProviderOutput):
    kind: str
    text: str
    occurrence: int


@dataclass(frozen=True)
class ContextAnchorSourceOutput(ProviderOutput):
    kind: str
    source_id: str
    memory_id: str
    source_text: str


@dataclass(frozen=True)
class ResolvedValueOutput(ProviderOutput):
    value_id: str
    resolved_text: str
    frame_parameter: ProviderObject
    sources: tuple[ProviderObject, ...]


@dataclass(frozen=True)
class ResolvedClauseOutput(ProviderOutput):
    current_clause_text: str
    occurrence: int
    resolved_text: str
    retained_frame_parts: tuple[FramePartSourceOutput, ...]
    values: tuple[ResolvedValueOutput, ...]


@dataclass(frozen=True)
class NoFrameParameterOutput(ProviderOutput):
    kind: str


@dataclass(frozen=True)
class FrameParameterOutput(ProviderOutput):
    kind: str
    frame_id: str
    parameter_id: str


@dataclass(frozen=True)
class SourceEvidenceOutput(ProviderOutput):
    source_id: str
    exact_source_texts: tuple[str, ...]


@dataclass(frozen=True)
class CandidateInterpretationOutput(ProviderOutput):
    contextualized_question: str
    context_evidence: tuple[SourceEvidenceOutput, ...]


@dataclass(frozen=True)
class UnresolvedOutcomeOutput(ProviderOutput):
    kind: str
    why_unresolved: str
    candidate_interpretations: tuple[CandidateInterpretationOutput, ...]


@dataclass(frozen=True)
class ResolvedOutcomeOutput(ProviderOutput):
    kind: str
    resolution_basis: str
    contextualized_question: str
    clauses: tuple[ResolvedClauseOutput, ...]


@dataclass(frozen=True)
class ConversationResolutionOutput(ProviderOutput):
    kind: str
    current_question_text: str
    outcome: ProviderObject
