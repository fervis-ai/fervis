"""Typed causes for canonical lookup clarifications."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.clarification.model import (
    Clarification,
    ClarificationEvidence,
    ClarificationEvidenceKind,
    ClarificationNeed,
    ClarificationOption,
    ClarificationOwner,
    ClarificationReason,
    ClarificationSubject,
    ClarificationSubjectKind,
    ConversationInterpretationCandidate,
    ConversationResolutionContinuation,
    FactPlanningCatalogInputContinuation,
    GroundingContinuation,
    QuestionContractContinuation,
    SourceBindingCatalogInputContinuation,
)


@dataclass(frozen=True)
class TargetReferenceNotFound:
    clarification_id: str
    requested_fact_id: str
    known_input_id: str
    source_text: str
    target_label: str
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetReferenceAmbiguous:
    clarification_id: str
    requested_fact_id: str
    known_input_id: str
    source_text: str
    target_label: str
    options: tuple[ClarificationOption, ...]
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetReferenceUnsupported:
    clarification_id: str
    requested_fact_id: str
    known_input_id: str
    source_text: str
    target_label: str
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class MissingAnswerMetric:
    clarification_id: str
    requested_fact_id: str
    source_text: str
    metric_needed: str
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class MissingCatalogRequiredValue:
    clarification_id: str
    requested_fact_id: str
    required_input_id: str
    label: str
    continuation: (
        SourceBindingCatalogInputContinuation
        | FactPlanningCatalogInputContinuation
    )
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class MissingCatalogChoice:
    clarification_id: str
    requested_fact_id: str
    required_choice_input_id: str
    label: str
    options: tuple[ClarificationOption, ...]
    continuation: (
        SourceBindingCatalogInputContinuation
        | FactPlanningCatalogInputContinuation
    )
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AmbiguousQuestionInterpretation:
    clarification_id: str
    requested_fact_id: str
    source_text: str
    candidates: tuple[ConversationInterpretationCandidate, ...] = ()
    accepts_free_text: bool = False
    evidence: tuple[ClarificationEvidence, ...] = ()
    proof_refs: tuple[str, ...] = ()


ClarificationCause = (
    TargetReferenceNotFound
    | TargetReferenceAmbiguous
    | TargetReferenceUnsupported
    | MissingAnswerMetric
    | MissingCatalogRequiredValue
    | MissingCatalogChoice
    | AmbiguousQuestionInterpretation
)


def clarify(cause: ClarificationCause) -> Clarification:
    if isinstance(cause, TargetReferenceNotFound):
        return _target_reference_clarification(
            cause,
            reason=ClarificationReason.UNRESOLVED_REFERENCE,
        )
    if isinstance(cause, TargetReferenceAmbiguous):
        return _target_reference_clarification(
            cause,
            reason=ClarificationReason.MULTIPLE_MATCHING_ENTITIES,
        )
    if isinstance(cause, TargetReferenceUnsupported):
        return _target_reference_clarification(
            cause,
            reason=ClarificationReason.UNSUPPORTED_REFERENCE,
        )
    if isinstance(cause, MissingAnswerMetric):
        return Clarification(
            id=cause.clarification_id,
            requested_fact_id=cause.requested_fact_id,
            need=ClarificationNeed.ANSWER_METRIC,
            reason=ClarificationReason.MISSING_ANSWER_METRIC,
            owner=ClarificationOwner.QUESTION_CONTRACT,
            continuation=QuestionContractContinuation(
                missing_item_id=cause.clarification_id,
                expected_value_kind="answer_definition",
            ),
            subjects=(
                ClarificationSubject(
                    kind=ClarificationSubjectKind.METRIC_PHRASE,
                    id=cause.clarification_id,
                    source_text=cause.source_text,
                    label=cause.metric_needed,
                ),
            ),
            evidence=_cause_evidence(cause.evidence, proof_refs=cause.proof_refs),
        )
    if isinstance(cause, MissingCatalogRequiredValue):
        return Clarification(
            id=cause.clarification_id,
            requested_fact_id=cause.requested_fact_id,
            need=ClarificationNeed.CATALOG_INPUT,
            reason=ClarificationReason.MISSING_REQUIRED_VALUE,
            owner=_catalog_owner(cause.continuation),
            continuation=cause.continuation,
            subjects=(
                ClarificationSubject(
                    kind=ClarificationSubjectKind.CATALOG_INPUT,
                    id=cause.required_input_id,
                    label=cause.label,
                ),
            ),
            evidence=_cause_evidence(cause.evidence, proof_refs=cause.proof_refs),
        )
    if isinstance(cause, MissingCatalogChoice):
        return Clarification(
            id=cause.clarification_id,
            requested_fact_id=cause.requested_fact_id,
            need=ClarificationNeed.CATALOG_INPUT,
            reason=ClarificationReason.CATALOG_REQUIRES_CHOICE,
            owner=_catalog_owner(cause.continuation),
            continuation=cause.continuation,
            subjects=(
                ClarificationSubject(
                    kind=ClarificationSubjectKind.CATALOG_CHOICE,
                    id=cause.required_choice_input_id,
                    label=cause.label,
                    options=cause.options,
                ),
            ),
            evidence=_cause_evidence(cause.evidence, proof_refs=cause.proof_refs),
        )
    if isinstance(cause, AmbiguousQuestionInterpretation):
        return Clarification(
            id=cause.clarification_id,
            requested_fact_id=cause.requested_fact_id,
            need=ClarificationNeed.QUESTION_INTERPRETATION,
            reason=ClarificationReason.AMBIGUOUS_INTERPRETATION,
            owner=ClarificationOwner.CONVERSATION_RESOLUTION,
            continuation=ConversationResolutionContinuation(
                candidates=cause.candidates,
                accepts_free_text=cause.accepts_free_text,
            ),
            subjects=(
                ClarificationSubject(
                    kind=ClarificationSubjectKind.INTERPRETATION,
                    id=cause.clarification_id,
                    source_text=cause.source_text,
                    options=tuple(
                        ClarificationOption(
                            id=candidate.id,
                            label=candidate.contextualized_question,
                        )
                        for candidate in cause.candidates
                    ),
                ),
            ),
            evidence=_cause_evidence(cause.evidence, proof_refs=cause.proof_refs),
        )
    raise TypeError("unsupported clarification cause")


def _target_reference_clarification(
    cause: TargetReferenceNotFound
    | TargetReferenceAmbiguous
    | TargetReferenceUnsupported,
    *,
    reason: ClarificationReason,
) -> Clarification:
    return Clarification(
        id=cause.clarification_id,
        requested_fact_id=cause.requested_fact_id,
        need=ClarificationNeed.TARGET_REFERENCE,
        reason=reason,
        owner=ClarificationOwner.GROUNDING,
        continuation=GroundingContinuation(
            known_input_id=cause.known_input_id,
            accepts_free_text=not bool(getattr(cause, "options", ())),
        ),
        subjects=(
            ClarificationSubject(
                kind=ClarificationSubjectKind.QUESTION_INPUT,
                id=cause.known_input_id,
                label=cause.target_label,
                source_text=cause.source_text,
                options=getattr(cause, "options", ()),
            ),
        ),
        evidence=_dedupe_evidence(
            (
                ClarificationEvidence(
                    kind=ClarificationEvidenceKind.KNOWN_INPUT,
                    id=f"known_input:{cause.known_input_id}",
                ),
                *cause.evidence,
                *_target_reference_candidate_evidence(cause),
                *_proof_ref_evidence(cause.proof_refs),
            )
        ),
    )


def _catalog_owner(
    continuation: (
        SourceBindingCatalogInputContinuation
        | FactPlanningCatalogInputContinuation
    ),
) -> ClarificationOwner:
    if isinstance(continuation, SourceBindingCatalogInputContinuation):
        return ClarificationOwner.SOURCE_BINDING
    return ClarificationOwner.FACT_PLANNING


def _target_reference_candidate_evidence(
    cause: TargetReferenceNotFound
    | TargetReferenceAmbiguous
    | TargetReferenceUnsupported,
) -> tuple[ClarificationEvidence, ...]:
    if not isinstance(cause, TargetReferenceAmbiguous):
        return ()
    return tuple(
        ClarificationEvidence(kind=ClarificationEvidenceKind.CANDIDATE, id=option.id)
        for option in cause.options
    )


def _cause_evidence(
    evidence: tuple[ClarificationEvidence, ...],
    *,
    proof_refs: tuple[str, ...],
) -> tuple[ClarificationEvidence, ...]:
    return _dedupe_evidence((*evidence, *_proof_ref_evidence(proof_refs)))


def _proof_ref_evidence(
    proof_refs: tuple[str, ...],
) -> tuple[ClarificationEvidence, ...]:
    return tuple(
        ClarificationEvidence(kind=ClarificationEvidenceKind.PROOF_REF, id=ref)
        for ref in proof_refs
    )


def _dedupe_evidence(
    evidence: tuple[ClarificationEvidence, ...],
) -> tuple[ClarificationEvidence, ...]:
    output: list[ClarificationEvidence] = []
    seen: set[tuple[object, ...]] = set()
    for item in evidence:
        key = (
            item.kind,
            item.id,
            item.read_id,
            item.endpoint_name,
            item.field_id,
            item.identity_field,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return tuple(output)
