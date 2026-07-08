"""First-class lookup clarification capability."""

from fervis.lookup.clarification.causes import (
    AmbiguousQuestionInterpretation,
    ClarificationCause,
    MissingAnswerMetric,
    MissingCatalogChoice,
    MissingCatalogRequiredValue,
    TargetReferenceAmbiguous,
    TargetReferenceNotFound,
    TargetReferenceUnsupported,
    clarify,
)
from fervis.lookup.clarification.model import (
    Clarification,
    ClarificationEvidence,
    ClarificationEvidenceKind,
    ClarificationNeed,
    ClarificationOption,
    ClarificationReason,
    ClarificationSubject,
    ClarificationSubjectKind,
)
from fervis.lookup.clarification.payload import (
    clarification_from_payload,
    clarification_payload,
    clarifications_payload,
)
from fervis.lookup.clarification.render import render_clarification_question

__all__ = (
    "AmbiguousQuestionInterpretation",
    "Clarification",
    "ClarificationCause",
    "ClarificationEvidence",
    "ClarificationEvidenceKind",
    "ClarificationNeed",
    "ClarificationOption",
    "ClarificationReason",
    "ClarificationSubject",
    "ClarificationSubjectKind",
    "MissingAnswerMetric",
    "MissingCatalogChoice",
    "MissingCatalogRequiredValue",
    "TargetReferenceAmbiguous",
    "TargetReferenceNotFound",
    "TargetReferenceUnsupported",
    "clarification_from_payload",
    "clarification_payload",
    "clarifications_payload",
    "clarify",
    "render_clarification_question",
)
