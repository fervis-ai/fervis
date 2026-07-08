"""Canonical lookup clarification rendering."""

from __future__ import annotations

from fervis.lookup.clarification.model import (
    Clarification,
    ClarificationNeed,
    ClarificationReason,
)


def render_clarification_question(clarification: Clarification) -> str:
    if clarification.need == ClarificationNeed.TARGET_REFERENCE:
        return _target_reference_question(clarification)
    if clarification.reason == ClarificationReason.MISSING_ANSWER_METRIC:
        return "Which metric should I use?"
    if clarification.reason == ClarificationReason.MISSING_REQUIRED_VALUE:
        subject = clarification.subjects[0]
        label = subject.label or subject.id
        return f"What {label} should I use?"
    if clarification.reason == ClarificationReason.CATALOG_REQUIRES_CHOICE:
        subject = clarification.subjects[0]
        label = subject.label or subject.id
        return f"Which {label} should I use?"
    if clarification.need == ClarificationNeed.QUESTION_INTERPRETATION:
        return "Which interpretation should I use?"
    return "Can you clarify the requested value?"


def _target_reference_question(clarification: Clarification) -> str:
    subject = clarification.subjects[0]
    label = subject.label or "entity"
    source_text = subject.source_text
    if clarification.reason == ClarificationReason.MULTIPLE_MATCHING_ENTITIES:
        return f"Which matching {label} should I use?"
    if clarification.reason == ClarificationReason.UNRESOLVED_REFERENCE:
        if source_text:
            return f'I could not find {label} "{source_text}". Which {label} should I use?'
        return f"Which {label} should I use?"
    if clarification.reason == ClarificationReason.UNSUPPORTED_REFERENCE:
        if source_text:
            return (
                f'I found "{source_text}", but it is not supported for this question. '
                f"Which supported {label} should I use?"
            )
        return f"Which supported {label} should I use?"
    return f"Which {label} should I use?"
