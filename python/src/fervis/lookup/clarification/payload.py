"""Canonical public clarification payloads."""

from __future__ import annotations

from collections.abc import Mapping

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
from fervis.lookup.clarification.render import render_clarification_question


def clarification_payload(clarification: Clarification) -> dict[str, object]:
    return {
        "id": clarification.id,
        "need": clarification.need.value,
        "reason": clarification.reason.value,
        "requestedFactId": clarification.requested_fact_id,
        "question": render_clarification_question(clarification),
        "subjects": [
            _subject_payload(subject) for subject in clarification.subjects
        ],
        "evidence": [
            _evidence_payload(evidence) for evidence in clarification.evidence
        ],
    }


def clarifications_payload(
    clarifications: tuple[Clarification, ...],
) -> dict[str, object]:
    return {
        "clarifications": [
            clarification_payload(item) for item in clarifications
        ],
    }


def clarification_from_payload(payload: Mapping[str, object]) -> Clarification:
    return Clarification(
        id=_required_text(payload, "id"),
        requested_fact_id=_required_text(payload, "requestedFactId"),
        need=ClarificationNeed(_required_text(payload, "need")),
        reason=ClarificationReason(_required_text(payload, "reason")),
        subjects=tuple(
            _subject_from_payload(subject)
            for subject in _required_mapping_items(payload, "subjects")
        ),
        evidence=tuple(
            _evidence_from_payload(evidence)
            for evidence in _mapping_items(payload, "evidence")
        ),
    )


def _subject_payload(subject: ClarificationSubject) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": subject.kind.value,
        "id": subject.id,
        "label": subject.label,
        "sourceText": subject.source_text,
        "options": [_option_payload(option) for option in subject.options],
    }
    return payload


def _subject_from_payload(payload: Mapping[str, object]) -> ClarificationSubject:
    return ClarificationSubject(
        kind=ClarificationSubjectKind(_required_text(payload, "kind")),
        id=_required_text(payload, "id"),
        label=_text(payload.get("label")),
        source_text=_text(payload.get("sourceText")),
        options=tuple(
            _option_from_payload(option)
            for option in _mapping_items(payload, "options")
        ),
    )


def _option_payload(option: ClarificationOption) -> dict[str, object]:
    payload = {
        "id": option.id,
        "label": option.label,
    }
    if option.value:
        payload["value"] = option.value
    if option.entity_kind:
        payload["entityKind"] = option.entity_kind
    if option.matched_label:
        payload["matchedLabel"] = option.matched_label
    if option.matched_field:
        payload["matchedField"] = option.matched_field
    if option.matched_value:
        payload["matchedValue"] = option.matched_value
    if option.resolver_read_id:
        payload["resolverReadId"] = option.resolver_read_id
    if option.resolver_label:
        payload["resolverLabel"] = option.resolver_label
    return payload


def _option_from_payload(payload: Mapping[str, object]) -> ClarificationOption:
    return ClarificationOption(
        id=_required_text(payload, "id"),
        label=_text(payload.get("label")),
        value=_text(payload.get("value")),
        entity_kind=_text(payload.get("entityKind")),
        matched_label=_text(payload.get("matchedLabel")),
        matched_field=_text(payload.get("matchedField")),
        matched_value=_text(payload.get("matchedValue")),
        resolver_read_id=_text(payload.get("resolverReadId")),
        resolver_label=_text(payload.get("resolverLabel")),
    )


def _evidence_payload(evidence: ClarificationEvidence) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": evidence.kind.value,
        "id": evidence.id,
    }
    if evidence.read_id:
        payload["readId"] = evidence.read_id
    if evidence.endpoint_name:
        payload["endpointName"] = evidence.endpoint_name
    if evidence.field_id:
        payload["fieldId"] = evidence.field_id
    if evidence.identity_field:
        payload["identityField"] = evidence.identity_field
    return payload


def _evidence_from_payload(payload: Mapping[str, object]) -> ClarificationEvidence:
    return ClarificationEvidence(
        kind=ClarificationEvidenceKind(_required_text(payload, "kind")),
        id=_required_text(payload, "id"),
        read_id=_text(payload.get("readId")),
        endpoint_name=_text(payload.get("endpointName")),
        field_id=_text(payload.get("fieldId")),
        identity_field=_text(payload.get("identityField")),
    )


def _required_mapping_items(
    payload: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...]:
    items = _mapping_items(payload, key)
    if not items:
        raise ValueError(f"clarification payload requires {key}")
    return items


def _mapping_items(
    payload: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"clarification payload {key} must be an array")
    return tuple(item for item in value if isinstance(item, Mapping))


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = _text(payload.get(key))
    if not value:
        raise ValueError(f"clarification payload requires {key}")
    return value


def _text(value: object) -> str:
    return str(value or "")
