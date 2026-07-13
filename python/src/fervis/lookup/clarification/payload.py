"""Canonical public clarification payloads."""

from __future__ import annotations

from collections.abc import Mapping
from fervis.lookup.canonical_data import RuntimeValue

from fervis.lookup.clarification.model import (
    CatalogInputTarget,
    Clarification,
    ClarificationContinuationSpec,
    ClarificationEvidence,
    ClarificationEvidenceKind,
    ClarificationNeed,
    ClarificationOption,
    ClarificationOwner,
    ClarificationReason,
    ClarificationSubject,
    ClarificationSubjectKind,
    ConversationInterpretationCandidate,
    ConversationInterpretationEvidence,
    ConversationResolutionContinuation,
    FactPlanningCatalogInputContinuation,
    GroundingContinuation,
    QuestionContractContinuation,
    SourceBindingCatalogInputContinuation,
)
from fervis.lookup.canonical_data import (
    EntityKeyComponentValue,
    EntityKeyValue,
    runtime_value_from_payload,
    runtime_value_to_payload,
)
from fervis.lookup.clarification.render import render_clarification_question


def clarification_payload(clarification: Clarification) -> dict[str, RuntimeValue]:
    return {
        "id": clarification.id,
        "need": clarification.need.value,
        "reason": clarification.reason.value,
        "owner": clarification.owner.value,
        "continuation": _continuation_payload(clarification.continuation),
        "requestedFactId": clarification.requested_fact_id,
        "question": render_clarification_question(clarification),
        "subjects": [_subject_payload(subject) for subject in clarification.subjects],
        "evidence": [
            _evidence_payload(evidence) for evidence in clarification.evidence
        ],
    }


def clarifications_payload(
    clarifications: tuple[Clarification, ...],
) -> dict[str, RuntimeValue]:
    return {
        "clarifications": [clarification_payload(item) for item in clarifications],
    }


def clarification_from_payload(payload: Mapping[str, object]) -> Clarification:
    return Clarification(
        id=_required_text(payload, "id"),
        requested_fact_id=_required_text(payload, "requestedFactId"),
        need=ClarificationNeed(_required_text(payload, "need")),
        reason=ClarificationReason(_required_text(payload, "reason")),
        owner=ClarificationOwner(_required_text(payload, "owner")),
        continuation=_continuation_from_payload(
            _required_mapping(payload, "continuation")
        ),
        subjects=tuple(
            _subject_from_payload(subject)
            for subject in _required_mapping_items(payload, "subjects")
        ),
        evidence=tuple(
            _evidence_from_payload(evidence)
            for evidence in _mapping_items(payload, "evidence")
        ),
    )


def _continuation_payload(
    continuation: ClarificationContinuationSpec,
) -> dict[str, RuntimeValue]:
    if isinstance(continuation, ConversationResolutionContinuation):
        return {
            "kind": "conversation_resolution",
            "candidates": [
                {
                    "id": candidate.id,
                    "contextualizedQuestion": candidate.contextualized_question,
                    "sourceEvidence": [
                        {
                            "sourceId": item.source_id,
                            "exactSourceTexts": list(item.exact_source_texts),
                        }
                        for item in candidate.source_evidence
                    ],
                }
                for candidate in continuation.candidates
            ],
            "acceptsFreeText": continuation.accepts_free_text,
        }
    if isinstance(continuation, QuestionContractContinuation):
        return {
            "kind": "question_contract",
            "missingItemId": continuation.missing_item_id,
            "expectedValueKind": continuation.expected_value_kind,
        }
    if isinstance(continuation, GroundingContinuation):
        return {
            "kind": "grounding",
            "knownInputId": continuation.known_input_id,
            "acceptsFreeText": continuation.accepts_free_text,
        }
    if isinstance(continuation, SourceBindingCatalogInputContinuation):
        return {
            "kind": "source_binding_catalog_input",
            "requestedFactId": continuation.requested_fact_id,
            "target": _catalog_target_payload(continuation.target),
        }
    if isinstance(continuation, FactPlanningCatalogInputContinuation):
        return {
            "kind": "fact_planning_catalog_input",
            "requestedFactId": continuation.requested_fact_id,
            "planningRequirementId": continuation.planning_requirement_id,
            "target": _catalog_target_payload(continuation.target),
        }
    raise TypeError("unsupported clarification continuation")


def _continuation_from_payload(
    payload: Mapping[str, object],
) -> ClarificationContinuationSpec:
    kind = _required_text(payload, "kind")
    if kind == "conversation_resolution":
        return ConversationResolutionContinuation(
            candidates=tuple(
                ConversationInterpretationCandidate(
                    id=_required_text(item, "id"),
                    contextualized_question=_required_text(
                        item, "contextualizedQuestion"
                    ),
                    source_evidence=tuple(
                        ConversationInterpretationEvidence(
                            source_id=_required_text(evidence, "sourceId"),
                            exact_source_texts=tuple(
                                _texts(evidence.get("exactSourceTexts"))
                            ),
                        )
                        for evidence in _required_mapping_items(
                            item, "sourceEvidence"
                        )
                    ),
                )
                for item in _mapping_items(payload, "candidates")
            ),
            accepts_free_text=_boolean(payload.get("acceptsFreeText")),
        )
    if kind == "question_contract":
        return QuestionContractContinuation(
            missing_item_id=_required_text(payload, "missingItemId"),
            expected_value_kind=_required_text(payload, "expectedValueKind"),
        )
    if kind == "grounding":
        return GroundingContinuation(
            known_input_id=_required_text(payload, "knownInputId"),
            accepts_free_text=_boolean(payload.get("acceptsFreeText")),
        )
    target = _catalog_target_from_payload(_required_mapping(payload, "target"))
    if kind == "source_binding_catalog_input":
        return SourceBindingCatalogInputContinuation(
            requested_fact_id=_required_text(payload, "requestedFactId"),
            target=target,
        )
    if kind == "fact_planning_catalog_input":
        return FactPlanningCatalogInputContinuation(
            requested_fact_id=_required_text(payload, "requestedFactId"),
            planning_requirement_id=_required_text(
                payload, "planningRequirementId"
            ),
            target=target,
        )
    raise ValueError(f"unsupported clarification continuation: {kind}")


def _catalog_target_payload(target: CatalogInputTarget) -> dict[str, RuntimeValue]:
    return {
        "rowSourceId": target.row_source_id,
        "paramId": target.param_id,
        "paramRef": target.param_ref,
        "valueType": target.value_type,
        "choices": list(target.choices),
    }


def _catalog_target_from_payload(payload: Mapping[str, object]) -> CatalogInputTarget:
    return CatalogInputTarget(
        row_source_id=_required_text(payload, "rowSourceId"),
        param_id=_required_text(payload, "paramId"),
        param_ref=_required_text(payload, "paramRef"),
        value_type=_required_text(payload, "valueType"),
        choices=tuple(_texts(payload.get("choices"))),
    )


def _subject_payload(subject: ClarificationSubject) -> dict[str, RuntimeValue]:
    payload: dict[str, RuntimeValue] = {
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


def _option_payload(option: ClarificationOption) -> dict[str, RuntimeValue]:
    payload: dict[str, RuntimeValue] = {
        "id": option.id,
        "label": option.label,
    }
    if option.value:
        payload["value"] = option.value
    if option.key is not None:
        payload["entityKind"] = option.key.entity_kind
        payload["keyId"] = option.key.key_id
        payload["keyComponents"] = [
            {
                "componentId": component.component_id,
                "value": runtime_value_to_payload(component.value),
            }
            for component in option.key.components
        ]
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
        key=_option_key_from_payload(payload),
        matched_label=_text(payload.get("matchedLabel")),
        matched_field=_text(payload.get("matchedField")),
        matched_value=_text(payload.get("matchedValue")),
        resolver_read_id=_text(payload.get("resolverReadId")),
        resolver_label=_text(payload.get("resolverLabel")),
    )


def _option_key_from_payload(
    payload: Mapping[str, object],
) -> EntityKeyValue | None:
    identity_fields = (
        payload.get("entityKind"),
        payload.get("keyId"),
        payload.get("keyComponents"),
    )
    if identity_fields == (None, None, None):
        return None
    entity_kind = _required_text(payload, "entityKind")
    key_id = _required_text(payload, "keyId")
    components = payload.get("keyComponents")
    if not isinstance(components, list) or not components:
        raise ValueError("clarification option keyComponents must be a non-empty array")
    parsed_components: list[EntityKeyComponentValue] = []
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("clarification option key component must be an object")
        parsed_components.append(
            EntityKeyComponentValue(
                component_id=_required_text(component, "componentId"),
                value=runtime_value_from_payload(component.get("value")),
            )
        )
    return EntityKeyValue(
        entity_kind=entity_kind,
        key_id=key_id,
        components=tuple(parsed_components),
    )


def _evidence_payload(evidence: ClarificationEvidence) -> dict[str, RuntimeValue]:
    payload: dict[str, RuntimeValue] = {
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


def _required_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"clarification payload requires {key} object")
    return value


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


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("clarification payload choices must be an array")
    return tuple(_text(item) for item in value)


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("clarification payload boolean must be bool")
    return value
