"""Parse a response only against its stored owner continuation."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from fervis.lookup.clarification.model import (
    CatalogInputTarget,
    Clarification,
    ClarificationOption,
    ClarificationOwnerResponse,
    ClarificationResponseSource,
    ConversationResolutionContinuation,
    ConversationResolutionResponse,
    ConversationInterpretationCandidate,
    ConversationInterpretationEvidence,
    FactPlanningCatalogInputContinuation,
    FactPlanningCatalogInputResponse,
    GroundingContinuation,
    GroundingIdentityResponse,
    GroundingTextResponse,
    QuestionContractContinuation,
    QuestionContractResponse,
    SourceBindingCatalogInputContinuation,
    SourceBindingCatalogInputResponse,
)
from fervis.lookup.canonical_data import entity_key_from_payload, entity_key_to_payload


def clarification_response_payload(
    response: ClarificationOwnerResponse,
) -> dict[str, object]:
    if isinstance(response, ConversationResolutionResponse):
        return {
            "kind": "conversation_resolution",
            "source": _source_payload(response.source),
            "candidate": (
                None if response.candidate is None else _candidate_payload(response.candidate)
            ),
        }
    if isinstance(response, QuestionContractResponse):
        return {
            "kind": "question_contract",
            "source": _source_payload(response.source),
            "missingItemId": response.missing_item_id,
            "expectedValueKind": response.expected_value_kind,
        }
    if isinstance(response, GroundingIdentityResponse):
        return {
            "kind": "grounding_identity",
            "responseId": response.response_id,
            "clarificationId": response.clarification_id,
            "requestedFactId": response.requested_fact_id,
            "knownInputId": response.known_input_id,
            "option": _option_payload(response.option),
        }
    if isinstance(response, GroundingTextResponse):
        return {
            "kind": "grounding_text",
            "source": _source_payload(response.source),
            "requestedFactId": response.requested_fact_id,
            "knownInputId": response.known_input_id,
        }
    if isinstance(response, SourceBindingCatalogInputResponse):
        return {
            "kind": "source_binding_catalog_input",
            "responseId": response.response_id,
            "clarificationId": response.clarification_id,
            "requestedFactId": response.requested_fact_id,
            "target": vars(response.target),
            "value": response.value,
        }
    if isinstance(response, FactPlanningCatalogInputResponse):
        return {
            "kind": "fact_planning_catalog_input",
            "responseId": response.response_id,
            "clarificationId": response.clarification_id,
            "requestedFactId": response.requested_fact_id,
            "planningRequirementId": response.planning_requirement_id,
            "target": vars(response.target),
            "value": response.value,
        }
    raise TypeError("unsupported clarification owner response")


def clarification_response_from_payload(
    payload: Mapping[str, object],
) -> ClarificationOwnerResponse:
    kind = _required_text(payload, "kind")
    if kind == "conversation_resolution":
        raw_candidate = payload.get("candidate")
        return ConversationResolutionResponse(
            source=_source_from_payload(_mapping(payload, "source")),
            candidate=(
                None
                if raw_candidate is None
                else _candidate_from_payload(_as_mapping(raw_candidate, "candidate"))
            ),
        )
    if kind == "question_contract":
        return QuestionContractResponse(
            source=_source_from_payload(_mapping(payload, "source")),
            missing_item_id=_required_text(payload, "missingItemId"),
            expected_value_kind=_required_text(payload, "expectedValueKind"),
        )
    if kind == "grounding_identity":
        return GroundingIdentityResponse(
            response_id=_required_text(payload, "responseId"),
            clarification_id=_required_text(payload, "clarificationId"),
            requested_fact_id=_required_text(payload, "requestedFactId"),
            known_input_id=_required_text(payload, "knownInputId"),
            option=_option_from_payload(_mapping(payload, "option")),
        )
    if kind == "grounding_text":
        return GroundingTextResponse(
            source=_source_from_payload(_mapping(payload, "source")),
            requested_fact_id=_required_text(payload, "requestedFactId"),
            known_input_id=_required_text(payload, "knownInputId"),
        )
    if kind == "source_binding_catalog_input":
        return SourceBindingCatalogInputResponse(
            response_id=_required_text(payload, "responseId"),
            clarification_id=_required_text(payload, "clarificationId"),
            requested_fact_id=_required_text(payload, "requestedFactId"),
            target=_target_from_payload(_mapping(payload, "target")),
            value=_required_text(payload, "value"),
        )
    if kind == "fact_planning_catalog_input":
        return FactPlanningCatalogInputResponse(
            response_id=_required_text(payload, "responseId"),
            clarification_id=_required_text(payload, "clarificationId"),
            requested_fact_id=_required_text(payload, "requestedFactId"),
            planning_requirement_id=_required_text(payload, "planningRequirementId"),
            target=_target_from_payload(_mapping(payload, "target")),
            value=_required_text(payload, "value"),
        )
    raise ValueError(f"unsupported clarification response kind: {kind}")


def _source_payload(source: ClarificationResponseSource) -> dict[str, str]:
    return {
        "responseId": source.response_id,
        "clarificationId": source.clarification_id,
        "exactUserText": source.exact_user_text,
    }


def _source_from_payload(payload: Mapping[str, object]) -> ClarificationResponseSource:
    return ClarificationResponseSource(
        response_id=_required_text(payload, "responseId"),
        clarification_id=_required_text(payload, "clarificationId"),
        exact_user_text=_required_text(payload, "exactUserText"),
    )


def _candidate_payload(candidate: ConversationInterpretationCandidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "contextualizedQuestion": candidate.contextualized_question,
        "sourceEvidence": [
            {"sourceId": item.source_id, "exactSourceTexts": list(item.exact_source_texts)}
            for item in candidate.source_evidence
        ],
    }


def _candidate_from_payload(payload: Mapping[str, object]) -> ConversationInterpretationCandidate:
    evidence = payload.get("sourceEvidence")
    if not isinstance(evidence, list):
        raise ValueError("candidate sourceEvidence must be a list")
    return ConversationInterpretationCandidate(
        id=_required_text(payload, "id"),
        contextualized_question=_required_text(payload, "contextualizedQuestion"),
        source_evidence=tuple(
            ConversationInterpretationEvidence(
                source_id=_required_text(_as_mapping(item, "sourceEvidence"), "sourceId"),
                exact_source_texts=tuple(
                    _text_list(_as_mapping(item, "sourceEvidence"), "exactSourceTexts")
                ),
            )
            for item in evidence
        ),
    )


def _target_from_payload(payload: Mapping[str, object]) -> CatalogInputTarget:
    choices = payload.get("choices", ())
    if not isinstance(choices, (list, tuple)) or any(not isinstance(item, str) for item in choices):
        raise ValueError("catalog target choices must be text")
    return CatalogInputTarget(
        row_source_id=_required_text(payload, "row_source_id"),
        param_id=_required_text(payload, "param_id"),
        param_ref=_required_text(payload, "param_ref"),
        value_type=_required_text(payload, "value_type"),
        choices=tuple(choices),
    )


def _option_from_payload(payload: Mapping[str, object]) -> ClarificationOption:
    return ClarificationOption(
        id=_required_text(payload, "id"),
        label=_optional_text(payload, "label"),
        value=_optional_text(payload, "value"),
        key=(
            entity_key_from_payload(payload["key"])
            if payload.get("key") is not None
            else None
        ),
        matched_label=_optional_text(payload, "matched_label"),
        matched_field=_optional_text(payload, "matched_field"),
        matched_value=_optional_text(payload, "matched_value"),
        resolver_read_id=_optional_text(payload, "resolver_read_id"),
        resolver_label=_optional_text(payload, "resolver_label"),
    )


def _option_payload(option: ClarificationOption) -> dict[str, object]:
    return {
        "id": option.id,
        "label": option.label,
        "value": option.value,
        "key": entity_key_to_payload(option.key) if option.key is not None else None,
        "matched_label": option.matched_label,
        "matched_field": option.matched_field,
        "matched_value": option.matched_value,
        "resolver_read_id": option.resolver_read_id,
        "resolver_label": option.resolver_label,
    }


def _mapping(payload: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _as_mapping(payload.get(field), field)


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"clarification response {field} must be an object")
    return value


def _required_text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"clarification response {field} is required")
    return value


def _optional_text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"clarification response {field} must be text")
    return value


def _text_list(payload: Mapping[str, object], field: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"clarification response {field} must be text")
    return value


def parse_clarification_response(
    clarification: Clarification,
    *,
    response_id: str,
    response_text: str,
    selected_option_id: str = "",
) -> ClarificationOwnerResponse:
    if not response_id.strip():
        raise ValueError("clarification response requires response_id")
    selected = _selected_option(clarification, selected_option_id)
    continuation = clarification.continuation
    if isinstance(continuation, ConversationResolutionContinuation):
        candidate = None
        if continuation.candidates:
            if selected is None:
                raise ValueError("conversation clarification requires a stored option")
            candidate = next(
                item for item in continuation.candidates if item.id == selected.id
            )
        elif not continuation.accepts_free_text:
            raise ValueError("conversation clarification has no response slot")
        return ConversationResolutionResponse(
            source=_source(clarification.id, response_id, response_text),
            candidate=candidate,
        )
    if isinstance(continuation, QuestionContractContinuation):
        if selected is not None:
            raise ValueError("question-contract clarification does not accept options")
        return QuestionContractResponse(
            source=_source(clarification.id, response_id, response_text),
            missing_item_id=continuation.missing_item_id,
            expected_value_kind=continuation.expected_value_kind,
        )
    if isinstance(continuation, GroundingContinuation):
        if selected is not None:
            return GroundingIdentityResponse(
                response_id=response_id,
                clarification_id=clarification.id,
                requested_fact_id=clarification.requested_fact_id,
                known_input_id=continuation.known_input_id,
                option=selected,
            )
        if not continuation.accepts_free_text:
            raise ValueError("grounding clarification requires a stored option")
        return GroundingTextResponse(
            source=_source(clarification.id, response_id, response_text),
            requested_fact_id=clarification.requested_fact_id,
            known_input_id=continuation.known_input_id,
        )
    if isinstance(continuation, SourceBindingCatalogInputContinuation):
        return SourceBindingCatalogInputResponse(
            response_id=response_id,
            clarification_id=clarification.id,
            requested_fact_id=continuation.requested_fact_id,
            target=continuation.target,
            value=_catalog_value(continuation.target, selected, response_text),
        )
    if isinstance(continuation, FactPlanningCatalogInputContinuation):
        return FactPlanningCatalogInputResponse(
            response_id=response_id,
            clarification_id=clarification.id,
            requested_fact_id=continuation.requested_fact_id,
            planning_requirement_id=continuation.planning_requirement_id,
            target=continuation.target,
            value=_catalog_value(continuation.target, selected, response_text),
        )
    raise TypeError("unsupported clarification continuation")


def _selected_option(
    clarification: Clarification,
    selected_option_id: str,
) -> ClarificationOption | None:
    if not selected_option_id.strip():
        return None
    matches = tuple(
        option
        for subject in clarification.subjects
        for option in subject.options
        if option.id == selected_option_id
    )
    if len(matches) != 1:
        raise ValueError("clarification response selects an unknown option")
    return matches[0]


def _source(
    clarification_id: str,
    response_id: str,
    response_text: str,
) -> ClarificationResponseSource:
    return ClarificationResponseSource(
        response_id=response_id,
        clarification_id=clarification_id,
        exact_user_text=response_text,
    )


def _catalog_value(
    target: CatalogInputTarget,
    selected: ClarificationOption | None,
    response_text: str,
) -> str:
    if target.choices:
        if selected is None or selected.id not in target.choices:
            raise ValueError("catalog clarification requires a stored choice")
        value = selected.value or selected.id
    elif selected is not None:
        raise ValueError("catalog free-text target does not accept an option")
    elif not response_text.strip():
        raise ValueError("catalog clarification requires a value")
    else:
        value = response_text
    value_type = target.value_type.casefold()
    if value_type in {"number", "integer"}:
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(
                f"catalog clarification value must be {value_type}"
            ) from exc
        if not number.is_finite() or (value_type == "integer" and number != number.to_integral()):
            raise ValueError(f"catalog clarification value must be {value_type}")
    if value_type == "boolean" and value.strip().casefold() not in {"true", "false"}:
        raise ValueError("catalog clarification value must be boolean")
    return value
