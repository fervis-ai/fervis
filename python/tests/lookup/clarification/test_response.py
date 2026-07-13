from __future__ import annotations

import pytest

from fervis.lookup.clarification.model import (
    ConversationResolutionResponse,
    FactPlanningCatalogInputResponse,
    GroundingIdentityResponse,
    GroundingTextResponse,
    QuestionContractResponse,
    SourceBindingCatalogInputResponse,
)
from fervis.lookup.clarification.payload import clarification_from_payload
from fervis.lookup.clarification.response import (
    clarification_response_from_payload,
    clarification_response_payload,
    parse_clarification_response,
)


def _subject(kind: str, id: str, options: tuple[str, ...] = (), *, canonical=False):
    return {
        "kind": kind,
        "id": id,
        "label": id,
        "sourceText": "",
        "options": [
            {
                "id": option,
                "label": "Customer One" if canonical else option,
                **(
                    {
                        "entityKind": "customer",
                        "keyId": "customer_id",
                        "matchedField": "customer_id",
                        "matchedValue": "1",
                    }
                    if canonical
                    else {}
                ),
            }
            for option in options
        ],
    }


def _target(
    param_id: str,
    *,
    value_type: str = "string",
    choices: tuple[str, ...] = (),
):
    return {
        "rowSourceId": "orders",
        "paramId": param_id,
        "paramRef": f"query.{param_id}",
        "valueType": value_type,
        "choices": list(choices),
    }


@pytest.mark.parametrize(
    ("owner", "continuation", "subject", "response_text", "selected", "response_type"),
    (
        (
            "conversation_resolution",
            {
                "kind": "conversation_resolution",
                "candidates": [
                    {
                        "id": "candidate_1",
                        "contextualizedQuestion": "How many orders?",
                        "sourceEvidence": [
                            {"sourceId": "memory_1", "exactSourceTexts": ["orders"]}
                        ],
                    }
                ],
                "acceptsFreeText": False,
            },
            _subject("interpretation", "clarification_1", ("candidate_1",)),
            "The orders interpretation.",
            "candidate_1",
            ConversationResolutionResponse,
        ),
        (
            "question_contract",
            {
                "kind": "question_contract",
                "missingItemId": "clarification_1",
                "expectedValueKind": "answer_definition",
            },
            _subject("metric_phrase", "clarification_1"),
            "Use net revenue.",
            "",
            QuestionContractResponse,
        ),
        (
            "grounding",
            {
                "kind": "grounding",
                "knownInputId": "customer",
                "acceptsFreeText": False,
            },
            _subject("question_input", "customer", ("customer_1",), canonical=True),
            "Customer One",
            "customer_1",
            GroundingIdentityResponse,
        ),
        (
            "grounding",
            {
                "kind": "grounding",
                "knownInputId": "customer",
                "acceptsFreeText": True,
            },
            _subject("question_input", "customer"),
            "Customer One",
            "",
            GroundingTextResponse,
        ),
        (
            "source_binding",
            {
                "kind": "source_binding_catalog_input",
                "requestedFactId": "fact_1",
                "target": _target("channel", choices=("online",)),
            },
            _subject("catalog_choice", "orders.channel", ("online",)),
            "Online",
            "online",
            SourceBindingCatalogInputResponse,
        ),
        (
            "fact_planning",
            {
                "kind": "fact_planning_catalog_input",
                "requestedFactId": "fact_1",
                "planningRequirementId": "requirement_1",
                "target": _target("limit", value_type="integer"),
            },
            _subject("catalog_input", "orders.limit"),
            "5",
            "",
            FactPlanningCatalogInputResponse,
        ),
    ),
)
def test_response_dispatches_once_to_closed_owner_variant(
    owner,
    continuation,
    subject,
    response_text,
    selected,
    response_type,
) -> None:
    response = parse_clarification_response(
        _clarification(owner, continuation, subject),
        response_id="response_1",
        response_text=response_text,
        selected_option_id=selected,
    )

    assert isinstance(response, response_type)
    assert (
        clarification_response_from_payload(clarification_response_payload(response))
        == response
    )


def test_finite_option_cannot_be_selected_by_matching_raw_text() -> None:
    clarification = _clarification(
        "grounding",
        {
            "kind": "grounding",
            "knownInputId": "customer",
            "acceptsFreeText": False,
        },
        _subject("question_input", "customer", ("customer_1",), canonical=True),
    )

    with pytest.raises(ValueError, match="stored option"):
        parse_clarification_response(
            clarification,
            response_id="response_1",
            response_text="Customer One",
        )


def test_catalog_response_rejects_value_outside_typed_target() -> None:
    clarification = _clarification(
        "fact_planning",
        {
            "kind": "fact_planning_catalog_input",
            "requestedFactId": "fact_1",
            "planningRequirementId": "requirement_1",
            "target": _target("limit", value_type="integer"),
        },
        _subject("catalog_input", "orders.limit"),
    )

    with pytest.raises(ValueError, match="integer"):
        parse_clarification_response(
            clarification,
            response_id="response_1",
            response_text="five",
        )


def _clarification(owner, continuation, subject):
    return clarification_from_payload(
        {
            "id": "clarification_1",
            "need": (
                "catalog_input"
                if owner in {"source_binding", "fact_planning"}
                else "question_interpretation"
                if owner == "conversation_resolution"
                else "answer_metric"
                if owner == "question_contract"
                else "target_reference"
            ),
            "reason": (
                "catalog_requires_choice"
                if continuation.get("target", {}).get("choices")
                else "missing_required_value"
                if owner in {"source_binding", "fact_planning"}
                else "ambiguous_interpretation"
                if owner == "conversation_resolution"
                else "missing_answer_metric"
                if owner == "question_contract"
                else "unresolved_reference"
            ),
            "owner": owner,
            "continuation": continuation,
            "requestedFactId": "fact_1",
            "question": "Clarify",
            "subjects": [subject],
            "evidence": [],
        }
    )
