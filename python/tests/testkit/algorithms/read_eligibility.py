from __future__ import annotations

from typing import Any

from jsonschema import ValidationError, validate

from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.read_eligibility.model import ReadEligibilityRequest
from fervis.lookup.read_eligibility.parser import parse_read_eligibility
from fervis.lookup.read_eligibility.prompt import ReadEligibilityTurnPrompt
from fervis.lookup.read_eligibility.recall import (
    prepare_catalog_selection_for_read_eligibility,
)
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)
from fervis.lookup.fact_plan.values import FactValue

from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)
from tests.testkit.catalog import catalog_from_payload
from tests.testkit.fixtures import load_conformance_fixture
from tests.testkit.question_contract import (
    question_contract_from_payload,
)
from tests.testkit.values import fact_value_from_payload


def run_read_eligibility_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    try:
        result = parse_read_eligibility(dict(payload["input"]["payload"]), request=request)
    except ValueError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return subset_mismatches(
        actual={
            "retained_read_ids_by_requested_fact": {
                fact_id: list(read_ids)
                for fact_id, read_ids in result.retained_read_ids_by_requested_fact().items()
            },
            "read_assessments": [
                {
                    "source_candidate_id": item.source_candidate_id,
                    "source_candidate_signature": item.source_candidate_signature,
                    "requested_fact_id": item.requested_fact_id,
                    "read_id": item.read_id,
                    "retention_decision": item.retention_decision,
                    "relevant_row_path_ids": list(item.relevant_row_path_ids),
                    "relevant_field_refs": list(item.relevant_field_refs),
                }
                for item in result.read_assessments
            ],
        },
        expected_subset=payload["expect"]["result_contains"],
    )


def run_read_eligibility_schema_validate_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    schema = ReadEligibilityTurnPrompt(request).response_contract().provider_schema
    try:
        validate(instance=dict(payload["input"]["payload"]), schema=schema)
    except ValidationError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected validation error: {exc.message}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    return []


def run_read_eligibility_prompt_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    invocation = ReadEligibilityTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context={},
            memory_payload={},
        )
    )
    actual = {
        "contains": {
            text: text in invocation.prompt_text
            for text in payload["input"].get("contains") or ()
        },
        "excludes": {
            text: text not in invocation.prompt_text
            for text in payload["input"].get("excludes") or ()
        },
        "array_schema_paths_missing_items": _array_schema_paths_missing_items(
            invocation.provider_schema
        ),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_read_eligibility_cards_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    card_payload = read_eligibility_candidate_surface(request).card_payload
    card = card_payload["requested_fact_read_candidates"][0]["read_candidates"][0]
    actual = {
        "first_response_row_evidence_token": card["response_rows"][0][
            "evidence_token"
        ],
        "field_evidence_tokens_by_field_id": {
            field["field_id"]: field["evidence_token"]
            for row in card["response_rows"]
            for field in row["fields"]
        },
        "applicable_known_inputs_by_read": {
            str(candidate.get("read_id") or ""): candidate.get(
                "applicable_known_inputs"
            )
            for group in card_payload.get("requested_fact_read_candidates") or ()
            if isinstance(group, dict)
            for candidate in group.get("read_candidates") or ()
            if isinstance(candidate, dict)
            and candidate.get("applicable_known_inputs")
        },
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_read_eligibility_prepare_recall_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    prepared = prepare_catalog_selection_for_read_eligibility(
        catalog_selection=request.catalog_selection,
        full_catalog=request.catalog_selection.relation_catalog,
        max_reads_per_fact=int(payload["input"].get("max_reads_per_fact") or 10),
    )
    surface = read_eligibility_candidate_surface(
        ReadEligibilityRequest(
            question=request.question,
            question_contract=request.question_contract,
            requested_facts=request.requested_facts,
            catalog_selection=prepared,
            conversation_context=request.conversation_context,
            available_values=request.available_values,
            conversation_resolution_overlay=request.conversation_resolution_overlay,
        )
    )
    card_payload = surface.card_payload
    actual = {
        "selected_read_ids_by_fact": {
            item.requested_fact_id: list(item.selected_read_ids)
            for item in prepared.requested_fact_selections
        },
        "card_read_ids_by_fact": {
            str(group.get("requested_fact_id") or ""): [
                str(card.get("read_id") or "")
                for card in group.get("read_candidates") or ()
                if isinstance(card, dict)
            ]
            for group in card_payload.get("requested_fact_read_candidates") or ()
            if isinstance(group, dict)
        },
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _request_from_input(input_payload: dict[str, Any]) -> ReadEligibilityRequest:
    fixture = load_conformance_fixture(
        "read_eligibility",
        str(input_payload["request_fixture"]),
    )
    request_payload = {**fixture, **dict(input_payload.get("request") or {})}
    question_contract = question_contract_from_payload(request_payload)
    return _request(
        question=str(request_payload["question"]),
        question_contract=question_contract,
        catalog=catalog_from_payload(request_payload["catalog"]),
        selected_read_ids_by_fact={
            str(fact_id): tuple(read_ids)
            for fact_id, read_ids in (
                request_payload.get("selected_read_ids_by_fact") or {}
            ).items()
        },
        unselected_positive_read_ids_by_fact={
            str(fact_id): tuple(read_ids)
            for fact_id, read_ids in (
                request_payload.get("unselected_positive_read_ids_by_fact") or {}
            ).items()
        },
        available_values=tuple(
            fact_value_from_payload(item)
            for item in request_payload.get("available_values") or ()
        ),
    )


def _request(
    *,
    question: str,
    question_contract: Any,
    catalog: Any,
    selected_read_ids_by_fact: dict[str, tuple[str, ...]],
    unselected_positive_read_ids_by_fact: dict[str, tuple[str, ...]] | None = None,
    available_values: tuple[FactValue, ...] = (),
) -> ReadEligibilityRequest:
    facts = question_contract.requested_facts
    unselected_positive = unselected_positive_read_ids_by_fact or {}
    return ReadEligibilityRequest(
        question=question,
        question_contract=question_contract,
        requested_facts=facts,
        catalog_selection=CatalogSelectionResult(
            relation_catalog=catalog,
            requested_fact_selections=tuple(
                RequestedFactCatalogSelection(
                    requested_fact_id=fact.id,
                    query_terms=(),
                    rankings=tuple(
                        CatalogSelectionRanking(read_id=read_id, score=10)
                        for read_id in selected_read_ids_by_fact[fact.id]
                    ),
                    selected_read_ids=selected_read_ids_by_fact[fact.id],
                    unselected_positive_read_ids=unselected_positive.get(fact.id, ()),
                )
                for fact in facts
            ),
            selected_read_ids=tuple(
                read_id
                for fact in facts
                for read_id in selected_read_ids_by_fact[fact.id]
            ),
        ),
        conversation_context={},
        available_values=available_values,
    )


def _array_schema_paths_missing_items(schema: object, path: str = "$") -> list[str]:
    if isinstance(schema, dict):
        errors: list[str] = []
        if schema.get("type") == "array" and "items" not in schema:
            errors.append(path)
        for key, value in schema.items():
            errors.extend(_array_schema_paths_missing_items(value, f"{path}.{key}"))
        return errors
    if isinstance(schema, list):
        errors = []
        for index, value in enumerate(schema):
            errors.extend(_array_schema_paths_missing_items(value, f"{path}[{index}]"))
        return errors
    return []
