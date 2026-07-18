from __future__ import annotations

from typing import Any

from jsonschema import ValidationError, validate

from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionRanking,
    CatalogSelectionResult,
    EntityTargetResolverSelection,
    RequestedFactCatalogSelection,
)
from fervis.lookup.fact_plan.row_sources import build_row_source_catalog
from fervis.lookup.grounding.resolution.references import (
    reference_binding_sources_by_known_input,
    reference_input_binding_tasks,
)
from fervis.lookup.grounding.model import (
    CompatibleInputBinding,
    ResolverRequestValue,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.read_eligibility.model import (
    ReadEligibilityRequest,
    RetainedReadAssessment,
)
from fervis.lookup.read_eligibility.parser import parse_read_eligibility
from fervis.lookup.read_eligibility.prompt import ReadEligibilityTurnPrompt
from fervis.lookup.read_eligibility.recall import (
    prepare_catalog_selection_for_read_eligibility,
)
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)

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


def run_read_eligibility_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    try:
        result = parse_read_eligibility(
            dict(payload["input"]["payload"]), request=request
        )
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
                if isinstance(item, RetainedReadAssessment)
            ],
            "canonical_inputs": [
                {
                    "known_input_id": item.known_input_id,
                    "canonical_option_id": item.canonical_option_id,
                    "interpretation_question": item.interpretation_question,
                    "canonical_option_assessments": dict(
                        item.canonical_option_assessments
                    ),
                    "because": item.because,
                }
                for item in result.canonical_inputs
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
        "object_schema_paths_not_closed": _object_schema_paths_not_closed(
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
        "first_response_row_evidence_token": card["response_rows"][0]["evidence_token"],
        "field_evidence_tokens_by_field_id": {
            field["field_id"]: field["evidence_token"]
            for row in card["response_rows"]
            for field in row["fields"]
        },
        "known_inputs_by_fact": {
            str(group.get("requested_fact_id") or ""): group.get("known_inputs")
            for group in card_payload.get("requested_fact_read_candidates") or ()
            if isinstance(group, dict)
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
        compatible_resolver_reads_by_known_input={
            str(known_input_id): tuple(read_ids)
            for known_input_id, read_ids in (
                request_payload.get("compatible_resolver_reads_by_known_input") or {}
            ).items()
        },
    )


def _request(
    *,
    question: str,
    question_contract: Any,
    catalog: Any,
    selected_read_ids_by_fact: dict[str, tuple[str, ...]],
    unselected_positive_read_ids_by_fact: dict[str, tuple[str, ...]] | None = None,
    compatible_resolver_reads_by_known_input: (
        dict[str, tuple[str, ...]] | None
    ) = None,
) -> ReadEligibilityRequest:
    facts = question_contract.requested_facts
    unselected_positive = unselected_positive_read_ids_by_fact or {}
    resolver_selections = tuple(
        EntityTargetResolverSelection(
            target_id=known_input_id,
            catalog_search_terms=(),
            selected_read_ids=read_ids,
        )
        for known_input_id, read_ids in (
            compatible_resolver_reads_by_known_input or {}
        ).items()
    )
    binding_tasks = reference_input_binding_tasks(
        question_contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input=reference_binding_sources_by_known_input(
            full_row_sources=build_row_source_catalog(catalog),
            resolver_selections=resolver_selections,
        ),
    )
    lookup_text_by_input_id = {
        known_input.id: known_input.resolved_value_text
        for fact in facts
        for known_input in fact.known_inputs
        if hasattr(known_input, "resolved_value_text")
    }
    compatible_reference_bindings = tuple(
        _compatible_binding(
            catalog=catalog,
            option=option,
            lookup_text=str(lookup_text_by_input_id[option.known_input_id]),
        )
        for task in binding_tasks
        for option in task.options
    )
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
        binding_tasks=binding_tasks,
        compatible_reference_bindings=compatible_reference_bindings,
        resolver_catalog=catalog,
    )


def _compatible_binding(
    *,
    catalog: Any,
    option: Any,
    lookup_text: str,
) -> CompatibleInputBinding:
    read = catalog.read(option.candidate.resolver_read_id)
    lookup_params = tuple(
        parameter
        for parameter in read.params
        if parameter.type in {"string", "uuid", "pk", "path"}
        and parameter.semantics != "response_shape"
    )
    match_paths = tuple(
        field.path
        for field in read.fields
        if field.type in {"string", "uuid", "pk"}
        and (field.row_path_id or "root") == option.candidate.resolver_row_path_id
    )
    return CompatibleInputBinding(
        option_id=option.id,
        lookup_value=lookup_text,
        request_values=tuple(
            ResolverRequestValue(param_ref=parameter.ref, value=lookup_text)
            for parameter in lookup_params[:1]
        ),
        response_match_field_paths=match_paths,
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


def _object_schema_paths_not_closed(schema: object, path: str = "$") -> list[str]:
    if isinstance(schema, dict):
        errors: list[str] = []
        if (
            schema.get("type") == "object"
            and schema.get("additionalProperties") is not False
        ):
            errors.append(path)
        for key, value in schema.items():
            errors.extend(_object_schema_paths_not_closed(value, f"{path}.{key}"))
        return errors
    if isinstance(schema, list):
        errors = []
        for index, value in enumerate(schema):
            errors.extend(_object_schema_paths_not_closed(value, f"{path}[{index}]"))
        return errors
    return []
