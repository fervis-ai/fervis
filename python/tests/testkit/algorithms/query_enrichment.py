from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import EndpointRead, RelationCatalog
from fervis.lookup.conversation_resolution import (
    ConversationDependencyOverlay,
    ConversationResolutionOverlay,
    ConversationValueFrameOverlay,
    conversation_resolution_query_enrichment_prompt_payload,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.query_enrichment import (
    QueryEnrichmentRequest,
    QueryEnrichmentTurnPrompt,
    build_query_enrichment_schema,
    parse_query_enrichment,
)

from tests.testkit.assertions import subset_mismatches
from tests.testkit.fixtures import load_conformance_fixture
from tests.testkit.question_contract import requested_fact_from_payload


def run_query_enrichment_parse_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    try:
        result = parse_query_enrichment(dict(payload["input"]["payload"]), request=request)
    except ValueError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    actual = {
        "requested_fact_resource_name_matches": [
            {
                "requested_fact_id": item.requested_fact_id,
                "answer_output_resource_lineage": [
                    {
                        "answer_output_id": row.answer_output_id,
                        "support_role": row.support_role,
                        "source_text": row.source_text,
                        "matching_resource_names": list(row.matching_resource_names),
                    }
                    for row in item.answer_output_resource_lineage
                ],
            }
            for item in result.requested_fact_resource_name_matches
        ],
        "entity_target_catalog_search_terms": [
            {
                "target_id": item.target_id,
                "catalog_search_terms": list(item.catalog_search_terms),
            }
            for item in result.entity_target_catalog_search_terms
        ],
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_query_enrichment_schema_case(payload: dict[str, Any]) -> list[str]:
    schema = build_query_enrichment_schema(
        resource_names=tuple(payload["input"].get("resource_names") or ())
    )
    fact_item_schema = schema["properties"]["requested_fact_resource_name_matches"][
        "items"
    ]
    match_item_schema = fact_item_schema["properties"][
        "answer_output_resource_lineage"
    ]["items"]
    matching_names_schema = match_item_schema["properties"]["matching_resource_names"]
    entity_item_schema = schema["properties"]["entity_target_catalog_search_terms"][
        "items"
    ]
    entity_terms_schema = entity_item_schema["properties"]["catalog_search_terms"]
    term_item_schema = entity_terms_schema["items"]
    source_text_matches_schema = fact_item_schema["properties"][
        "answer_output_resource_lineage"
    ]
    actual = {
        "lineage_property_order": list(match_item_schema["properties"]),
        "matching_resource_names_has_max_items": "maxItems" in matching_names_schema,
        "matching_resource_names_items": matching_names_schema["items"],
        "entity_terms_max_items": entity_terms_schema.get("maxItems"),
        "entity_term_schema": {
            "property_order": list(term_item_schema["properties"]),
            "term": term_item_schema["properties"]["term"],
            "required": term_item_schema["required"],
        },
        "source_text_matches_max_items": source_text_matches_schema.get("maxItems"),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_query_enrichment_prompt_case(payload: dict[str, Any]) -> list[str]:
    request = _request_from_input(payload["input"])
    prompt = QueryEnrichmentTurnPrompt(request).to_model_payload(
        build_turn_prompt_context(
            current_question=request.question,
            conversation_context=request.conversation_context,
            conversation_resolution_overlay=conversation_resolution_query_enrichment_prompt_payload(
                request.conversation_resolution_overlay
            ),
        )
    ).prompt_text
    prompt_object = QueryEnrichmentTurnPrompt(request)
    actual = {
        "prompt_text": prompt,
        "requested_facts_payload": prompt_object.requested_facts_payload(),
        "entity_targets_payload": prompt_object.entity_targets_payload(),
        "contains": {
            text: text in prompt for text in payload["input"].get("contains") or ()
        },
        "excludes": {
            text: text not in prompt for text in payload["input"].get("excludes") or ()
        },
        "ordered_before": {
            f"{left} < {right}": prompt.index(left) < prompt.index(right)
            for left, right in payload["input"].get("ordered_before") or ()
        },
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _request_from_input(input_payload: dict[str, Any]) -> QueryEnrichmentRequest:
    request_payload = dict(input_payload.get("request") or {})
    if "request_fixture" in input_payload:
        fixture = load_conformance_fixture(
            "query_enrichment",
            str(input_payload["request_fixture"]),
        )
        request_payload = {**fixture, **request_payload}
    return _request(request_payload)


def _request(payload: dict[str, Any]) -> QueryEnrichmentRequest:
    return QueryEnrichmentRequest(
        question=str(payload["question"]),
        conversation_context=dict(payload.get("conversation_context") or {}),
        requested_facts=tuple(
            requested_fact_from_payload(item)
            for item in payload.get("requested_facts") or ()
        ),
        relation_catalog=RelationCatalog(
            reads=tuple(
                EndpointRead(
                    id=str(item["id"]),
                    endpoint_name=str(item.get("endpoint_name") or item["id"]),
                    resource_names=tuple(
                        str(name) for name in item.get("resource_names") or ()
                    ),
                )
                for item in payload["reads"]
            )
        ),
        conversation_resolution_overlay=(
            _conversation_overlay(payload["conversation_resolution_overlay"])
            if isinstance(payload.get("conversation_resolution_overlay"), dict)
            else None
        ),
    )


def _conversation_overlay(payload: dict[str, Any]) -> ConversationResolutionOverlay:
    return ConversationResolutionOverlay(
        current_question=str(payload["current_question"]),
        value_frames=tuple(
            ConversationValueFrameOverlay(
                current_clause_text=str(item["current_clause_text"]),
                current_value_text=str(item["current_value_text"]),
                current_value_kind=str(item["current_value_kind"]),
                resolved_frame_text=str(item["resolved_frame_text"]),
                must_preserve_terms=tuple(item.get("must_preserve_terms") or ()),
                used_context_frame_ids=tuple(item.get("used_context_frame_ids") or ()),
            )
            for item in payload.get("value_frames") or ()
        ),
        references=tuple(
            ConversationDependencyOverlay(
                current_clause_text=str(item["current_clause_text"]),
                anchor_text=str(item["anchor_text"]),
                occurrence=int(item.get("occurrence") or 1),
                resolved_text=str(item["resolved_text"]),
                must_preserve_terms=tuple(item.get("must_preserve_terms") or ()),
                source_ids=tuple(item.get("source_ids") or ()),
            )
            for item in payload.get("references") or ()
        ),
        scopes=(),
        activated_memory_ids=(),
        used_source_card_ids=(),
    )
