from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog.selection import (
    AnswerOutputResourceLineage,
    CatalogSelectionRequest,
    EntityTargetCatalogSearchTerms,
    RequestedFactResourceNameMatches,
    ResolverCatalogSelectionRequest,
    select_resolver_relation_catalog,
    select_relation_catalog,
)
from tests.testkit.assertions import exact_mismatches, subset_mismatches
from tests.testkit.catalog import catalog_from_payload
from tests.testkit.question_contract import requested_fact_from_payload
from tests.testkit.values import fact_value_from_payload


def run_catalog_selection_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    result = select_relation_catalog(
        CatalogSelectionRequest(
            relation_catalog=catalog_from_payload(input_payload["catalog"]),
            requested_facts=tuple(
                requested_fact_from_payload(item)
                for item in input_payload.get("requested_facts") or ()
            ),
            resource_name_matches=tuple(
                _resource_name_matches(item)
                for item in input_payload.get("resource_name_matches") or ()
            ),
            available_values=tuple(
                fact_value_from_payload(item)
                for item in input_payload.get("available_values") or ()
            ),
            max_reads_per_fact=int(input_payload["max_reads_per_fact"]),
        )
    )
    actual = {
        "selected_read_ids": list(result.selected_read_ids),
        "selected_read_id_set": sorted(result.selected_read_ids),
        "selected_read_membership": {
            read_id: True for read_id in result.selected_read_ids
        },
        "selected_catalog": {
            "read_ids": [item.id for item in result.relation_catalog.reads],
            "fact_refs": [item.ref for item in result.relation_catalog.facts],
        },
        "requested_fact_selections": [
            {
                "requested_fact_id": item.requested_fact_id,
                "query_terms": list(item.query_terms),
                "selected_read_ids": list(item.selected_read_ids),
                "selected_read_id_set": sorted(item.selected_read_ids),
                "selected_read_membership": {
                    read_id: True for read_id in item.selected_read_ids
                },
                "unselected_positive_read_ids": list(item.unselected_positive_read_ids),
                "rankings": [
                    {
                        "read_id": ranking.read_id,
                        "score": ranking.score,
                        "matched_terms": list(ranking.matched_terms),
                        "matched_fact_refs": list(ranking.matched_fact_refs),
                        "matched_field_refs": list(ranking.matched_field_refs),
                    }
                    for ranking in item.rankings
                ],
            }
            for item in result.requested_fact_selections
        ],
    }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual, expected=payload["expect"]["result_equals"]
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def run_resolver_catalog_selection_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    result = select_resolver_relation_catalog(
        ResolverCatalogSelectionRequest(
            relation_catalog=catalog_from_payload(input_payload["catalog"]),
            entity_target_catalog_search_terms=tuple(
                EntityTargetCatalogSearchTerms(
                    target_id=str(item["target_id"]),
                    catalog_search_terms=tuple(item.get("catalog_search_terms") or ()),
                )
                for item in input_payload.get("entity_targets") or ()
            ),
            max_reads_per_target=int(input_payload["max_reads_per_target"]),
        )
    )
    actual = {
        "selected_read_ids": list(result.selected_read_ids),
        "selected_read_membership": {
            read_id: True for read_id in result.selected_read_ids
        },
        "entity_target_selections": [
            {
                "target_id": item.target_id,
                "catalog_search_terms": list(item.catalog_search_terms),
                "selected_read_ids": list(item.selected_read_ids),
                "selected_read_membership": {
                    read_id: True for read_id in item.selected_read_ids
                },
            }
            for item in result.entity_target_selections
        ],
    }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual, expected=payload["expect"]["result_equals"]
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _resource_name_matches(
    payload: dict[str, Any],
) -> RequestedFactResourceNameMatches:
    return RequestedFactResourceNameMatches(
        requested_fact_id=str(payload["requested_fact_id"]),
        answer_output_resource_lineage=tuple(
            AnswerOutputResourceLineage(
                answer_output_id=str(item["answer_output_id"]),
                support_role=str(item["support_role"]),
                source_text=str(item["source_text"]),
                matching_resource_names=tuple(
                    item.get("matching_resource_names") or ()
                ),
            )
            for item in payload.get("answer_output_resource_lineage") or ()
        ),
    )
