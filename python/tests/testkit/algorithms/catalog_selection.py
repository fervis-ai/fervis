from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog.selection.model import CatalogSelectionResult
from fervis.lookup.relation_catalog.selection import (
    relation_catalog_for_read_ids,
    select_resolver_reads,
    selected_read_ids_from_fact_selections,
)
from fervis.lookup.relation_catalog.selection.selector.fact_selection import (
    select_resource_name_groups,
)
from tests.testkit.assertions import exact_mismatches, subset_mismatches
from tests.testkit.catalog import catalog_from_payload


def run_catalog_selection_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    catalog = catalog_from_payload(input_payload["catalog"])
    fact_ids = tuple(str(item["id"]) for item in input_payload["requested_facts"])
    raw_matches = tuple(input_payload.get("resource_name_matches") or ())
    matches_by_fact = {str(item["requested_fact_id"]): item for item in raw_matches}
    if len(matches_by_fact) != len(raw_matches) or set(matches_by_fact) != set(
        fact_ids
    ):
        raise ValueError("resource-name recall must cover every requested fact once")
    selections = tuple(
        select_resource_name_groups(
            requested_fact_id=fact_id,
            resource_name_groups=tuple(
                tuple(lineage.get("matching_resource_names") or ())
                for lineage in matches_by_fact[fact_id].get(
                    "answer_output_resource_lineage"
                )
                or ()
            ),
            relation_catalog=catalog,
            max_reads_per_fact=int(input_payload["max_reads_per_fact"]),
        )
        for fact_id in fact_ids
    )
    selected_read_ids = selected_read_ids_from_fact_selections(selections)
    result = CatalogSelectionResult(
        relation_catalog=relation_catalog_for_read_ids(
            catalog, read_ids=selected_read_ids
        ),
        requested_fact_selections=selections,
        selected_read_ids=selected_read_ids,
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
    catalog = catalog_from_payload(input_payload["catalog"])
    selections = tuple(
        (
            str(item["target_id"]),
            tuple(item.get("catalog_search_terms") or ()),
            select_resolver_reads(
                catalog,
                catalog_search_terms=tuple(item.get("catalog_search_terms") or ()),
                limit=int(input_payload["max_reads_per_target"]),
            ),
        )
        for item in input_payload.get("entity_targets") or ()
    )
    selected_read_ids = tuple(
        dict.fromkeys(read.id for _, _, reads in selections for read in reads)
    )
    actual = {
        "selected_read_ids": list(selected_read_ids),
        "selected_read_membership": {read_id: True for read_id in selected_read_ids},
        "entity_target_selections": [
            {
                "target_id": target_id,
                "catalog_search_terms": list(search_terms),
                "selected_read_ids": [read.id for read in reads],
                "selected_read_membership": {read.id: True for read in reads},
            }
            for target_id, search_terms, reads in selections
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
