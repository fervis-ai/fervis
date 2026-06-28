from __future__ import annotations

from typing import Any

from fervis.lookup.fact_planning.fact_requirements import fact_endpoint_requirements
from fervis.lookup.fact_plan.row_sources import build_row_source_catalog

from tests.testkit.assertions import subset_mismatches
from tests.testkit.catalog import (
    catalog_from_payload,
    catalog_selection_from_payload,
)


def run_fact_requirements_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    catalog = catalog_from_payload(input_payload["catalog"])
    row_sources = build_row_source_catalog(catalog)
    requirements = fact_endpoint_requirements(
        catalog=catalog,
        catalog_selection=catalog_selection_from_payload(
            input_payload["catalog_selection"],
            catalog=catalog,
        ),
        available_values=(),
        available_value_uses=(),
        row_sources=row_sources,
    )
    return subset_mismatches(
        actual=_requirements_payload(requirements, row_sources=row_sources),
        expected_subset=payload["expect"]["result_contains"],
    )


def _requirements_payload(requirements: Any, *, row_sources: Any) -> dict[str, Any]:
    return {
        "requested_facts": [
            {
                "requested_fact_id": item.requested_fact_id,
                "executable_sources": [
                    _source_ref(row_sources.source(source_id))
                    for source_id in item.executable_row_source_ids
                ],
                "missing_inputs": [
                    {
                        "read_id": row_sources.source(input_item.row_source_id).read_id,
                        "row_path_id": row_sources.source(
                            input_item.row_source_id
                        ).row_path_id,
                        "param_id": input_item.param_id,
                    }
                    for input_item in item.missing_inputs
                ],
            }
            for item in requirements.requested_facts
        ],
        "clarifiable_missing_inputs": [
            {
                "read_id": row_sources.source(input_item.row_source_id).read_id,
                "row_path_id": row_sources.source(input_item.row_source_id).row_path_id,
                "param_id": input_item.param_id,
            }
            for input_item in requirements.clarifiable_missing_inputs
        ],
    }


def _source_ref(source: Any) -> dict[str, str]:
    return {
        "read_id": source.read_id,
        "row_path_id": source.row_path_id,
    }
