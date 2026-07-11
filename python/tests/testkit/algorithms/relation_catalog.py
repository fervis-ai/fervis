from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import validate_relation_catalog
from fervis.lookup.fact_planning.required_inputs import required_inputs
from fervis.lookup.fact_plan.row_sources import (
    build_row_source_catalog,
    row_source_prompt_payload,
    row_sources_for_read_id,
)

from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)
from tests.testkit.catalog import catalog_from_payload


def run_relation_catalog_case(payload: dict[str, Any]) -> list[str]:
    catalog = catalog_from_payload(payload["input"]["catalog"])
    try:
        catalog = validate_relation_catalog(catalog)
    except Exception as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    row_sources = build_row_source_catalog(catalog)
    expected = payload["expect"]["result_contains"]
    source = _source_for_expected(row_sources, expected)
    api_sources = [item for item in row_sources.sources if item.kind.value == "api_read"]
    prompt_sources = row_source_prompt_payload(row_sources)["row_sources"]
    prompt_api_sources = [
        item for item in prompt_sources if item.get("kind") == "api_read"
    ]
    required = required_inputs(row_sources)
    api_source_ids = {item.id for item in api_sources}
    result = {
        "read_id": source.read_id,
        "endpoint_name": catalog.read(source.read_id).endpoint_name,
        "row_path_id": source.row_path_id,
        "fields": {
            field.id: {
                "field_ref": field.field_ref,
                "type": field.type,
            }
            for field in source.fields
        },
        "params": {
            param.id: {
                "param_ref": param.param_ref,
                "default": param.default,
            }
            for param in source.params
        },
        "sources": [
            {
                "id": item.id,
                "kind": item.kind.value,
                "read_id": item.read_id,
                "row_path_id": item.row_path_id,
                "fields": [
                    {
                        "field_id": field.id,
                        "field_ref": field.field_ref,
                        "type": field.type,
                    }
                    for field in item.fields
                ],
                "params": [
                    {
                        "param_id": param.id,
                        "param_ref": param.param_ref,
                        "default": param.default,
                    }
                    for param in item.params
                ],
                "blocked_facts": [
                    {
                        "fact_ref": fact.fact_ref,
                        "availability": fact.availability,
                        "field_id": fact.field_id,
                        "proof_refs": list(fact.proof_refs),
                    }
                    for fact in item.blocked_facts
                ],
            }
            for item in row_sources.sources
        ],
        "source_id_properties": {
            "count": len(row_sources.sources),
            "unique_count": len({item.id for item in row_sources.sources}),
            "all_opaque_prefix": all(
                item.id.startswith("rs_") for item in row_sources.sources
            ),
        },
        "api_source_id_properties": {
            "count": len(api_sources),
            "unique_count": len({item.id for item in api_sources}),
            "all_opaque_prefix": all(item.id.startswith("rs_") for item in api_sources),
        },
        "api_sources": [
            {
                "id": item.id,
                "kind": item.kind.value,
                "read_id": item.read_id,
                "row_path_id": item.row_path_id,
                "fields": [
                    {
                        "field_id": field.id,
                        "field_ref": field.field_ref,
                        "type": field.type,
                    }
                    for field in item.fields
                ],
                "params": [
                    {
                        "param_id": param.id,
                        "param_ref": param.param_ref,
                        "default": param.default,
                    }
                    for param in item.params
                ],
                "blocked_facts": [
                    {
                        "fact_ref": fact.fact_ref,
                        "availability": fact.availability,
                        "field_id": fact.field_id,
                        "proof_refs": list(fact.proof_refs),
                    }
                    for fact in item.blocked_facts
                ],
            }
            for item in api_sources
        ],
        "required_inputs": [
            {
                "read_id": row_sources.source(item.row_source_id).read_id,
                "row_path_id": row_sources.source(item.row_source_id).row_path_id,
                "param_id": item.param_id,
            }
            for item in required
        ],
        "api_required_inputs": [
            {
                "read_id": row_sources.source(item.row_source_id).read_id,
                "row_path_id": row_sources.source(item.row_source_id).row_path_id,
                "param_id": item.param_id,
            }
            for item in required
            if item.row_source_id in api_source_ids
        ],
        "prompt_row_sources": prompt_sources,
        "prompt_api_row_sources": prompt_api_sources,
    }
    return subset_mismatches(actual=result, expected_subset=expected)


def _source_for_expected(row_sources: Any, expected: dict[str, Any]) -> Any:
    read_id = str(expected["read_id"])
    row_path_id = str(expected.get("row_path_id") or "root")
    for source in row_sources_for_read_id(read_id, row_sources=row_sources):
        if source.row_path_id == row_path_id:
            return source
    raise KeyError(f"row source not found for {read_id}:{row_path_id}")
