from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import RowCardinality
from fervis.lookup.fact_plan.row_sources import RowSource, RowSourceKind
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    extract_row_source_rows,
)

from tests.testkit.assertions import subset_mismatches


def run_endpoint_response_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        rows = _extract_rows(input_payload)
    except EndpointResponseError as exc:
        expected_error = payload["expect"].get("error_contains")
        if expected_error and expected_error in str(exc):
            return []
        return [f"unexpected error: {exc}"]
    if "error_contains" in payload["expect"]:
        return [f"expected error containing {payload['expect']['error_contains']!r}"]
    return subset_mismatches(
        actual={"rows": list(rows)},
        expected_subset=payload["expect"]["result_contains"],
    )


def _extract_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    body = payload["body"]
    row_source = RowSource(
        id="case_row_source",
        kind=RowSourceKind.API_READ,
        label="case row source",
        row_path=str(payload["row_path"]),
        parent_row_path=str(payload.get("parent_row_path") or ""),
        row_cardinality=RowCardinality(str(payload.get("cardinality") or "many")),
    )
    return extract_row_source_rows(body, row_source=row_source)
