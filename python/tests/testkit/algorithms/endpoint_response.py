from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import RowCardinality
from fervis.lookup.fact_plan.row_sources import RowSource, RowSourceKind
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    extract_row_source_rows,
)

from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def run_endpoint_response_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    try:
        rows = _extract_rows(input_payload)
    except EndpointResponseError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected",
                expected=payload["expect"],
            )
        return [f"unexpected error: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
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
        parent_row_cardinality=(
            RowCardinality(str(payload["parent_cardinality"]))
            if payload.get("parent_cardinality")
            else None
        ),
        row_cardinality=RowCardinality(str(payload.get("cardinality") or "many")),
    )
    return extract_row_source_rows(body, row_source=row_source)
