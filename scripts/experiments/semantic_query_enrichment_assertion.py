"""Semantic assertions for the reusable Query Enrichment boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_semantic_query_enrichment_boundary import request_from_payload  # noqa: E402
from fervis.lookup.query_enrichment.semantic_parser import (  # noqa: E402
    parse_semantic_query_enrichment,
)


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    try:
        result = parse_semantic_query_enrichment(
            arguments,
            request=request_from_payload(context),
        )
    except ValueError as exc:
        return [f"semantic query enrichment parser rejected output: {exc}"]

    matches = {item.bucket_ref: item for item in result.recall_bucket_matches}
    errors: list[str] = []
    for bucket in context["recall_buckets"]:
        expected = set(bucket.get("expected_any_resource_names") or ())
        actual = set(matches[bucket["bucket_ref"]].exhaustive_resource_names)
        if expected and not expected.intersection(actual):
            errors.append(
                f"{bucket['bucket_ref']} lacks one of "
                f"{bucket['expected_any_resource_names']}"
            )
    search_terms = {
        item.input_use_ref: set(item.catalog_search_terms)
        for item in result.input_resource_search_terms
    }
    for task in context.get("reference_tasks") or ():
        expected = set(task.get("expected_any_resource_names") or ())
        actual = search_terms[task["input_use_ref"]]
        if expected and not expected.intersection(actual):
            errors.append(
                f"{task['input_use_ref']} lacks one of "
                f"{task['expected_any_resource_names']}"
            )
    return errors


__all__ = ["validate"]
