"""Outcome assertions for a captured production Query Enrichment turn."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    if not context.get("recall_buckets") and not context.get("reference_tasks"):
        return ["query enrichment assertion requires an expected recall outcome"]
    matches = {
        str(item.get("bucket_ref")): item
        for item in arguments.get("recall_bucket_matches", ())
        if isinstance(item, dict)
    }
    errors: list[str] = []
    for expected_bucket in context.get("recall_buckets", ()):
        bucket_ref = str(expected_bucket["bucket_ref"])
        actual = matches.get(bucket_ref)
        if actual is None:
            errors.append(f"recall bucket is missing: {bucket_ref}")
            continue
        expected = set(expected_bucket.get("expected_any_resource_names") or ())
        recalled = set(actual.get("exhaustive_resource_names") or ())
        if expected and not expected.intersection(recalled):
            errors.append(f"{bucket_ref} lacks one of {sorted(expected)}")
        first_limit = expected_bucket.get("first_resource_name_limit")
        first_expected = set(
            expected_bucket.get("expected_resource_names_within_first") or ()
        )
        if first_limit is not None and first_expected:
            field = str(
                expected_bucket.get("resource_name_field") or "matching_resource_names"
            )
            first_names = tuple(actual.get(field, ()))[: int(first_limit)]
            if not first_expected.intersection(first_names):
                errors.append(
                    f"{bucket_ref} first {first_limit} {field} values are "
                    f"{first_names!r}; expected one of {sorted(first_expected)!r}"
                )

    search_terms = {
        str(item.get("input_use_ref")): set(item.get("catalog_search_terms") or ())
        for item in arguments.get("input_resource_search_terms", ())
        if isinstance(item, dict)
    }
    for expected_task in context.get("reference_tasks", ()):
        input_use_ref = str(expected_task["input_use_ref"])
        expected = set(expected_task.get("expected_any_resource_names") or ())
        actual = search_terms.get(input_use_ref)
        if actual is None:
            errors.append(f"input resource search is missing: {input_use_ref}")
        elif expected and not expected.intersection(actual):
            errors.append(f"{input_use_ref} lacks one of {sorted(expected)}")
    return errors


__all__ = ["validate"]
