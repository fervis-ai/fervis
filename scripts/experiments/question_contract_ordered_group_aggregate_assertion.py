"""Validate an ordered aggregate computed independently for each candidate."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    outcome = arguments.get("outcome") or {}
    requests = outcome.get("answer_requests") or []
    if len(requests) != 1:
        return [f"expected one answer request, got {len(requests)}"]
    expression = requests[0].get("answer_expression") or {}
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append(f"family is not grouped_aggregate: {expression.get('family')!r}")
    if not expression.get("group_key"):
        errors.append("group_key is missing")
    else:
        group_key = expression["group_key"]
        source_kind = (group_key.get("value_source") or {}).get("kind")
        if source_kind != "source_value":
            errors.append(f"group key is not source-derived: {source_kind!r}")
        subject = (requests[0].get("answer_subject") or {}).get("subject_text", "")
        if subject.strip().casefold() == str(group_key.get("description", "")).strip().casefold():
            errors.append("answer_subject duplicates the group key")
    if not expression.get("ordering"):
        errors.append("ordering is missing")
    if (expression.get("selection") or {}).get("kind") != "take_one":
        errors.append(f"unexpected selection: {expression.get('selection')!r}")
    uses = requests[0].get("question_input_uses") or []
    population_use_ids = {
        item.get("use_id")
        for item in uses
        if item.get("owner_kind") == "POPULATION_TESTS"
    }
    referenced_use_ids = {
        ref
        for test in (requests[0].get("answer_population") or {}).get(
            "membership_tests", []
        )
        for ref in test.get("population_use_refs") or []
    }
    if population_use_ids - referenced_use_ids:
        errors.append("population input is not referenced by a membership test")
    return errors
