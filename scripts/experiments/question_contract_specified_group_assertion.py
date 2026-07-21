"""Validate a group axis supplied by explicit question inputs."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    outcome = arguments.get("outcome") or {}
    requests = outcome.get("answer_requests") or []
    if len(requests) != 1:
        return [f"expected one answer request, got {len(requests)}"]
    request = requests[0]
    expression = request.get("answer_expression") or {}
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append(f"family is not grouped_aggregate: {expression.get('family')!r}")
    group_key = expression.get("group_key") or {}
    source_kind = (group_key.get("value_source") or {}).get("kind")
    if source_kind != "specified_question_inputs":
        errors.append(f"unexpected group source: {source_kind!r}")
    group_uses = [
        item
        for item in request.get("question_input_uses") or []
        if item.get("owner_kind") == "GROUP_KEY"
    ]
    if len(group_uses) != 2:
        errors.append(f"expected two GROUP_KEY inputs, got {len(group_uses)}")
    subject = (request.get("answer_subject") or {}).get("subject_text", "")
    if subject.strip().casefold() == str(group_key.get("description", "")).strip().casefold():
        errors.append("answer_subject duplicates the group key")
    population_use_ids = {
        item.get("use_id")
        for item in request.get("question_input_uses") or []
        if item.get("owner_kind") == "POPULATION_TESTS"
    }
    referenced_use_ids = {
        ref
        for test in (request.get("answer_population") or {}).get(
            "membership_tests", []
        )
        for ref in test.get("population_use_refs") or []
    }
    if population_use_ids - referenced_use_ids:
        errors.append("population input is not referenced by a membership test")
    return errors
