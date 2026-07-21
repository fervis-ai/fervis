"""Validate compositional Question Contract result shapes."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    outcome = arguments.get("outcome") or {}
    requests = outcome.get("answer_requests") or []
    if len(requests) != 1:
        return [f"expected one answer request, got {len(requests)}"]
    request = requests[0]
    expression = request.get("answer_expression") or {}
    label = str(context.get("label") or "")
    if label in {"first_two", "top_three"}:
        return _validate_bounded_take(
            outcome,
            request,
            expression,
            expected_limit="2" if label == "first_two" else "3",
        )
    if label == "group_by_dimension":
        return _validate_grouped_aggregate(
            outcome,
            request,
            expression,
            expected_grain=None,
        )
    if label == "aggregate_by_grain":
        return _validate_grouped_aggregate(
            outcome,
            request,
            expression,
            expected_grain="day",
        )
    if label == "specified_groups":
        return _validate_specified_groups(outcome, request, expression)
    if label == "historical_specified_groups":
        return _validate_historical_specified_groups(outcome, request, expression)
    return [f"unknown assertion label: {label}"]


def _validate_bounded_take(
    outcome: dict[str, Any],
    request: dict[str, Any],
    expression: dict[str, Any],
    *,
    expected_limit: str,
) -> list[str]:
    errors: list[str] = []
    if expression.get("family") != "list_rows":
        errors.append("bounded row request is not list_rows")
    if (expression.get("selection") or {}).get("kind") != "take":
        errors.append("explicit result count did not produce take")
    if not expression.get("ordering"):
        errors.append("bounded take lacks ordering")
    limits = [
        item
        for item in outcome.get("question_inputs") or []
        if item.get("role") == "result_limit"
    ]
    if len(limits) != 1 or str(limits[0].get("operand_text") or "") != expected_limit:
        errors.append(f"expected one canonical result limit {expected_limit}")
        return errors
    input_ref = limits[0].get("input_ref")
    owners = [
        use
        for use in request.get("question_input_uses") or []
        if use.get("input_ref") == input_ref
    ]
    if len(owners) != 1 or owners[0].get("owner_kind") != "RESULT_LIMIT":
        errors.append("result limit is not exclusively RESULT_LIMIT-owned")
    return errors


def _validate_historical_specified_groups(
    outcome: dict[str, Any],
    request: dict[str, Any],
    expression: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append("specified-member measure is not grouped_aggregate")
    if (expression.get("group_key") or {}).get("domain") != "SPECIFIED_QUESTION_INPUTS":
        errors.append("specified groups lack their closed domain")
    reference_inputs = {
        item.get("input_ref")
        for item in outcome.get("question_inputs") or []
        if item.get("role") == "reference_value"
    }
    uses = request.get("question_input_uses") or []
    group_inputs = {
        use.get("input_ref")
        for use in uses
        if use.get("owner_kind") == "GROUP_KEY"
    }
    if len(reference_inputs) != 2 or group_inputs != reference_inputs:
        errors.append("specified group key does not own both reference inputs")
    population_use_ids = {
        use.get("use_id")
        for use in uses
        if use.get("owner_kind") == "POPULATION_TESTS"
    }
    consumed_use_ids = {
        use_id
        for test in (request.get("answer_population") or {}).get("membership_tests") or []
        for use_id in test.get("question_input_use_refs") or []
    }
    if population_use_ids != consumed_use_ids:
        errors.append("population tests do not consume exactly their uses")
    return errors


def _validate_specified_groups(
    outcome: dict[str, Any],
    request: dict[str, Any],
    expression: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append("specified-member measure is not grouped_aggregate")
    value_source = (expression.get("group_key") or {}).get("value_source") or {}
    if value_source.get("kind") != "specified_question_inputs":
        errors.append("supplied group members are not specified_question_inputs")
    reference_inputs = {
        item.get("input_ref")
        for item in outcome.get("question_inputs") or []
        if item.get("role") == "reference_value"
    }
    if len(reference_inputs) != 2:
        errors.append(f"expected two reference-value group members, got {len(reference_inputs)}")
    uses = request.get("question_input_uses") or []
    group_refs = {
        use.get("input_ref")
        for use in uses
        if use.get("owner_kind") == "GROUP_KEY"
    }
    if group_refs != reference_inputs:
        errors.append("specified group key does not own exactly its supplied members")
    population_refs = {
        use.get("input_ref")
        for use in uses
        if use.get("owner_kind") == "POPULATION_TESTS"
    }
    consumed_population_refs = {
        input_ref
        for test in (request.get("answer_population") or {}).get("membership_tests") or []
        for input_ref in test.get("question_input_refs") or []
    }
    if consumed_population_refs != population_refs:
        errors.append("population tests do not consume exactly the population inputs")
    if reference_inputs & consumed_population_refs:
        errors.append("specified group members are duplicated as population inputs")
    return errors


def _validate_grouped_aggregate(
    outcome: dict[str, Any],
    request: dict[str, Any],
    expression: dict[str, Any],
    *,
    expected_grain: str | None,
) -> list[str]:
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append("per-group aggregate is not grouped_aggregate")
    group_key = expression.get("group_key") or {}
    value_source = group_key.get("value_source") or {}
    if (expression.get("selection") or {}).get("kind") != "all_results":
        errors.append("unbounded grouped result does not keep all groups")
    grains = [
        item
        for item in outcome.get("question_inputs") or []
        if item.get("role") == "grouping_grain"
    ]
    if expected_grain is None:
        if grains:
            errors.append("grouping dimension became a grouping_grain input")
        if value_source.get("kind") != "source_value":
            errors.append("source grouping dimension is not a source_value")
        return errors
    if grains:
        errors.append("temporal grouping configuration became a question input")
    if value_source.get("kind") != "temporal_bucket":
        errors.append("temporal grouping does not use temporal_bucket")
    if str(value_source.get("grain") or "").casefold() != expected_grain:
        errors.append(f"expected temporal bucket grain {expected_grain}")
    group_inputs = [
        item
        for item in request.get("question_input_uses") or []
        if item.get("owner_kind") == "GROUP_KEY"
    ]
    if group_inputs:
        errors.append("source-derived group key has GROUP_KEY question inputs")
    return errors
