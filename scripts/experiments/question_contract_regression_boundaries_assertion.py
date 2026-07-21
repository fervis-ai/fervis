"""Validate established Question Contract boundaries during stability replays."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    outcome = arguments.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("kind") != "question_contract":
        return ["outcome is not a question contract"]
    requests = outcome.get("answer_requests")
    inputs = outcome.get("question_inputs")
    if not isinstance(requests, list) or len(requests) != 1:
        return ["expected exactly one answer request"]
    if not isinstance(inputs, list):
        return ["question_inputs is not an array"]
    request = requests[0]
    if not isinstance(request, dict):
        return ["answer request is not an object"]
    label = str(context.get("label") or "")
    if label == "area_reference":
        return _area_reference(inputs, request)
    if label == "cash_subject":
        return _cash_subject(inputs, request)
    if label == "choice_time":
        return _choice_time(inputs, request)
    if label == "specified_group":
        return _specified_group(inputs, request)
    if label == "ordered_group_total":
        return _ordered_group_total(inputs, request)
    return [f"unknown assertion label: {label}"]


def _input_by_operand(inputs: list[object]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("operand_text") or "").casefold(): item
        for item in inputs
        if isinstance(item, dict)
    }


def _uses(request: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("input_ref")): str(item.get("owner_kind"))
        for item in request.get("question_input_uses") or []
        if isinstance(item, dict)
    }


def _test_refs(request: dict[str, Any]) -> set[str]:
    population = request.get("answer_population") or {}
    tests = [
        item
        for item in population.get("membership_tests") or []
        if isinstance(item, dict)
    ]
    test_ids = {str(item.get("test_id")) for item in tests if item.get("test_id")}
    ledger_refs = {
        str(item.get("input_ref"))
        for item in request.get("question_input_uses") or []
        if isinstance(item, dict)
        and item.get("owner_kind") == "POPULATION_TESTS"
        and test_ids.intersection(
            str(ref) for ref in item.get("membership_test_ids") or []
        )
    }
    if ledger_refs:
        return ledger_refs
    inputs_by_use_id = {
        str(item.get("use_id")): str(item.get("input_ref"))
        for item in request.get("question_input_uses") or []
        if isinstance(item, dict)
        and item.get("owner_kind") == "POPULATION_TESTS"
    }
    return {
        inputs_by_use_id.get(str(ref), "")
        for test in tests
        for ref in (
            test.get("population_use_refs")
            or test.get("question_input_use_refs")
            or []
        )
    }


def _orphan_test_ids(request: dict[str, Any]) -> set[str]:
    population = request.get("answer_population") or {}
    declared = {
        str(item.get("test_id"))
        for item in population.get("membership_tests") or []
        if isinstance(item, dict) and item.get("test_id")
    }
    consumed = {
        str(test_id)
        for item in request.get("question_input_uses") or []
        if isinstance(item, dict) and item.get("owner_kind") == "POPULATION_TESTS"
        for test_id in item.get("membership_test_ids") or []
    }
    return declared - consumed


def _area_reference(
    inputs: list[object], request: dict[str, Any]
) -> list[str]:
    by_operand = _input_by_operand(inputs)
    item = by_operand.get("nairobi")
    errors: list[str] = []
    if len(by_operand) != 1 or item is None:
        errors.append("Nairobi is not the sole question input")
        return errors
    if item.get("role") != "reference_value":
        errors.append("Nairobi is not a reference_value")
    ref = str(item.get("input_ref"))
    if _uses(request) != {ref: "POPULATION_TESTS"} or _test_refs(request) != {ref}:
        errors.append("Nairobi is not owned by its population predicate")
    return errors


def _cash_subject(inputs: list[object], request: dict[str, Any]) -> list[str]:
    by_operand = _input_by_operand(inputs)
    errors: list[str] = []
    if set(by_operand) != {"this month"}:
        errors.append(f"cash-deposit subject became an input: {sorted(by_operand)}")
        return errors
    item = by_operand["this month"]
    ref = str(item.get("input_ref"))
    if item.get("role") != "time_value":
        errors.append("this month is not a time_value")
    if _uses(request) != {ref: "POPULATION_TESTS"} or _test_refs(request) != {ref}:
        errors.append("this month is not owned by its population predicate")
    return errors


def _choice_time(inputs: list[object], request: dict[str, Any]) -> list[str]:
    by_operand = _input_by_operand(inputs)
    errors: list[str] = []
    if set(by_operand) != {"in-person", "this month"}:
        errors.append(f"choice/time inventory is {sorted(by_operand)}")
        return errors
    if by_operand["in-person"].get("role") != "predicate_value":
        errors.append("in-person is not a predicate_value")
    if by_operand["this month"].get("role") != "time_value":
        errors.append("this month is not a time_value")
    refs = {str(item.get("input_ref")) for item in by_operand.values()}
    if _uses(request) != {ref: "POPULATION_TESTS" for ref in refs}:
        errors.append("choice/time inputs are not exclusively population-owned")
    if _test_refs(request) != refs:
        errors.append("population predicates do not consume both inputs")
    return errors


def _specified_group(inputs: list[object], request: dict[str, Any]) -> list[str]:
    by_operand = _input_by_operand(inputs)
    expression = request.get("answer_expression") or {}
    value_source = (expression.get("group_key") or {}).get("value_source") or {}
    staff_refs = {
        str(item.get("input_ref"))
        for operand, item in by_operand.items()
        if operand != "today"
    }
    today = by_operand.get("today")
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append("pair request is not grouped_aggregate")
    if value_source != {"kind": "specified_question_inputs"}:
        errors.append("group key is not specified_question_inputs")
    if today is None:
        errors.append("today input is missing")
        return errors
    today_ref = str(today.get("input_ref"))
    expected_uses = {
        **{ref: "GROUP_KEY" for ref in staff_refs},
        today_ref: "POPULATION_TESTS",
    }
    if _uses(request) != expected_uses:
        errors.append("question-input ownership does not match group and population")
    if _test_refs(request) != {today_ref}:
        errors.append("group members leaked into population predicates")
    if orphan_test_ids := _orphan_test_ids(request):
        errors.append(f"membership tests lack operands: {sorted(orphan_test_ids)}")
    return errors


def _ordered_group_total(
    inputs: list[object], request: dict[str, Any]
) -> list[str]:
    expression = request.get("answer_expression") or {}
    value_source = (expression.get("group_key") or {}).get("value_source") or {}
    errors: list[str] = []
    if expression.get("family") != "grouped_aggregate":
        errors.append("ordered per-group total is not grouped_aggregate")
    if value_source.get("kind") != "source_value":
        errors.append("group key is not source_value")
    if (expression.get("selection") or {}).get("kind") != "take_one":
        errors.append("superlative selection is not take_one")
    if not expression.get("ordering"):
        errors.append("ordered group total lacks ordering")
    by_operand = _input_by_operand(inputs)
    if set(by_operand) != {"this month"}:
        errors.append(f"time inventory is {sorted(by_operand)}")
    return errors
