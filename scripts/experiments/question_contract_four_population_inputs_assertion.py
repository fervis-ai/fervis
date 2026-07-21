"""Validate four distinct population operands and their fact-local ownership."""

from __future__ import annotations

from typing import Any


_EXPECTED = {
    "in-person": "predicate_value",
    "1000": "threshold_value",
    "Acacia Mall": "reference_value",
    "March 2026": "time_value",
}


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    outcome = arguments.get("outcome") or {}
    requests = outcome.get("answer_requests") or []
    if len(requests) != 1:
        return [f"expected one answer request, got {len(requests)}"]

    errors: list[str] = []
    actual: dict[str, str] = {}
    input_refs: dict[str, str] = {}
    for item in outcome.get("question_inputs") or []:
        text = str(item.get("operand_text") or "")
        if item.get("role") == "threshold_value" and "1000" in text:
            text = "1000"
        actual[text] = str(item.get("role") or "")
        input_refs[text] = str(item.get("input_ref") or "")
    if actual != _EXPECTED:
        errors.append(f"input inventory is {actual!r}, expected {_EXPECTED!r}")
        return errors

    request = requests[0]
    tests = (request.get("answer_population") or {}).get("membership_tests") or []
    test_refs = {
        ref for test in tests for ref in test.get("question_input_refs") or []
    }
    if test_refs != set(input_refs.values()):
        errors.append("population tests do not consume all four inputs exactly")
    uses = request.get("question_input_uses") or []
    use_refs = {
        str(item.get("input_ref") or "")
        for item in uses
        if item.get("owner_kind") == "POPULATION_TESTS"
    }
    if use_refs != set(input_refs.values()):
        errors.append("population ownership does not cover all four inputs")
    return errors
