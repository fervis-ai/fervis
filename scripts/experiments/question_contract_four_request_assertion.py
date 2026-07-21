"""Validate a four-fact Question Contract without a redundant request count."""

from __future__ import annotations

from typing import Any


_EXPECTED_FACT_MARKERS = {
    "sales": ("sales",),
    "staff shifts": ("staff", "shift"),
    "payroll": ("payroll",),
    "cash deposits": ("cash", "deposit"),
}


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    outcome = arguments.get("outcome") or {}
    errors: list[str] = []
    if "answer_requests_count" in outcome:
        errors.append("outcome still authors answer_requests_count")

    requests = outcome.get("answer_requests") or []
    if len(requests) != len(_EXPECTED_FACT_MARKERS):
        errors.append(f"expected four answer requests, got {len(requests)}")
        return errors

    facts = [str(request.get("answer_fact") or "").lower() for request in requests]
    for label, markers in _EXPECTED_FACT_MARKERS.items():
        matching = [fact for fact in facts if all(marker in fact for marker in markers)]
        if len(matching) != 1:
            errors.append(f"expected exactly one {label} request, got {len(matching)}")

    for request in requests:
        expression = request.get("answer_expression") or {}
        if expression.get("family") != "scalar_aggregate":
            errors.append(
                f"request is not scalar_aggregate: {request.get('answer_fact')!r}"
            )

    inputs = outcome.get("question_inputs") or []
    if len(inputs) != 1 or inputs[0].get("value_source_text") != "March 2026":
        errors.append("shared March 2026 input is not declared exactly once")
        return errors

    input_ref = inputs[0].get("input_ref")
    for request in requests:
        tests = (request.get("answer_population") or {}).get("membership_tests") or []
        used_refs = {
            ref for test in tests for ref in test.get("question_input_refs") or []
        }
        if input_ref not in used_refs:
            errors.append(
                f"request does not consume the shared time input: "
                f"{request.get('answer_fact')!r}"
            )
    return errors
