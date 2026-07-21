"""Validate four same-population aggregate requests remain separate facts."""

from __future__ import annotations

from typing import Any


_EXPECTED_FACT_MARKERS = {
    "count": ("how many",),
    "total": ("total", "revenue"),
    "average": ("average", "amount"),
    "maximum": ("largest", "amount"),
}


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    del context
    outcome = arguments.get("outcome") or {}
    errors: list[str] = []
    requests = outcome.get("answer_requests") or []
    if outcome.get("answer_requests_count") != len(_EXPECTED_FACT_MARKERS):
        errors.append("answer_requests_count does not commit to four facts")
    if len(requests) != len(_EXPECTED_FACT_MARKERS):
        errors.append(f"expected four answer requests, got {len(requests)}")
        return errors

    facts = [str(request.get("answer_fact") or "").lower() for request in requests]
    for label, markers in _EXPECTED_FACT_MARKERS.items():
        matching = [fact for fact in facts if all(marker in fact for marker in markers)]
        if len(matching) != 1:
            errors.append(f"expected exactly one {label} request, got {len(matching)}")

    for request in requests:
        if (request.get("answer_expression") or {}).get("family") != "scalar_aggregate":
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
        if input_ref not in {
            ref for test in tests for ref in test.get("question_input_refs") or []
        }:
            errors.append(
                f"request does not consume the shared time input: "
                f"{request.get('answer_fact')!r}"
            )
    return errors
