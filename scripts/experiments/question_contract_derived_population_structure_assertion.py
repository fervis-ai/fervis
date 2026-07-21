"""Validate model-authored population semantics after derived structure is removed."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    outcome = arguments.get("outcome") or {}
    requests = outcome.get("answer_requests") or []
    if len(requests) != 1:
        return [f"expected one answer request, got {len(requests)}"]
    request = requests[0]
    errors: list[str] = []
    subject = (request.get("answer_subject") or {}).get("subject_text")
    if not isinstance(subject, str) or not subject.strip():
        errors.append("answer subject lacks one candidate kind")
    tests = (request.get("answer_population") or {}).get("membership_tests") or []
    if any(test.get("kind") != "EXPLICIT_USER_CONSTRAINT" for test in tests):
        errors.append("population includes backend-derived membership structure")
    input_refs = {
        item.get("input_ref") for item in outcome.get("question_inputs") or []
    }
    consumed = {
        ref for test in tests for ref in test.get("question_input_refs") or []
    }
    if consumed != input_refs:
        errors.append("explicit population tests do not consume exactly the inputs")
    return errors
