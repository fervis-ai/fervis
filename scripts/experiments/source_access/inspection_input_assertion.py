"""Semantic assertion for a captured inspection-input grounding turn."""

from __future__ import annotations

from typing import Any


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    expected = context.get("reads")
    if not isinstance(expected, dict) or not expected:
        return ["inspection input assertion requires expected read decisions"]
    actual = arguments.get("reads")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return ["inspection input decisions do not cover the expected reads"]
    errors = []
    for read_id, wanted in expected.items():
        observed = actual.get(read_id)
        if not isinstance(wanted, dict) or not isinstance(observed, dict):
            errors.append(f"{read_id}: inspection decision must be an object")
            continue
        if observed.get("kind") != wanted.get("kind"):
            errors.append(f"{read_id}: incorrect inspection decision kind")
            continue
        if wanted.get("kind") == "bound_arguments" and (
            observed.get("parameter_inputs") != wanted.get("parameter_inputs")
        ):
            errors.append(f"{read_id}: incorrect typed parameter-to-input mapping")
    return errors


__all__ = ["validate"]
