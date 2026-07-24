"""Assertions for the reusable semantic Grounding boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_semantic_grounding_boundary import build_request
from fervis.lookup.grounding.semantic_parser import parse_semantic_grounding


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    request = build_request(context)
    try:
        result = parse_semantic_grounding(arguments, request=request)
    except ValueError as exc:
        return [f"semantic grounding parser rejected output: {exc}"]
    if context.get("kind") == "time":
        if result.identity_tasks:
            return ["time-only grounding produced an identity task"]
        if len(result.canonical_values) != 1:
            return ["time input did not produce one canonical value"]
        [value] = result.canonical_values
        expected = (str(context["expected_start"]), str(context["expected_end"]))
        actual = (
            value.typed_value.payload.resolved_start,
            value.typed_value.payload.resolved_end,
        )
        return (
            []
            if actual == expected
            else [f"resolved bounds are {actual}, expected {expected}"]
        )
    [task] = result.identity_tasks
    expected_read = str(context["expected_read"])
    reads = [route.option.candidate.resolver_read_id for route in task.resolver_routes]
    if reads != [expected_read]:
        return [f"compatible resolver reads are {reads}, expected {[expected_read]}"]
    if len(task.canonical_options) != 1:
        return ["expected one canonical identity meaning"]
    return []


__all__ = ["validate"]
