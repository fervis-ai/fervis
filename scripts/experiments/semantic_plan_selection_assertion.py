"""Assertions for the semantic Plan Selection boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_semantic_plan_selection_boundary import build_request
from fervis.lookup.plan_selection.semantic_parser import (
    parse_semantic_plan_selection,
)


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    request = build_request(context)
    try:
        [strategy] = parse_semantic_plan_selection(arguments, request=request)
    except ValueError as exc:
        return [f"semantic plan selection parser rejected output: {exc}"]
    if len(strategy.branches) != 1:
        return ["strategy did not contain exactly one branch"]
    [branch] = strategy.branches
    direct_refs = {
        item.source_ref
        for item in strategy.source_assessments
        if item.alignment.value == "DIRECT"
    }
    if len(branch.source_refs) != 1 or branch.source_refs[0] not in direct_refs:
        return ["strategy did not select one direct source"]
    return []


__all__ = ["validate"]
