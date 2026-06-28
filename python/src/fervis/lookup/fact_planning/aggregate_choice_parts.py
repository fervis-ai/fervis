"""Shared compact-choice primitives for aggregate fact planning."""

from __future__ import annotations

from html import escape
from typing import Any


AGGREGATE_FUNCTIONS = ("sum", "min", "max", "avg")
COUNT_FUNCTION = "count"


def aggregate_function_candidates(
    metric_candidates: tuple[dict[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    functions = tuple(
        dict.fromkeys(
            function
            for metric in metric_candidates
            for function in tuple(metric.get("allowed_functions") or ())
        )
    )
    return tuple(
        {
            "id": f"function_{function}",
            "value": function,
            "meaning": aggregate_function_meaning(function),
        }
        for function in functions
    )


def aggregate_function_meaning(function: str) -> str:
    if function == "sum":
        return "total across matching rows"
    if function == "max":
        return "largest value among matching rows"
    if function == "min":
        return "smallest value among matching rows"
    if function == "avg":
        return "average value across matching rows"
    if function == "count":
        return "number of matching rows"
    raise ValueError(f"unsupported aggregate function: {function}")


def selected_choice(
    selection: Any,
    candidates: Any,
    *,
    label: str,
    require_selection_basis: bool = True,
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise ValueError(f"{label} selection is required")
    if (
        require_selection_basis
        and not str(selection.get("selection_basis") or "").strip()
    ):
        raise ValueError(f"{label} selection requires selection_basis")
    selection_id = str(selection.get("id") or "")
    if not selection_id:
        raise ValueError(f"{label} selection requires id")
    matches = tuple(
        candidate
        for candidate in candidates or ()
        if isinstance(candidate, dict)
        and str(candidate.get("id") or "") == selection_id
    )
    if len(matches) != 1:
        raise ValueError(f"{label} selection references unavailable choice")
    return dict(matches[0])


def xml_attr(value: object) -> str:
    return escape(str(value or "").strip(), quote=True)


def xml_text(value: object) -> str:
    return escape(str(value or "").strip())
