"""Render identifier helpers for pattern fact-plan compilation."""

from __future__ import annotations

import re


def _result_output_id(
    answer_index: int,
    output_field_id: str,
    *,
    namespace_result_outputs: bool,
) -> str:
    if not namespace_result_outputs:
        return output_field_id
    return f"answer_{answer_index}_{output_field_id}"


def _safe_field_id(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower()).strip("_")
    if not candidate:
        return "value"
    if candidate[0].isdigit():
        return f"field_{candidate}"
    return candidate
