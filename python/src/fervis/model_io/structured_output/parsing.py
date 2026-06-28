"""Structured-output payload parsing."""

from __future__ import annotations

import json
from typing import Any


def tool_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(str(raw or "{}"))
    if not isinstance(parsed, dict):
        raise ValueError("Provider tool output must be a JSON object.")
    return parsed


def raw_tool_output_text(output: dict[str, Any]) -> str:
    return json.dumps(output, default=str, sort_keys=True)
