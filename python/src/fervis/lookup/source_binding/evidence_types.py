"""Type predicates for source-binding evidence items."""

from __future__ import annotations

from typing import Any


def evidence_item_can_measure(evidence_item: dict[str, Any]) -> bool:
    return str(evidence_item.get("type") or "").lower() in {
        "decimal",
        "float",
        "integer",
        "number",
        "numeric",
    }
