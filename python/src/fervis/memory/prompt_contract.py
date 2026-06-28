"""Model-facing memory contract projection."""

from __future__ import annotations

from typing import Any


def planner_memory_contract(memory_frame: dict[str, Any]) -> dict[str, Any]:
    index = memory_frame.get("memoryIndex") if isinstance(memory_frame, dict) else None
    if not isinstance(index, dict):
        return {"available": False, "factArtifactCount": 0}
    items = [item for item in index.get("items") or () if isinstance(item, dict)]
    artifact_ids = {
        str(item.get("artifactId") or "").strip()
        for item in items
        if str(item.get("artifactId") or "").strip()
    }
    return {
        "available": bool(items),
        "factArtifactCount": len(artifact_ids),
        "indexOrder": index.get("order") or "newest_first",
        "indexTruncated": bool(index.get("truncated") is True),
        "addressKinds": _address_kinds(index),
    }


def _address_kinds(index: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in index.get("items") or ():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        if kind and kind not in values:
            values.append(kind)
    return values
