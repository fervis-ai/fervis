"""Stable lineage row identifiers."""

from __future__ import annotations

from hashlib import sha256
import json


def lineage_id(prefix: str, *parts: object) -> str:
    normalized = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"
