"""Shared result-data accessors for question/run surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from fervis.observability.event_contracts import EventPayloadKey


def result_data_clarifications(
    result_data: Mapping[str, Any] | None,
) -> list[Any]:
    if result_data is None:
        return []
    details = result_data.get("details")
    if not isinstance(details, Mapping):
        return []
    clarifications = details.get("clarifications")
    if not isinstance(clarifications, list):
        return []
    return list(clarifications)


_DELIVERY_RESULT_AUDIT_KEYS = frozenset({EventPayloadKey.PROOF_REFS, "proof_refs"})


def delivery_result_data(result_data: Any) -> dict[str, Any] | None:
    if not isinstance(result_data, dict):
        return None
    return _strip_delivery_result_audit_keys(result_data)


def _strip_delivery_result_audit_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_delivery_result_audit_keys(item)
            for key, item in value.items()
            if key not in _DELIVERY_RESULT_AUDIT_KEYS
        }
    if isinstance(value, list):
        return [_strip_delivery_result_audit_keys(item) for item in value]
    return value


def terminal_result_message(kind: str, details: Mapping[str, Any]) -> str:
    """Project complete terminal prose from typed details, including older records."""
    message = details.get("message")
    if isinstance(message, str) and message.strip():
        return message
    if kind == "no_data":
        return "No matching data was found."
    if kind == "undefined":
        operation = details.get("operation")
        reason = operation.get("reasonCode") if isinstance(operation, Mapping) else None
        return str(reason) if reason else "The requested calculation is undefined."
    if kind == "impossible":
        blocked = [
            str(item.get("requiredFor") or item.get("factRef") or item.get("requestedFactId") or "")
            for item in details.get("blockedRequirements", [])
            if isinstance(item, Mapping)
        ]
        subjects = "; ".join(item for item in blocked if item)
        return f"I cannot answer {subjects} from the available API evidence." if subjects else "I cannot answer that from the available API evidence."
    raise ValueError(f"unsupported factual terminal kind: {kind}")
