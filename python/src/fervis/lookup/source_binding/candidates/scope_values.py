"""Backend-authored scope values for source-binding candidate coverage."""

import json

from ._shared import Any


def memory_relation_scope_value_segments(
    relation: dict[str, Any],
) -> tuple[frozenset[str], ...]:
    raw = (
        (relation.get("completeness") or {}).get("scopeFingerprint")
        if isinstance(relation.get("completeness"), dict)
        else ""
    )
    output: list[frozenset[str]] = []
    for segment in str(raw or "").split("|"):
        try:
            decoded = json.loads(segment)
        except json.JSONDecodeError:
            continue
        endpoint_args = (
            decoded.get("endpointArgs") if isinstance(decoded, dict) else None
        )
        if not isinstance(endpoint_args, dict):
            continue
        values: set[str] = set()
        for value in endpoint_args.values():
            if isinstance(value, (dict, list)) or value in ("", None):
                continue
            values.add(str(value))
        if values:
            output.append(frozenset(values))
    return tuple(output)


def scope_values_include_time(
    scope_values: frozenset[str],
    *,
    value: object,
) -> bool:
    payload = getattr(value, "payload", None)
    resolved_start = str(getattr(payload, "resolved_start", "") or "")
    resolved_end = str(getattr(payload, "resolved_end", "") or "")
    if not resolved_start or not resolved_end:
        return False
    return resolved_start in scope_values and resolved_end in scope_values
