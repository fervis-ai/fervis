"""Canonical group-key preference for grouped/ranked operation surfaces."""

from __future__ import annotations

from typing import Any

from fervis.lookup.relation_catalog import identity_payload_is_primary_stable


def prefer_canonical_group_support_sets(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    support_sets_by_output: dict[str, list[dict[str, Any]]] = {}
    for support_set in support_sets:
        answer_output_id = str(support_set.get("answer_output_id") or "")
        support_sets_by_output.setdefault(answer_output_id, []).append(support_set)

    output: list[dict[str, Any]] = []
    for grouped_support_sets in support_sets_by_output.values():
        if any(
            support_set_has_canonical_group(support_set)
            for support_set in grouped_support_sets
        ):
            output.extend(
                support_set
                for support_set in grouped_support_sets
                if not support_set_has_group(support_set)
                or support_set_has_canonical_group(support_set)
            )
            continue
        output.extend(grouped_support_sets)
    return tuple(output)


def prefer_canonical_group_slots(
    slots: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if any(slot_has_canonical_group(slot) for slot in slots):
        return tuple(slot for slot in slots if slot_has_canonical_group(slot))
    return slots


def support_set_has_group(support_set: dict[str, Any]) -> bool:
    return any(
        isinstance(slot, dict) and slot.get("group_key_evidence")
        for slot in support_set.get("fulfillment_slots") or ()
    )


def support_set_has_canonical_group(support_set: dict[str, Any]) -> bool:
    return any(
        isinstance(slot, dict) and slot_has_canonical_group(slot)
        for slot in support_set.get("fulfillment_slots") or ()
    )


def slot_has_canonical_group(slot: dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict) and group_evidence_is_canonical(item)
        for item in slot.get("group_key_evidence") or ()
    )


def group_evidence_is_canonical(item: dict[str, Any]) -> bool:
    return identity_payload_is_primary_stable(item.get("identity"))
