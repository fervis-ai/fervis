"""Plan-selection support grouping for grouped/ranked operations."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Any

from fervis.lookup.operation_families.grouped_ranked.canonical_groups import (
    prefer_canonical_group_slots,
    prefer_canonical_group_support_sets,
)


def grouped_ranked_support_set_groups(
    support_sets: tuple[dict[str, Any], ...],
    *,
    requirement_id: str,
    required_answer_output_ids: tuple[str, ...],
    source_candidate_id: str,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if requirement_id != "operation":
        return ()
    raw_support_sets = tuple(
        support_set
        for support_set in support_sets
        if str(support_set.get("answer_output_id") or "")
        in set(required_answer_output_ids)
    )
    complete_support_sets = prefer_canonical_group_support_sets(
        _complete_aggregate_support_sets(raw_support_sets)
    )
    if _support_sets_cover_required_outputs(
        complete_support_sets,
        required_answer_output_ids=required_answer_output_ids,
    ) and _support_sets_are_aggregate_complete(complete_support_sets):
        return (complete_support_sets,)
    aggregate_support_sets = prefer_canonical_group_support_sets(
        _unique_support_sets(
            [
                *(
                    _same_output_aggregate_support_sets(
                        candidate_id=source_candidate_id,
                        support_sets=raw_support_sets,
                    )
                ),
                *(
                    _multi_output_aggregate_support_sets(
                        candidate_id=source_candidate_id,
                        support_sets=raw_support_sets,
                        required_answer_output_ids=required_answer_output_ids,
                    )
                ),
            ]
        )
    )
    selected_groups: list[tuple[dict[str, Any], ...]] = []
    for operation_support_sets in _aggregate_operation_support_set_groups(
        aggregate_support_sets
    ):
        if not _support_sets_cover_required_outputs(
            operation_support_sets,
            required_answer_output_ids=required_answer_output_ids,
        ):
            continue
        if not _support_sets_are_aggregate_complete(operation_support_sets):
            continue
        selected_raw_support_sets = _raw_support_sets_for_aggregate_slots(
            raw_support_sets,
            aggregate_support_sets=operation_support_sets,
        )
        if selected_raw_support_sets:
            selected_groups.append(selected_raw_support_sets)
    return tuple(selected_groups)


def _same_output_aggregate_support_sets(
    *,
    candidate_id: str,
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    support_sets_by_output: dict[str, list[dict[str, Any]]] = {}
    for support_set in support_sets:
        answer_output_id = str(support_set.get("answer_output_id") or "")
        if answer_output_id:
            support_sets_by_output.setdefault(answer_output_id, []).append(support_set)
    output: list[dict[str, Any]] = []
    for answer_output_id, answer_support_sets in support_sets_by_output.items():
        slots = _support_set_slots(answer_support_sets)
        group_slots = prefer_canonical_group_slots(
            tuple(slot for slot in slots if _slot_has_group_role(slot))
        )
        metric_slots = tuple(slot for slot in slots if _slot_has_metric_role(slot))
        output.extend(
            {
                "fulfillment_support_set_id": _operation_bundle_id(
                    candidate_id=candidate_id,
                    answer_output_id=answer_output_id,
                    slots=(group_slot, metric_slot),
                ),
                "answer_output_id": answer_output_id,
                "fulfillment_slots": [group_slot, metric_slot],
            }
            for group_slot in group_slots
            for metric_slot in metric_slots
            if _aggregate_slots_are_complete((group_slot, metric_slot))
        )
    return tuple(output)


def _multi_output_aggregate_support_sets(
    *,
    candidate_id: str,
    support_sets: tuple[dict[str, Any], ...],
    required_answer_output_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    if len(required_answer_output_ids) <= 1:
        return ()
    required = set(required_answer_output_ids)
    relevant_support_sets = tuple(
        support_set
        for support_set in support_sets
        if str(support_set.get("answer_output_id") or "") in required
    )
    group_slot_sets = _group_axis_slot_sets(relevant_support_sets)
    metric_slot_sets = _metric_slot_sets(relevant_support_sets)
    return tuple(
        {
            "fulfillment_support_set_id": _operation_bundle_id(
                candidate_id=candidate_id,
                answer_output_id=answer_output_id,
                slots=(*group_slots, *metric_slots),
            ),
            "answer_output_id": answer_output_id,
            "fulfillment_slots": [*group_slots, *metric_slots],
        }
        for group_slots in group_slot_sets
        for metric_slots in metric_slot_sets
        if _aggregate_slots_are_complete([*group_slots, *metric_slots])
        for answer_output_id in required_answer_output_ids
    )


def _group_axis_slot_sets(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    group_slots_by_output = _slots_by_answer_output(
        support_sets,
        slot_predicate=_slot_has_group_role,
    )
    if not group_slots_by_output:
        return ()
    slot_options = tuple(
        prefer_canonical_group_slots(tuple(slots))
        for _, slots in group_slots_by_output.items()
    )
    if any(not slots for slots in slot_options):
        return ()
    return tuple(tuple(slots) for slots in product(*slot_options))


def _metric_slot_sets(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    metric_slots_by_output = _slots_by_answer_output(
        support_sets,
        slot_predicate=_slot_has_metric_role,
    )
    if not metric_slots_by_output:
        return ()
    slot_options = tuple(tuple(slots) for _, slots in metric_slots_by_output.items())
    if any(not slots for slots in slot_options):
        return ()
    return tuple(tuple(slots) for slots in product(*slot_options))


def _slots_by_answer_output(
    support_sets: tuple[dict[str, Any], ...],
    *,
    slot_predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    seen_by_output: dict[str, set[str]] = {}
    for support_set in support_sets:
        answer_output_id = str(support_set.get("answer_output_id") or "")
        if not answer_output_id:
            continue
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict) or not slot_predicate(slot):
                continue
            slot_id = str(slot.get("fulfillment_slot_id") or "")
            if not slot_id:
                continue
            seen = seen_by_output.setdefault(answer_output_id, set())
            if slot_id in seen:
                continue
            seen.add(slot_id)
            output.setdefault(answer_output_id, []).append(slot)
    return output


def _complete_aggregate_support_sets(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        support_set
        for support_set in support_sets
        if _support_sets_are_aggregate_complete((support_set,))
    )


def _aggregate_operation_support_set_groups(
    support_sets: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    groups: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        list[dict[str, Any]],
    ] = {}
    for support_set in support_sets:
        key = _aggregate_operation_key(support_set)
        if key is None:
            continue
        groups.setdefault(key, []).append(support_set)
    return tuple(tuple(items) for items in groups.values())


def _aggregate_operation_key(
    support_set: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    slots = [
        slot
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
    ]
    row_paths = {
        _evidence_row_path_id(item)
        for slot in slots
        for item in _aggregate_evidence_items(slot)
        if _evidence_row_path_id(item)
    }
    if len(row_paths) != 1:
        return None
    group_evidence_ids = tuple(
        sorted(
            str(item.get("evidence_id") or "")
            for slot in slots
            for item in slot.get("group_key_evidence") or ()
            if isinstance(item, dict) and str(item.get("evidence_id") or "")
        )
    )
    metric_evidence_ids = tuple(
        sorted(
            str(item.get("evidence_id") or "")
            for slot in slots
            for key in ("metric_measure_evidence", "row_count_basis_evidence")
            for item in slot.get(key) or ()
            if isinstance(item, dict) and str(item.get("evidence_id") or "")
        )
    )
    if not group_evidence_ids or not metric_evidence_ids:
        return None
    return (tuple(sorted(row_paths)), group_evidence_ids, metric_evidence_ids)


def _support_sets_are_aggregate_complete(
    support_sets: tuple[dict[str, Any], ...],
) -> bool:
    return _aggregate_slots_are_complete(_support_set_slots(list(support_sets)))


def _aggregate_slots_are_complete(
    slots: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> bool:
    return (
        _slots_include_group_evidence(slots)
        and _slots_include_metric_evidence(slots)
        and _group_evidence_is_executable(slots)
        and _aggregate_slots_have_distinct_group_and_metric(slots)
        and _aggregate_slots_share_row_path(slots)
    )


def _support_sets_cover_required_outputs(
    support_sets: tuple[dict[str, Any], ...],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> bool:
    return set(required_answer_output_ids) <= {
        str(support_set.get("answer_output_id") or "") for support_set in support_sets
    }


def _unique_support_sets(
    support_sets: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for support_set in support_sets:
        support_set_id = str(support_set.get("fulfillment_support_set_id") or "")
        if not support_set_id or support_set_id in seen:
            continue
        seen.add(support_set_id)
        output.append(support_set)
    return tuple(output)


def _raw_support_sets_for_aggregate_slots(
    raw_support_sets: tuple[dict[str, Any], ...],
    *,
    aggregate_support_sets: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    selected_slot_ids = {
        str(slot.get("fulfillment_slot_id") or "")
        for support_set in aggregate_support_sets
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict) and str(slot.get("fulfillment_slot_id") or "")
    }
    return tuple(
        support_set
        for support_set in raw_support_sets
        if any(
            isinstance(slot, dict)
            and str(slot.get("fulfillment_slot_id") or "") in selected_slot_ids
            for slot in support_set.get("fulfillment_slots") or ()
        )
    )


def _support_set_slots(
    support_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for support_set in support_sets:
        for slot in support_set.get("fulfillment_slots") or ():
            if not isinstance(slot, dict):
                continue
            slot_id = str(slot.get("fulfillment_slot_id") or "")
            if not slot_id or slot_id in seen:
                continue
            slots.append(slot)
            seen.add(slot_id)
    return slots


def _slot_has_metric_role(slot: dict[str, Any]) -> bool:
    return bool(
        slot.get("metric_measure_evidence") or slot.get("row_count_basis_evidence")
    )


def _slot_has_group_role(slot: dict[str, Any]) -> bool:
    return bool(slot.get("group_key_evidence"))


def _slots_include_group_evidence(slots: list[dict[str, Any]]) -> bool:
    return any(_slot_has_group_role(slot) for slot in slots)


def _slots_include_metric_evidence(slots: list[dict[str, Any]]) -> bool:
    return any(_slot_has_metric_role(slot) for slot in slots)


def _group_evidence_is_executable(slots: list[dict[str, Any]]) -> bool:
    return all(
        str(item.get("type") or "").lower()
        not in {"any", "json", "object", "array", "list", "row_population"}
        for slot in slots
        for item in slot.get("group_key_evidence") or ()
        if isinstance(item, dict)
    )


def _aggregate_slots_have_distinct_group_and_metric(
    slots: list[dict[str, Any]],
) -> bool:
    group_evidence_ids = {
        evidence_id
        for slot in slots
        for item in slot.get("group_key_evidence") or ()
        if isinstance(item, dict)
        for evidence_id in (str(item.get("evidence_id") or ""),)
        if evidence_id
    }
    metric_evidence_ids = {
        evidence_id
        for slot in slots
        for key in ("metric_measure_evidence", "row_count_basis_evidence")
        for item in slot.get(key) or ()
        if isinstance(item, dict)
        for evidence_id in (str(item.get("evidence_id") or ""),)
        if evidence_id
    }
    return bool(metric_evidence_ids and (group_evidence_ids - metric_evidence_ids))


def _aggregate_slots_share_row_path(slots: list[dict[str, Any]]) -> bool:
    row_path_ids: set[str] = set()
    for slot in slots:
        for item in _aggregate_evidence_items(slot):
            row_path_id = _evidence_row_path_id(item)
            if not row_path_id:
                return False
            row_path_ids.add(row_path_id)
    return len(row_path_ids) <= 1


def _aggregate_evidence_items(slot: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "group_key_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict)
    )


def _evidence_row_path_id(item: dict[str, Any]) -> str:
    return str(item.get("row_path_id") or "").strip()


def _slot_evidence_ids(slot: dict[str, Any]) -> set[str]:
    return {
        str(item.get("evidence_id") or "")
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "group_key_evidence",
            "scope_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict) and str(item.get("evidence_id") or "")
    }


def _operation_bundle_id(
    *,
    candidate_id: str,
    answer_output_id: str,
    slots: tuple[dict[str, Any], ...],
) -> str:
    slot_key = "__".join(
        str(slot.get("fulfillment_slot_id") or "")
        for slot in slots
        if str(slot.get("fulfillment_slot_id") or "")
    )
    return f"support.{candidate_id}.{answer_output_id}.operation.{slot_key}"
