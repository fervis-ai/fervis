"""Plan-selection support grouping for grouped aggregate operations."""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Callable
from itertools import product
from fervis.lookup.fact_plan.row_sources.model import (
    row_source_value_type,
    row_source_value_type_is_scalar,
)
from fervis.lookup.source_binding.candidates.contracts import (
    EvidenceItem,
    FulfillmentSlot,
    FulfillmentSupportSet,
    evidence_row_path_id,
)


def grouped_aggregate_support_set_groups(
    support_sets: tuple[FulfillmentSupportSet, ...],
    *,
    requirement_id: str,
    required_answer_output_ids: tuple[str, ...],
    source_candidate_id: str,
) -> tuple[tuple[FulfillmentSupportSet, ...], ...]:
    if requirement_id != "operation":
        return ()
    raw_support_sets = tuple(
        support_set
        for support_set in support_sets
        if support_set.answer_output_id in set(required_answer_output_ids)
    )
    complete_support_sets = _complete_aggregate_support_sets(raw_support_sets)
    if _support_sets_cover_required_outputs(
        complete_support_sets,
        required_answer_output_ids=required_answer_output_ids,
    ) and _support_sets_are_aggregate_complete(complete_support_sets):
        return (complete_support_sets,)
    aggregate_support_sets = _unique_support_sets(
        [
            *_same_output_aggregate_support_sets(
                candidate_id=source_candidate_id,
                support_sets=raw_support_sets,
            ),
            *_multi_output_aggregate_support_sets(
                candidate_id=source_candidate_id,
                support_sets=raw_support_sets,
                required_answer_output_ids=required_answer_output_ids,
            ),
        ]
    )
    selected_groups: list[tuple[FulfillmentSupportSet, ...]] = []
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
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    support_sets_by_output: dict[str, list[FulfillmentSupportSet]] = {}
    for support_set in support_sets:
        answer_output_id = support_set.answer_output_id
        if answer_output_id:
            support_sets_by_output.setdefault(answer_output_id, []).append(support_set)
    output: list[FulfillmentSupportSet] = []
    for answer_output_id, answer_support_sets in support_sets_by_output.items():
        slots = _support_set_slots(answer_support_sets)
        group_slots = tuple(slot for slot in slots if _slot_has_group_role(slot))
        metric_slots = tuple(slot for slot in slots if _slot_has_metric_role(slot))
        output.extend(
            FulfillmentSupportSet(
                fulfillment_support_set_id=_operation_bundle_id(
                    candidate_id=candidate_id,
                    answer_output_id=answer_output_id,
                    slots=(group_slot, metric_slot),
                ),
                answer_output_id=answer_output_id,
                fulfillment_slots=(group_slot, metric_slot),
            )
            for group_slot in group_slots
            for metric_slot in metric_slots
            if _aggregate_slots_are_complete((group_slot, metric_slot))
        )
    return tuple(output)


def _multi_output_aggregate_support_sets(
    *,
    candidate_id: str,
    support_sets: tuple[FulfillmentSupportSet, ...],
    required_answer_output_ids: tuple[str, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    if len(required_answer_output_ids) <= 1:
        return ()
    required = set(required_answer_output_ids)
    relevant_support_sets = tuple(
        support_set
        for support_set in support_sets
        if support_set.answer_output_id in required
    )
    group_slot_sets = _group_axis_slot_sets(relevant_support_sets)
    metric_slot_sets = _metric_slot_sets(relevant_support_sets)
    return tuple(
        FulfillmentSupportSet(
            fulfillment_support_set_id=_operation_bundle_id(
                candidate_id=candidate_id,
                answer_output_id=answer_output_id,
                slots=(*group_slots, *metric_slots),
            ),
            answer_output_id=answer_output_id,
            fulfillment_slots=(*group_slots, *metric_slots),
        )
        for group_slots in group_slot_sets
        for metric_slots in metric_slot_sets
        if _aggregate_slots_are_complete([*group_slots, *metric_slots])
        for answer_output_id in required_answer_output_ids
    )


def _group_axis_slot_sets(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[tuple[FulfillmentSlot, ...], ...]:
    group_slots_by_output = _slots_by_answer_output(
        support_sets,
        slot_predicate=_slot_has_group_role,
    )
    if not group_slots_by_output:
        return ()
    slot_options = tuple(tuple(slots) for _, slots in group_slots_by_output.items())
    if any(not slots for slots in slot_options):
        return ()
    return tuple(tuple(slots) for slots in product(*slot_options))


def _metric_slot_sets(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[tuple[FulfillmentSlot, ...], ...]:
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
    support_sets: tuple[FulfillmentSupportSet, ...],
    *,
    slot_predicate: Callable[[FulfillmentSlot], bool],
) -> dict[str, list[FulfillmentSlot]]:
    output: dict[str, list[FulfillmentSlot]] = {}
    seen_by_output: dict[str, set[str]] = {}
    for support_set in support_sets:
        answer_output_id = support_set.answer_output_id
        if not answer_output_id:
            continue
        for slot in support_set.fulfillment_slots:
            if not slot_predicate(slot):
                continue
            slot_id = slot.fulfillment_slot_id
            if not slot_id:
                continue
            seen = seen_by_output.setdefault(answer_output_id, set())
            if slot_id in seen:
                continue
            seen.add(slot_id)
            output.setdefault(answer_output_id, []).append(slot)
    return output


def _complete_aggregate_support_sets(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    return tuple(
        support_set
        for support_set in support_sets
        if _support_sets_are_aggregate_complete((support_set,))
    )


def _aggregate_operation_support_set_groups(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[tuple[FulfillmentSupportSet, ...], ...]:
    groups: dict[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        list[FulfillmentSupportSet],
    ] = {}
    for support_set in support_sets:
        key = _aggregate_operation_key(support_set)
        if key is None:
            continue
        groups.setdefault(key, []).append(support_set)
    return tuple(tuple(items) for items in groups.values())


def _aggregate_operation_key(
    support_set: FulfillmentSupportSet,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    slots = support_set.fulfillment_slots
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
            item.evidence_id
            for slot in slots
            for item in _group_evidence(slot)
            if item.evidence_id
        )
    )
    metric_evidence_ids = tuple(
        sorted(
            item.evidence_id
            for slot in slots
            for item in _metric_evidence(slot)
            if item.evidence_id
        )
    )
    if not group_evidence_ids or not metric_evidence_ids:
        return None
    return (tuple(sorted(row_paths)), group_evidence_ids, metric_evidence_ids)


def _support_sets_are_aggregate_complete(
    support_sets: tuple[FulfillmentSupportSet, ...],
) -> bool:
    return _aggregate_slots_are_complete(_support_set_slots(list(support_sets)))


def _aggregate_slots_are_complete(
    slots: Sequence[FulfillmentSlot],
) -> bool:
    return (
        _slots_include_group_evidence(slots)
        and _slots_include_metric_evidence(slots)
        and _group_evidence_is_executable(slots)
        and _aggregate_slots_have_distinct_group_and_metric(slots)
        and _aggregate_slots_share_row_path(slots)
    )


def _support_sets_cover_required_outputs(
    support_sets: tuple[FulfillmentSupportSet, ...],
    *,
    required_answer_output_ids: tuple[str, ...],
) -> bool:
    return set(required_answer_output_ids) <= {
        support_set.answer_output_id for support_set in support_sets
    }


def _unique_support_sets(
    support_sets: list[FulfillmentSupportSet],
) -> tuple[FulfillmentSupportSet, ...]:
    output: list[FulfillmentSupportSet] = []
    seen: set[str] = set()
    for support_set in support_sets:
        support_set_id = support_set.fulfillment_support_set_id
        if not support_set_id or support_set_id in seen:
            continue
        seen.add(support_set_id)
        output.append(support_set)
    return tuple(output)


def _raw_support_sets_for_aggregate_slots(
    raw_support_sets: tuple[FulfillmentSupportSet, ...],
    *,
    aggregate_support_sets: tuple[FulfillmentSupportSet, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    selected_slot_ids = {
        slot.fulfillment_slot_id
        for support_set in aggregate_support_sets
        for slot in support_set.fulfillment_slots
        if slot.fulfillment_slot_id
    }
    return tuple(
        support_set
        for support_set in raw_support_sets
        if any(
            slot.fulfillment_slot_id in selected_slot_ids
            for slot in support_set.fulfillment_slots
        )
    )


def _support_set_slots(
    support_sets: Sequence[FulfillmentSupportSet],
) -> list[FulfillmentSlot]:
    slots: list[FulfillmentSlot] = []
    seen: set[str] = set()
    for support_set in support_sets:
        for slot in support_set.fulfillment_slots:
            slot_id = slot.fulfillment_slot_id
            if not slot_id or slot_id in seen:
                continue
            slots.append(slot)
            seen.add(slot_id)
    return slots


def _slot_has_metric_role(slot: FulfillmentSlot) -> bool:
    return bool(slot.metric_measure_evidence or slot.row_count_basis_evidence)


def _slot_has_group_role(slot: FulfillmentSlot) -> bool:
    return bool(slot.entity_evidence or slot.value_evidence)


def _slots_include_group_evidence(slots: Sequence[FulfillmentSlot]) -> bool:
    return any(_slot_has_group_role(slot) for slot in slots)


def _slots_include_metric_evidence(slots: Sequence[FulfillmentSlot]) -> bool:
    return any(_slot_has_metric_role(slot) for slot in slots)


def _group_evidence_is_executable(slots: Sequence[FulfillmentSlot]) -> bool:
    entity_evidence = tuple(
        item
        for slot in slots
        for item in slot.entity_evidence
    )
    value_evidence = tuple(
        item
        for slot in slots
        for item in slot.value_evidence
    )
    return all(bool(item.components) for item in entity_evidence) and all(
        bool(item.field_id)
        and row_source_value_type_is_scalar(
            row_source_value_type(item.type)
        )
        for item in value_evidence
    )


def _aggregate_slots_have_distinct_group_and_metric(
    slots: Sequence[FulfillmentSlot],
) -> bool:
    group_evidence_ids = {
        evidence_id
        for slot in slots
        for item in _group_evidence(slot)
        for evidence_id in (item.evidence_id,)
        if evidence_id
    }
    metric_evidence_ids = {
        evidence_id
        for slot in slots
        for item in _metric_evidence(slot)
        for evidence_id in (item.evidence_id,)
        if evidence_id
    }
    return bool(metric_evidence_ids and (group_evidence_ids - metric_evidence_ids))


def _aggregate_slots_share_row_path(slots: Sequence[FulfillmentSlot]) -> bool:
    row_path_ids: set[str] = set()
    for slot in slots:
        for item in _aggregate_evidence_items(slot):
            row_path_id = _evidence_row_path_id(item)
            if not row_path_id:
                return False
            row_path_ids.add(row_path_id)
    return len(row_path_ids) <= 1


def _aggregate_evidence_items(slot: FulfillmentSlot) -> tuple[EvidenceItem, ...]:
    return (
        *slot.metric_measure_evidence,
        *slot.row_count_basis_evidence,
        *slot.entity_evidence,
        *slot.value_evidence,
    )


def _group_evidence(slot: FulfillmentSlot) -> tuple[EvidenceItem, ...]:
    return (*slot.entity_evidence, *slot.value_evidence)


def _metric_evidence(slot: FulfillmentSlot) -> tuple[EvidenceItem, ...]:
    return (*slot.metric_measure_evidence, *slot.row_count_basis_evidence)


def _evidence_row_path_id(item: EvidenceItem) -> str:
    return evidence_row_path_id(item)


def _slot_evidence_ids(slot: FulfillmentSlot) -> set[str]:
    return {
        item.evidence_id
        for item in _aggregate_evidence_items(slot)
        if item.evidence_id
    }


def _operation_bundle_id(
    *,
    candidate_id: str,
    answer_output_id: str,
    slots: tuple[FulfillmentSlot, ...],
) -> str:
    slot_key = "__".join(
        slot.fulfillment_slot_id
        for slot in slots
        if slot.fulfillment_slot_id
    )
    return f"support.{candidate_id}.{answer_output_id}.operation.{slot_key}"
