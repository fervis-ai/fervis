"""Backend-projected source fulfillment slots."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.source_binding.evidence_types import (
    evidence_item_can_measure,
)

from ._shared import Any
from fervis.lookup.source_binding.candidates.contracts import (
    EvidenceItem,
    EntityEvidence,
    CandidateKeyEvidence,
    EntityReferenceEvidence,
    FieldEvidence,
    FulfillmentSlot,
    FulfillmentSupportSet,
    RowPopulationEvidence,
    ValueEvidence,
    parse_evidence_item,
)


@dataclass(frozen=True)
class _EvidenceGroup:
    compatibility_basis: str
    metric_items: tuple[FieldEvidence, ...] = ()
    value_items: tuple[FieldEvidence, ...] = ()
    count_basis_items: tuple[RowPopulationEvidence, ...] = ()
    entity_items: tuple[EntityEvidence, ...] = ()


FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE = {
    "GROUP_KEY": ("entity", "value"),
    "ROW_COUNT": ("count_basis", "metric"),
    "MEASURED_VALUE": ("metric",),
    "POPULATION_SCOPE": ("entity", "value"),
    "ANSWER_VALUE": ("entity", "metric", "value"),
}

if set(FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE) != set(
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES
):
    raise ValueError("source-binding fulfillment role dispatch is incomplete")


def _candidate_with_fulfillment_slots(
    candidate: dict[str, Any],
    *,
    requested_facts: tuple[RequestedFact, ...],
    support_field_refs: frozenset[str] | None = None,
) -> dict[str, Any]:
    evidence_items = tuple(
        parse_evidence_item(item)
        for item in candidate.get("evidence_items") or ()
        if isinstance(item, dict) and str(item.get("evidence_id") or "")
    )
    support_evidence_items = _support_evidence_items(
        evidence_items,
        support_field_refs=support_field_refs,
    )
    row_population_path_ids = _candidate_row_population_path_ids(candidate)
    if not requested_facts or not support_evidence_items:
        return candidate
    output = dict(candidate)
    fulfillment_slots = [
        slot
        for fact in requested_facts
        for answer_output in fact.support_answer_outputs
        for group in _answer_output_evidence_item_groups(
            support_evidence_items,
            answer_output_id=answer_output.id,
            answer_output_role=answer_output.role,
            row_population_path_ids=row_population_path_ids,
        )
        for slot in (
            _fulfillment_slot(
                candidate_id=str(output.get("source_candidate_id") or ""),
                answer_output_id=answer_output.id,
                answer_output_role=answer_output.role,
                evidence_group=group,
            ),
        )
    ]
    output["fulfillment_slots"] = [slot.payload() for slot in fulfillment_slots]
    output["fulfillment_support_sets"] = _fulfillment_support_sets(
        candidate_id=str(output.get("source_candidate_id") or ""),
        fulfillment_slots=tuple(fulfillment_slots),
    )
    output["fulfillment_support_sets"] = [
        support_set.payload() for support_set in output["fulfillment_support_sets"]
    ]
    return output


def _support_evidence_items(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    support_field_refs: frozenset[str] | None,
) -> tuple[EvidenceItem, ...]:
    if support_field_refs is None:
        return evidence_items
    return tuple(
        item
        for item in evidence_items
        if item.type in {"candidate_key", "entity_reference", "row_population"}
        or isinstance(item, FieldEvidence)
        and item.field_ref in support_field_refs
    )


def _answer_output_evidence_item_groups(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    answer_output_id: str,
    answer_output_role: str,
    row_population_path_ids: tuple[str, ...],
) -> tuple[_EvidenceGroup, ...]:
    explicitly_scoped = tuple(
        item
        for item in evidence_items
        if isinstance(item, (FieldEvidence, ValueEvidence))
        and answer_output_id in item.answer_output_ids
    )
    row_population_groups = _row_population_count_basis_groups(
        evidence_items,
        row_path_ids=row_population_path_ids,
        compatibility_basis="source_result_grain",
    )
    entity_groups = _entity_evidence_groups(
        evidence_items,
        compatibility_basis="declared_candidate_key",
    )
    if explicitly_scoped:
        groups = (
            *row_population_groups,
            *entity_groups,
            *_evidence_item_groups(
                explicitly_scoped,
                compatibility_basis="explicit_answer_output_metadata",
            ),
        )
    elif any(
        item.answer_output_ids
        for item in evidence_items
        if isinstance(item, (FieldEvidence, ValueEvidence))
    ):
        groups = (*row_population_groups, *entity_groups)
    else:
        groups = _open_candidate_evidence_item_groups(
            evidence_items,
            row_population_path_ids=row_population_path_ids,
            compatibility_basis="open_candidate_field",
        )
    return tuple(
        group
        for group in groups
        if _evidence_group_matches_answer_output_role(
            group,
            answer_output_role=answer_output_role,
        )
    )


def _evidence_group_matches_answer_output_role(
    group: _EvidenceGroup,
    *,
    answer_output_role: str,
) -> bool:
    if not answer_output_role:
        return True
    allowed_kinds = _allowed_evidence_group_kinds(answer_output_role)
    if allowed_kinds is None:
        raise ValueError("unsupported answer output support role")
    allowed_kind_set = set(allowed_kinds)
    group_kind_set = _evidence_group_kinds(group)
    evidence_group_has_allowed_kind = bool(allowed_kind_set & group_kind_set)
    return evidence_group_has_allowed_kind


def _allowed_evidence_group_kinds(
    answer_output_role: str,
) -> tuple[str, ...] | None:
    return FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE.get(answer_output_role)


def _evidence_group_kinds(group: _EvidenceGroup) -> set[str]:
    kinds: set[str] = set()
    if group.entity_items:
        kinds.add("entity")
    if group.count_basis_items:
        kinds.add("count_basis")
    if group.metric_items:
        kinds.add("metric")
    if group.value_items:
        kinds.add("value")
    return kinds


def _candidate_row_population_path_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(grain.get("row_path_id") or "")
            for grain in candidate.get("result_grains") or ()
            if isinstance(grain, dict)
            and str(grain.get("cardinality") or "").lower() == "many"
            and str(grain.get("row_path_id") or "")
        )
    )


def _open_candidate_evidence_item_groups(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    row_population_path_ids: tuple[str, ...],
    compatibility_basis: str,
) -> tuple[_EvidenceGroup, ...]:
    return (
        *_row_population_count_basis_groups(
            evidence_items,
            row_path_ids=row_population_path_ids,
            compatibility_basis=compatibility_basis,
        ),
        *_evidence_item_groups(
            evidence_items,
            compatibility_basis=compatibility_basis,
        ),
    )


def _row_population_count_basis_groups(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    row_path_ids: tuple[str, ...],
    compatibility_basis: str,
) -> tuple[_EvidenceGroup, ...]:
    return tuple(
        _EvidenceGroup(
            count_basis_items=(item,),
            compatibility_basis=compatibility_basis,
        )
        for row_path_id in row_path_ids
        for item in _row_population_count_basis_items(
            evidence_items,
            row_path_id=row_path_id,
        )
    )


def _row_population_count_basis_items(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    row_path_id: str,
) -> tuple[RowPopulationEvidence, ...]:
    return tuple(
        item
        for item in evidence_items
        if isinstance(item, RowPopulationEvidence) and item.row_path_id == row_path_id
    )


def _evidence_item_groups(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    compatibility_basis: str,
) -> tuple[_EvidenceGroup, ...]:
    entity_groups = _entity_evidence_groups(
        evidence_items,
        compatibility_basis=compatibility_basis,
    )
    metric_groups = tuple(
        _EvidenceGroup(
            metric_items=(item,),
            compatibility_basis=compatibility_basis,
        )
        for item in evidence_items
        if isinstance(item, FieldEvidence) and evidence_item_can_measure(item)
    )
    value_groups = tuple(
        _EvidenceGroup(
            value_items=(item,),
            compatibility_basis=compatibility_basis,
        )
        for item in evidence_items
        if isinstance(item, FieldEvidence) and _evidence_item_can_be_direct_value(item)
    )
    return (*entity_groups, *metric_groups, *value_groups)


def _evidence_item_can_be_direct_value(item: EvidenceItem) -> bool:
    if not isinstance(item, FieldEvidence):
        return False
    if item.presentation_only:
        return False
    if item.entity_evidence_member:
        return False
    if evidence_item_can_measure(item):
        return False
    if not item.field_id:
        return False
    return item.type.lower() not in {
        "any",
        "array",
        "json",
        "list",
        "object",
        "row_population",
    }


def _entity_evidence_groups(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    compatibility_basis: str,
) -> tuple[_EvidenceGroup, ...]:
    return tuple(
        _EvidenceGroup(
            entity_items=(item,),
            compatibility_basis=compatibility_basis,
        )
        for item in evidence_items
        if isinstance(item, (CandidateKeyEvidence, EntityReferenceEvidence))
    )


def _fulfillment_slot(
    *,
    candidate_id: str,
    answer_output_id: str,
    answer_output_role: str,
    evidence_group: _EvidenceGroup,
) -> FulfillmentSlot:
    evidence_ids = tuple(
        dict.fromkeys(
            evidence_item.evidence_id
            for evidence_item in (
                *evidence_group.metric_items,
                *evidence_group.value_items,
                *evidence_group.count_basis_items,
                *evidence_group.entity_items,
            )
            if evidence_item.evidence_id
        )
    )
    return FulfillmentSlot(
        fulfillment_slot_id=_fulfillment_slot_id(
            candidate_id=candidate_id,
            answer_output_id=answer_output_id,
            evidence_ids=evidence_ids,
            role_key=_fulfillment_slot_role_key(evidence_group),
        ),
        answer_output_id=answer_output_id,
        compatibility_basis=evidence_group.compatibility_basis,
        answer_output_role=answer_output_role,
        metric_measure_evidence=evidence_group.metric_items,
        value_evidence=evidence_group.value_items,
        row_count_basis_evidence=evidence_group.count_basis_items,
        entity_evidence=evidence_group.entity_items,
    )


def _fulfillment_slot_role_key(evidence_group: _EvidenceGroup) -> str:
    if evidence_group.metric_items:
        return "metric"
    if evidence_group.value_items:
        return "value"
    if evidence_group.count_basis_items:
        return "count"
    if evidence_group.entity_items:
        return "entity"
    return "support"


def _fulfillment_slot_id(
    *,
    candidate_id: str,
    answer_output_id: str,
    evidence_ids: tuple[str, ...],
    role_key: str,
) -> str:
    evidence_key = "__".join(evidence_ids)
    return f"slot.{candidate_id}.{answer_output_id}.{role_key}.{evidence_key}"


def _fulfillment_support_sets(
    *,
    candidate_id: str,
    fulfillment_slots: tuple[FulfillmentSlot, ...],
) -> list[FulfillmentSupportSet]:
    slots_by_output: dict[str, list[FulfillmentSlot]] = {}
    for slot in fulfillment_slots:
        answer_output_id = slot.answer_output_id
        if not answer_output_id:
            continue
        slots_by_output.setdefault(answer_output_id, []).append(slot)
    output: list[FulfillmentSupportSet] = []
    for answer_output_id, slots in slots_by_output.items():
        if not slots:
            continue
        output.extend(
            _role_aware_fulfillment_support_sets(
                candidate_id=candidate_id,
                answer_output_id=answer_output_id,
                slots=tuple(slots),
            )
        )
    return output


def _role_aware_fulfillment_support_sets(
    *,
    candidate_id: str,
    answer_output_id: str,
    slots: tuple[FulfillmentSlot, ...],
) -> tuple[FulfillmentSupportSet, ...]:
    return tuple(
        _fulfillment_support_set(
            candidate_id=candidate_id,
            answer_output_id=answer_output_id,
            slots=(slot,),
        )
        for slot in slots
        if _support_slot_has_evidence(slot)
    )


def _support_slot_has_evidence(slot: FulfillmentSlot) -> bool:
    return bool(
        slot.metric_measure_evidence
        or slot.value_evidence
        or slot.row_count_basis_evidence
        or slot.entity_evidence
    )


def _fulfillment_support_set(
    *,
    candidate_id: str,
    answer_output_id: str,
    slots: tuple[FulfillmentSlot, ...],
) -> FulfillmentSupportSet:
    support_set_id = _fulfillment_support_set_id(
        candidate_id=candidate_id,
        answer_output_id=answer_output_id,
        slot_ids=tuple(
            slot.fulfillment_slot_id for slot in slots if slot.fulfillment_slot_id
        ),
    )
    return FulfillmentSupportSet(
        fulfillment_support_set_id=support_set_id,
        answer_output_id=answer_output_id,
        fulfillment_slots=slots,
    )


def _fulfillment_support_set_id(
    *,
    candidate_id: str,
    answer_output_id: str,
    slot_ids: tuple[str, ...],
) -> str:
    slot_key = "__".join(slot_ids)
    return f"support.{candidate_id}.{answer_output_id}.{slot_key}"
