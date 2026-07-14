"""Plan selection support options projected from source candidates."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.lookup.source_binding.candidates.contracts import (
    EvidenceItem,
    FulfillmentSlot,
    FulfillmentSupportSet,
    evidence_field_ids,
)
from fervis.lookup.source_binding.candidates.model import SourceCandidate


@dataclass(frozen=True)
class PlanSelectionSupportOption:
    support_set_id: str
    support_roles: frozenset[str]
    binding_support_set_id: str = ""
    answer_output_id: str = ""
    support_refs_by_role: tuple[tuple[str, tuple[str, ...]], ...] = ()
    field_ids: tuple[str, ...] = ()


def plan_selection_support_options(
    candidate: SourceCandidate,
) -> tuple[PlanSelectionSupportOption, ...]:
    """Return the plan selection-selectable support options for a candidate."""

    intrinsic_options = tuple(
        option
        for option in (
            _value_support_option(candidate),
            _calendar_support_option(candidate),
        )
        if option is not None
    )
    fulfillment_options = tuple(
        option
        for index, support_set in enumerate(candidate.fulfillment_support_sets, start=1)
        for option in (_fulfillment_support_option(support_set, index=index),)
        if option is not None
    )
    return (*intrinsic_options, *fulfillment_options)


def plan_selection_fulfillment_support_sets(
    candidate: SourceCandidate,
) -> tuple[FulfillmentSupportSet, ...]:
    return candidate.fulfillment_support_sets


def _fulfillment_support_option(
    support_set: FulfillmentSupportSet,
    *,
    index: int,
) -> PlanSelectionSupportOption | None:
    binding_support_set_id = _support_set_binding_id(support_set)
    if not binding_support_set_id:
        return None
    support_roles: list[str] = []
    support_refs_by_role: dict[str, list[str]] = {}
    field_ids: list[str] = []
    for slot in support_set.fulfillment_slots:
        for role in _slot_support_roles(slot):
            if role not in support_roles:
                support_roles.append(role)
        for role, refs in _slot_support_refs_by_role(slot).items():
            support_refs_by_role.setdefault(role, [])
            support_refs_by_role[role].extend(
                ref for ref in refs if ref not in support_refs_by_role[role]
            )
        for field_id in _slot_field_ids(slot):
            if field_id not in field_ids:
                field_ids.append(field_id)
    return PlanSelectionSupportOption(
        support_set_id=f"support_set_{index}",
        binding_support_set_id=binding_support_set_id,
        support_roles=frozenset(support_roles),
        answer_output_id=support_set.answer_output_id,
        support_refs_by_role=tuple(
            (role, tuple(refs)) for role, refs in support_refs_by_role.items() if refs
        ),
        field_ids=tuple(field_ids),
    )


def _value_support_option(
    candidate: SourceCandidate,
) -> PlanSelectionSupportOption | None:
    if candidate.kind != "value":
        return None
    value_id = candidate.value_id or candidate.id
    if not value_id:
        return None
    return PlanSelectionSupportOption(
        support_set_id=f"support.{value_id}.value",
        support_roles=frozenset(("VALUE_SOURCE",)),
        support_refs_by_role=(("VALUE_SOURCE", (value_id,)),),
    )


def _calendar_support_option(
    candidate: SourceCandidate,
) -> PlanSelectionSupportOption | None:
    calendar_id = candidate.calendar_id
    source_candidate_id = candidate.id or calendar_id
    if not calendar_id or not source_candidate_id:
        return None
    return PlanSelectionSupportOption(
        support_set_id=f"support.{source_candidate_id}.calendar",
        support_roles=frozenset(("CALENDAR_SOURCE",)),
        support_refs_by_role=(("CALENDAR_SOURCE", (calendar_id,)),),
    )


def _support_set_binding_id(support_set: FulfillmentSupportSet) -> str:
    return support_set.fulfillment_support_set_id or support_set.fulfillment_choice_id


def _slot_support_roles(slot: FulfillmentSlot) -> tuple[str, ...]:
    role = slot.answer_output_role
    if role:
        return (role,)
    inferred_roles = tuple(
        evidence_role
        for evidence, evidence_role in (
            (slot.metric_measure_evidence, "MEASURED_VALUE"),
            (slot.value_evidence, "ANSWER_VALUE"),
            (slot.row_count_basis_evidence, "ROW_COUNT"),
            (slot.entity_evidence, "GROUP_KEY"),
        )
        if evidence
    )
    if len(inferred_roles) != 1:
        raise ValueError("fulfillment slot requires one support role")
    return inferred_roles


def _slot_support_refs_by_role(slot: FulfillmentSlot) -> dict[str, list[str]]:
    role = _slot_support_roles(slot)[0]
    refs = [_support_ref(evidence) for evidence in _slot_evidence(slot)]
    return {role: list(dict.fromkeys(ref for ref in refs if ref))}


def _slot_field_ids(slot: FulfillmentSlot) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            field_id
            for item in _slot_evidence(slot)
            for field_id in _evidence_field_ids(item)
            if field_id
        )
    )


def _evidence_field_ids(evidence: EvidenceItem) -> tuple[str, ...]:
    return evidence_field_ids(evidence)


def _slot_evidence(slot: FulfillmentSlot) -> tuple[EvidenceItem, ...]:
    return (
        *slot.metric_measure_evidence,
        *slot.value_evidence,
        *slot.row_count_basis_evidence,
        *slot.entity_evidence,
    )


def _support_ref(evidence: EvidenceItem) -> str:
    return evidence.evidence_id
