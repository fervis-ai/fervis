"""Backend-projected source fulfillment slots."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog import identity_payload_is_primary_stable
from fervis.lookup.question_contract import RequestedFact
from fervis.lookup.question_contract.answer_output_support import (
    ANSWER_OUTPUT_SUPPORT_ROLE_VALUES,
)
from fervis.lookup.source_binding.evidence_types import (
    evidence_item_can_measure,
)

from ._shared import Any
@dataclass(frozen=True)
class _EvidenceGroup:
    compatibility_basis: str
    metric_items: tuple[dict[str, Any], ...] = ()
    count_basis_items: tuple[dict[str, Any], ...] = ()
    scope_items: tuple[dict[str, Any], ...] = ()
    group_key_items: tuple[dict[str, Any], ...] = ()


FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE = {
    "GROUP_KEY": ("group_key",),
    "ROW_POPULATION": ("count_basis",),
    "MEASURED_VALUE": ("metric",),
    "POPULATION_SCOPE": ("scope",),
    "ANSWER_VALUE": ("group_key", "metric", "scope"),
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
        item
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
            allow_scoped_metrics_for_row_population=support_field_refs is not None,
        )
        for slot in (
            _fulfillment_slot(
                candidate_id=str(output.get("source_candidate_id") or ""),
                answer_output_id=answer_output.id,
                evidence_group=group,
            ),
        )
    ]
    output["fulfillment_slots"] = fulfillment_slots
    output["fulfillment_support_sets"] = _fulfillment_support_sets(
        candidate_id=str(output.get("source_candidate_id") or ""),
        fulfillment_slots=tuple(fulfillment_slots),
    )
    return output


def _support_evidence_items(
    evidence_items: tuple[dict[str, Any], ...],
    *,
    support_field_refs: frozenset[str] | None,
) -> tuple[dict[str, Any], ...]:
    if support_field_refs is None:
        return evidence_items
    return tuple(
        item
        for item in evidence_items
        if str(item.get("type") or "") == "row_population"
        or str(item.get("field_ref") or "") in support_field_refs
    )


def _answer_output_evidence_item_groups(
    evidence_items: tuple[dict[str, Any], ...],
    *,
    answer_output_id: str,
    answer_output_role: str,
    row_population_path_ids: tuple[str, ...],
    allow_scoped_metrics_for_row_population: bool,
) -> tuple[_EvidenceGroup, ...]:
    explicitly_scoped = tuple(
        item
        for item in evidence_items
        if answer_output_id
        in {str(raw) for raw in item.get("answer_output_ids") or () if str(raw).strip()}
    )
    row_population_groups = _row_population_count_basis_groups(
        evidence_items,
        row_path_ids=row_population_path_ids,
        compatibility_basis="source_result_grain",
    )
    if explicitly_scoped:
        groups = (
            *row_population_groups,
            *_evidence_item_groups(
                _with_structural_identity_items(
                    explicitly_scoped,
                    evidence_items=evidence_items,
                ),
                compatibility_basis="explicit_answer_output_metadata",
            ),
        )
    elif any(item.get("answer_output_ids") for item in evidence_items):
        groups = row_population_groups
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
            allow_scoped_metrics_for_row_population=(
                allow_scoped_metrics_for_row_population
            ),
        )
    )


def _evidence_group_matches_answer_output_role(
    group: _EvidenceGroup,
    *,
    answer_output_role: str,
    allow_scoped_metrics_for_row_population: bool,
) -> bool:
    if not answer_output_role:
        return True
    allowed_kinds = _allowed_evidence_group_kinds(
        answer_output_role,
        allow_scoped_metrics_for_row_population=allow_scoped_metrics_for_row_population,
    )
    if allowed_kinds is None:
        raise ValueError("unsupported answer output support role")
    allowed_kind_set = set(allowed_kinds)
    group_kind_set = _evidence_group_kinds(group)
    evidence_group_has_allowed_kind = bool(allowed_kind_set & group_kind_set)
    return evidence_group_has_allowed_kind


def _allowed_evidence_group_kinds(
    answer_output_role: str,
    *,
    allow_scoped_metrics_for_row_population: bool,
) -> tuple[str, ...] | None:
    kinds = FULFILLMENT_EVIDENCE_GROUP_KINDS_BY_ANSWER_ROLE.get(answer_output_role)
    if kinds is None:
        return None
    answer_role_is_row_population = answer_output_role == "ROW_POPULATION"
    scoped_metric_support_is_allowed = allow_scoped_metrics_for_row_population
    if answer_role_is_row_population and scoped_metric_support_is_allowed:
        return (*kinds, "metric")
    return kinds


def _evidence_group_kinds(group: _EvidenceGroup) -> set[str]:
    kinds: set[str] = set()
    if group.group_key_items:
        kinds.add("group_key")
    if group.count_basis_items:
        kinds.add("count_basis")
    if group.metric_items:
        kinds.add("metric")
    if group.scope_items:
        kinds.add("scope")
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
    evidence_items: tuple[dict[str, Any], ...],
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
    evidence_items: tuple[dict[str, Any], ...],
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
    evidence_items: tuple[dict[str, Any], ...],
    *,
    row_path_id: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in evidence_items
        if _evidence_item_is_row_population_for_path(item, row_path_id=row_path_id)
    )


def _with_structural_identity_items(
    items: tuple[dict[str, Any], ...],
    *,
    evidence_items: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    output = list(items)
    seen = {str(item.get("evidence_id") or "") for item in output}
    for item in evidence_items:
        evidence_id = str(item.get("evidence_id") or "")
        evidence_item_already_selected = evidence_id in seen
        evidence_item_has_stable_identity = _field_has_stable_identity_shape(item)
        if evidence_item_already_selected or not evidence_item_has_stable_identity:
            continue
        output.append(item)
        seen.add(evidence_id)
    return tuple(output)


def _evidence_item_groups(
    evidence_items: tuple[dict[str, Any], ...],
    *,
    compatibility_basis: str,
) -> tuple[_EvidenceGroup, ...]:
    group_key_groups = tuple(
        _EvidenceGroup(
            group_key_items=(item,),
            compatibility_basis=compatibility_basis,
        )
        for item in evidence_items
        if _field_can_group_rows(item)
    )
    metric_groups = tuple(
        _EvidenceGroup(
            metric_items=(item,),
            compatibility_basis=compatibility_basis,
        )
        for item in evidence_items
        if evidence_item_can_measure(item)
    )
    return (*group_key_groups, *metric_groups)


def _field_can_group_rows(evidence_item: dict[str, Any]) -> bool:
    field_id = str(evidence_item.get("field_id") or "")
    field_has_id = bool(field_id)
    if not field_has_id:
        return False
    field_is_measure = evidence_item_can_measure(evidence_item)
    if field_is_measure:
        return False
    field_type = str(evidence_item.get("type") or "").lower()
    field_type_can_group_rows = field_type not in {
        "any",
        "json",
        "object",
        "array",
        "list",
    }
    return field_type_can_group_rows


def _evidence_item_is_row_population_for_path(
    item: dict[str, Any],
    *,
    row_path_id: str,
) -> bool:
    evidence_is_for_row_path = str(item.get("row_path_id") or "") == row_path_id
    evidence_is_row_population = str(item.get("type") or "") == "row_population"
    return evidence_is_for_row_path and evidence_is_row_population


def _field_has_stable_identity_shape(evidence_item: dict[str, Any]) -> bool:
    return identity_payload_is_primary_stable(evidence_item.get("identity"))


def _fulfillment_slot(
    *,
    candidate_id: str,
    answer_output_id: str,
    evidence_group: _EvidenceGroup,
) -> dict[str, Any]:
    evidence_ids = tuple(
        dict.fromkeys(
            str(evidence_item.get("evidence_id") or "")
            for evidence_item in (
                *evidence_group.metric_items,
                *evidence_group.count_basis_items,
                *evidence_group.scope_items,
                *evidence_group.group_key_items,
            )
            if str(evidence_item.get("evidence_id") or "")
        )
    )
    output = {
        "fulfillment_slot_id": _fulfillment_slot_id(
            candidate_id=candidate_id,
            answer_output_id=answer_output_id,
            evidence_ids=evidence_ids,
            role_key=_fulfillment_slot_role_key(evidence_group),
        ),
        "answer_output_id": answer_output_id,
        "compatibility_basis": evidence_group.compatibility_basis,
    }
    if evidence_group.scope_items:
        output["scope_evidence"] = [
            _fulfillment_evidence_item(evidence_item)
            for evidence_item in evidence_group.scope_items
        ]
    if evidence_group.metric_items:
        output["metric_measure_evidence"] = [
            _fulfillment_evidence_item(evidence_item)
            for evidence_item in evidence_group.metric_items
        ]
    if evidence_group.count_basis_items:
        output["row_count_basis_evidence"] = [
            _fulfillment_evidence_item(evidence_item)
            for evidence_item in evidence_group.count_basis_items
        ]
    if evidence_group.group_key_items:
        output["group_key_evidence"] = [
            _fulfillment_evidence_item(evidence_item)
            for evidence_item in evidence_group.group_key_items
        ]
    return output


def _fulfillment_evidence_item(evidence_item: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: str(evidence_item.get(key) or "")
        for key in (
            "evidence_id",
            "field_id",
            "field_ref",
            "label",
            "path",
            "response_path",
            "type",
            "row_cardinality",
            "row_path_id",
            "row_source_id",
        )
        if str(evidence_item.get(key) or "")
    }
    identity = evidence_item.get("identity")
    if isinstance(identity, dict) and identity:
        output["identity"] = dict(identity)
    roles = tuple(str(role) for role in evidence_item.get("roles") or () if str(role))
    if roles:
        output["roles"] = list(roles)
    if "field_id" not in output:
        output["field_id"] = output.get("evidence_id", "")
    return output


def _fulfillment_slot_role_key(evidence_group: _EvidenceGroup) -> str:
    if evidence_group.metric_items:
        return "metric"
    if evidence_group.count_basis_items:
        return "count"
    if evidence_group.group_key_items:
        return "group"
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
    fulfillment_slots: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    slots_by_output: dict[str, list[dict[str, Any]]] = {}
    for slot in fulfillment_slots:
        answer_output_id = str(slot.get("answer_output_id") or "")
        if not answer_output_id:
            continue
        slots_by_output.setdefault(answer_output_id, []).append(slot)
    output: list[dict[str, Any]] = []
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
    slots: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _fulfillment_support_set(
            candidate_id=candidate_id,
            answer_output_id=answer_output_id,
            slots=(slot,),
        )
        for slot in slots
        if _support_slot_has_evidence(slot)
    )


def _support_slot_has_evidence(slot: dict[str, Any]) -> bool:
    return any(
        slot.get(key)
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "scope_evidence",
            "group_key_evidence",
        )
    )


def _fulfillment_support_set(
    *,
    candidate_id: str,
    answer_output_id: str,
    slots: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "fulfillment_support_set_id": _fulfillment_support_set_id(
            candidate_id=candidate_id,
            answer_output_id=answer_output_id,
            slot_ids=tuple(
                str(slot.get("fulfillment_slot_id") or "")
                for slot in slots
                if str(slot.get("fulfillment_slot_id") or "")
            ),
        ),
        "answer_output_id": answer_output_id,
        "fulfillment_slots": list(slots),
    }


def _fulfillment_support_set_id(
    *,
    candidate_id: str,
    answer_output_id: str,
    slot_ids: tuple[str, ...],
) -> str:
    slot_key = "__".join(slot_ids)
    return f"support.{candidate_id}.{answer_output_id}.{slot_key}"
