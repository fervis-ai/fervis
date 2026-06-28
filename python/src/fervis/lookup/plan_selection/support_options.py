"""Plan selection support options projected from source candidates."""

from __future__ import annotations

from typing import Any


def plan_selection_support_options(
    candidate: dict[str, Any],
) -> tuple[dict[str, object], ...]:
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
        for index, support_set in enumerate(
            _candidate_fulfillment_support_sets(candidate),
            start=1,
        )
        if isinstance(support_set, dict)
        for option in (_fulfillment_support_option(support_set, index=index),)
        if option is not None
    )
    return (*intrinsic_options, *fulfillment_options)


def plan_selection_support_option_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(option.get("support_set_id") or "")
        for option in plan_selection_support_options(candidate)
        if str(option.get("support_set_id") or "")
    )


def plan_selection_fulfillment_support_sets(
    candidate: dict[str, Any],
) -> tuple[Any, ...]:
    return _candidate_fulfillment_support_sets(candidate)


def _fulfillment_support_option(
    support_set: dict[str, Any],
    *,
    index: int,
) -> dict[str, object] | None:
    binding_support_set_id = _support_set_binding_id(support_set)
    if not binding_support_set_id:
        return None
    support_roles: list[str] = []
    support_refs_by_role: dict[str, list[str]] = {}
    field_ids: list[str] = []
    for slot in support_set.get("fulfillment_slots") or ():
        if not isinstance(slot, dict):
            continue
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
    payload: dict[str, object] = {
        "support_set_id": f"support_set_{index}",
        "binding_support_set_id": binding_support_set_id,
        "support_roles": support_roles,
    }
    answer_output_id = str(support_set.get("answer_output_id") or "")
    if answer_output_id:
        payload["answer_output_id"] = answer_output_id
    if support_refs_by_role:
        payload["support_refs_by_role"] = {
            role: refs for role, refs in support_refs_by_role.items() if refs
        }
    if field_ids:
        payload["field_ids"] = field_ids
    return payload


def _value_support_option(candidate: dict[str, Any]) -> dict[str, object] | None:
    if str(candidate.get("kind") or "") != "value":
        return None
    value_id = str(
        candidate.get("value_id") or candidate.get("source_candidate_id") or ""
    )
    if not value_id:
        return None
    return {
        "support_set_id": f"support.{value_id}.value",
        "support_roles": ["VALUE_SOURCE"],
        "support_refs_by_role": {"VALUE_SOURCE": [value_id]},
    }


def _calendar_support_option(candidate: dict[str, Any]) -> dict[str, object] | None:
    calendar_id = str(candidate.get("calendar_id") or "")
    source_candidate_id = str(candidate.get("source_candidate_id") or calendar_id)
    if not calendar_id or not source_candidate_id:
        return None
    return {
        "support_set_id": f"support.{source_candidate_id}.calendar",
        "support_roles": ["CALENDAR_SOURCE"],
        "support_refs_by_role": {"CALENDAR_SOURCE": [calendar_id]},
    }


def _candidate_fulfillment_support_sets(candidate: dict[str, Any]) -> tuple[Any, ...]:
    direct = candidate.get("fulfillment_support_sets")
    if direct:
        return tuple(direct)
    binding_surface = candidate.get("binding_surface")
    if isinstance(binding_surface, dict):
        return tuple(binding_surface.get("fulfillment_support_sets") or ())
    return ()


def _support_set_binding_id(support_set: dict[str, Any]) -> str:
    return str(
        support_set.get("fulfillment_support_set_id")
        or support_set.get("fulfillment_choice_id")
        or ""
    )


def _slot_support_roles(slot: dict[str, Any]) -> tuple[str, ...]:
    roles: list[str] = []
    for key, role in (
        ("metric_measure_evidence", "MEASURED_VALUE"),
        ("row_count_basis_evidence", "ROW_POPULATION"),
        ("group_key_evidence", "GROUP_KEY"),
        ("scope_evidence", "POPULATION_SCOPE"),
    ):
        if slot.get(key):
            roles.append(role)
    return tuple(roles)


def _slot_support_refs_by_role(slot: dict[str, Any]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for key, role in (
        ("metric_measure_evidence", "MEASURED_VALUE"),
        ("row_count_basis_evidence", "ROW_POPULATION"),
        ("group_key_evidence", "GROUP_KEY"),
        ("scope_evidence", "POPULATION_SCOPE"),
    ):
        refs = [
            _support_ref(evidence)
            for evidence in slot.get(key) or ()
            if isinstance(evidence, dict)
        ]
        refs = [ref for ref in refs if ref]
        if refs:
            output[role] = list(dict.fromkeys(refs))
    return output


def _slot_field_ids(slot: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            field_id
            for key in (
                "metric_measure_evidence",
                "row_count_basis_evidence",
                "group_key_evidence",
                "scope_evidence",
            )
            for item in slot.get(key) or ()
            if isinstance(item, dict)
            for field_id in (str(item.get("field_id") or ""),)
            if field_id
        )
    )


def _support_ref(evidence: dict[str, Any]) -> str:
    for key in ("evidence_id", "row_path_id", "field_id"):
        value = str(evidence.get(key) or "")
        if value:
            return value
    if str(evidence.get("type") or "") == "row_population":
        return "row_population"
    return ""
