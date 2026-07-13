"""Model-facing fulfillment evidence projection."""

from __future__ import annotations

from ._shared import Any


_SLOT_EVIDENCE_KEYS = (
    ("metric_measure_evidence", "metric"),
    ("value_evidence", "value"),
    ("row_count_basis_evidence", "row_count_basis"),
    ("entity_evidence", "entity"),
)


def model_visible_fulfillment_evidence(
    choice: dict[str, Any],
    *,
    candidate: dict[str, Any],
) -> tuple[dict[str, str], ...]:
    """Project a fulfillment choice into compact semantic evidence items."""

    choice_values_by_field = _choice_values_by_field(candidate)
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for slot in choice.get("fulfillment_slots") or ():
        if not isinstance(slot, dict):
            continue
        for slot_key, evidence_kind in _SLOT_EVIDENCE_KEYS:
            for item in slot.get(slot_key) or ():
                if not isinstance(item, dict):
                    continue
                evidence = _model_visible_evidence_item(
                    item,
                    kind=evidence_kind,
                    choice_values_by_field=choice_values_by_field,
                )
                key = (
                    evidence.get("kind", ""),
                    evidence.get("evidence_id", ""),
                    evidence.get("field", ""),
                )
                if key in seen:
                    continue
                output.append(evidence)
                seen.add(key)
    return tuple(output)


def _model_visible_evidence_item(
    item: dict[str, Any],
    *,
    kind: str,
    choice_values_by_field: dict[tuple[str, str], tuple[str, ...]],
) -> dict[str, str]:
    field = str(item.get("field_id") or "")
    component_field_ids = tuple(
        str(component.get("field_id") or "")
        for component in item.get("components") or ()
        if isinstance(component, dict) and str(component.get("field_id") or "")
    )
    row_path = str(item.get("row_path_id") or "")
    field_type = str(item.get("type") or "")
    output = {
        "kind": kind,
        "field": field or ", ".join(component_field_ids),
        "label": str(item.get("label") or ""),
        "row_path": row_path,
        "type": field_type,
        "evidence_id": str(item.get("evidence_id") or ""),
    }
    values = choice_values_by_field.get((field, row_path), ())
    if field_type == "choice" and values:
        output["meaning"] = f"Choices: {', '.join(values)}."
    evidence_type = str(item.get("type") or "")
    entity_kind = str(item.get("entity_kind") or item.get("target_entity_kind") or "")
    key_id = str(item.get("key_id") or item.get("target_key_id") or "")
    if entity_kind and key_id:
        output["entity_key"] = f"{entity_kind}.{key_id}"
    if kind == "entity" and evidence_type:
        output["kind"] = evidence_type
    return {key: value for key, value in output.items() if value}


def _choice_values_by_field(
    candidate: dict[str, Any],
) -> dict[tuple[str, str], tuple[str, ...]]:
    output: dict[tuple[str, str], tuple[str, ...]] = {}
    for predicate in candidate.get("row_predicates") or ():
        if not isinstance(predicate, dict):
            continue
        if str(predicate.get("type") or "") != "choice":
            continue
        values = tuple(
            str(value) for value in predicate.get("allowed_values") or () if str(value)
        )
        if not values:
            continue
        output[
            (
                str(predicate.get("field_id") or ""),
                str(predicate.get("row_path_id") or ""),
            )
        ] = values
    return output
