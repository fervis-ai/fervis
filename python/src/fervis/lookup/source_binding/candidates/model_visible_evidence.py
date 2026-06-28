"""Model-facing fulfillment evidence projection."""

from __future__ import annotations

from ._shared import Any


_SLOT_EVIDENCE_KEYS = (
    ("scope_evidence", "scope"),
    ("metric_measure_evidence", "metric"),
    ("row_count_basis_evidence", "row_count_basis"),
    ("group_key_evidence", "group_key"),
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
    row_path = str(item.get("row_path_id") or "")
    field_type = str(item.get("type") or "")
    output = {
        "kind": kind,
        "field": field,
        "label": str(item.get("label") or _field_label(field)),
        "row_path": row_path,
        "type": field_type,
        "evidence_id": str(item.get("evidence_id") or ""),
    }
    values = choice_values_by_field.get((field, row_path), ())
    if field_type == "choice" and values:
        output["meaning"] = f"Choices: {', '.join(values)}."
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


def _field_label(field_id: str) -> str:
    words = [word for word in str(field_id or "").replace("_", " ").split() if word]
    return " ".join(_title_word(word) for word in words)


def _title_word(word: str) -> str:
    if word.casefold() == "id":
        return "ID"
    return word[:1].upper() + word[1:]
