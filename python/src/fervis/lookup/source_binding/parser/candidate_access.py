"""Candidate payload accessors for source-binding parsing."""

from __future__ import annotations

from typing import Any

from fervis.lookup.fact_plan.relations import RelationSourceRowFilter
from fervis.lookup.relation_catalog import IdentityMetadata
from fervis.lookup.source_binding.evidence_types import evidence_item_can_measure
from fervis.lookup.source_binding.model import BoundSource, SourceEvidenceItem, SourceField, SourceFulfillment
from fervis.lookup.source_binding.review_surface import source_binding_review_surface


__all__ = [
    "candidate_applied_filters",
    "candidate_cardinality",
    "candidate_evidence_ids",
    "candidate_evidence_items",
    "candidate_field_ids",
    "candidate_metric_measure_evidence_ids",
    "candidate_row_count_basis_evidence_ids",
    "candidate_source_evidence_items",
    "candidate_source_fields",
    "candidate_value_is_used_by_bound_source",
    "identity_metadata",
]


def candidate_value_is_used_by_bound_source(
    candidate: Any,
    bound: BoundSource,
) -> bool:
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        return True
    source_field_id = str(payload.get("source_field_id") or "")
    if not source_field_id:
        return True
    answer_field_ids = {
        item.field_id
        for item in bound.evidence_items
        if item.evidence_id
        in {
            evidence_id
            for fulfillment in bound.fulfillments
            for evidence_id in (
                *fulfillment.metric_measure_evidence_ids,
                *fulfillment.row_count_basis_evidence_ids,
                *fulfillment.group_key_evidence_ids,
            )
        }
        and item.field_id
    }
    return not answer_field_ids or source_field_id in answer_field_ids


def candidate_metric_measure_evidence_ids(candidate: Any) -> tuple[str, ...]:
    available = candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            evidence_id
            for item in candidate_evidence_items(candidate)
            if evidence_item_can_measure(item)
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id and evidence_id in available
        )
    )


def candidate_row_count_basis_evidence_ids(candidate: Any) -> tuple[str, ...]:
    available = candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            evidence_id
            for item in candidate_evidence_items(candidate)
            if str(item.get("type") or "").lower() == "row_population"
            for evidence_id in (str(item.get("evidence_id") or ""),)
            if evidence_id and evidence_id in available
        )
    )


def candidate_evidence_items(candidate: Any) -> tuple[dict[str, Any], ...]:
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        return ()
    return tuple(
        item for item in payload.get("evidence_items") or () if isinstance(item, dict)
    )


def candidate_evidence_ids(candidate: Any) -> set[str]:
    payload = getattr(candidate, "payload", None)
    evidence_items = payload.get("evidence_items") if isinstance(payload, dict) else ()
    if evidence_items:
        return {
            evidence_id
            for item in evidence_items or ()
            if isinstance(item, dict)
            for evidence_id in (str(item.get("evidence_id") or "").strip(),)
            if evidence_id
        }
    field_ids = candidate_field_ids(candidate)
    if field_ids:
        return field_ids
    value_id = str(getattr(candidate, "value_id", "") or "").strip()
    return {value_id} if value_id else set()


def candidate_cardinality(candidate: Any) -> str:
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("cardinality") or "").strip()


def candidate_applied_filters(candidate: Any) -> tuple[dict[str, Any], ...]:
    payload = getattr(candidate, "payload", None)
    filters = payload.get("applied_filters") if isinstance(payload, dict) else ()
    return tuple(dict(item) for item in filters or () if isinstance(item, dict))


def candidate_field_ids(candidate: Any) -> set[str]:
    return {
        field_id
        for field in candidate.fields
        if isinstance(field, dict)
        for field_id in (str(field.get("field_id") or field.get("id") or "").strip(),)
        if field_id
    }


def candidate_source_fields(
    candidate: Any,
    *,
    evidence_items: tuple[SourceEvidenceItem, ...] = (),
    fulfillments: tuple[SourceFulfillment, ...] = (),
    row_filters: tuple[RelationSourceRowFilter, ...] = (),
) -> tuple[SourceField, ...]:
    fields = [
        SourceField(
            field_id=field_id,
            type=str(field.get("type") or ""),
            roles=tuple(str(role) for role in field.get("roles") or ()),
            label=str(field.get("label") or ""),
            row_cardinality=str(field.get("row_cardinality") or ""),
            identity=identity_metadata(field.get("identity")),
        )
        for field in candidate.fields
        if isinstance(field, dict)
        for field_id in (str(field.get("field_id") or field.get("id") or "").strip(),)
        if field_id and _candidate_field_selectable_for_planning(field)
    ]
    existing_field_ids = {field.field_id for field in fields}
    selected_evidence_ids = {
        evidence_id
        for fulfillment in fulfillments
        for evidence_id in fulfillment.all_evidence_ids()
    }
    fields.extend(
        SourceField(
            field_id=item.field_id,
            type=item.type,
            row_cardinality=item.row_cardinality,
            identity=item.identity,
        )
        for item in evidence_items
        if item.evidence_id in selected_evidence_ids
        and item.field_id
        and item.type != "row_population"
        and _field_type_selectable_for_planning(item.type)
        and item.field_id not in existing_field_ids
    )
    existing_field_ids.update(field.field_id for field in fields)
    predicate_types = {
        axis.field_id: axis.field_type
        for axis in source_binding_review_surface(candidate).row_predicates.values()
    }
    fields.extend(
        SourceField(
            field_id=row_filter.field_id,
            type=predicate_types.get(row_filter.field_id, ""),
            roles=("predicate",),
        )
        for row_filter in row_filters
        if row_filter.field_id and row_filter.field_id not in existing_field_ids
    )
    return tuple(fields)


def _candidate_field_selectable_for_planning(field: dict[str, Any]) -> bool:
    return _field_type_selectable_for_planning(str(field.get("type") or ""))


def _field_type_selectable_for_planning(field_type: str) -> bool:
    return field_type.lower() != "object"


def candidate_source_evidence_items(candidate: Any) -> tuple[SourceEvidenceItem, ...]:
    payload = getattr(candidate, "payload", None)
    evidence_items = payload.get("evidence_items") if isinstance(payload, dict) else ()
    row_cardinality_by_field_id = {
        str(field.get("field_id") or field.get("id") or "").strip(): str(
            field.get("row_cardinality") or ""
        ).strip()
        for field in getattr(candidate, "fields", ())
        if isinstance(field, dict)
        and str(field.get("field_id") or field.get("id") or "").strip()
    }
    return tuple(
        SourceEvidenceItem(
            evidence_id=evidence_id,
            field_id=str(item.get("field_id") or "").strip(),
            value_id=str(item.get("value_id") or "").strip(),
            type=str(item.get("type") or "").strip(),
            row_cardinality=(
                str(item.get("row_cardinality") or "").strip()
                or row_cardinality_by_field_id.get(
                    str(item.get("field_id") or "").strip(), ""
                )
            ),
            row_source_id=str(item.get("row_source_id") or "").strip(),
            identity=identity_metadata(item.get("identity")),
        )
        for item in evidence_items or ()
        if isinstance(item, dict)
        for evidence_id in (str(item.get("evidence_id") or "").strip(),)
        if evidence_id
    )


def identity_metadata(raw: Any) -> IdentityMetadata | None:
    if not isinstance(raw, dict) or not raw:
        return None
    entity_ref = str(raw.get("entity_ref") or raw.get("entityRef") or "").strip()
    identity_field = str(raw.get("identity_field") or raw.get("idField") or "").strip()
    if not entity_ref or not identity_field:
        return None
    return IdentityMetadata(
        entity_ref=entity_ref,
        identity_field=identity_field,
        primary_key=bool(raw.get("primary_key") or raw.get("primaryKey")),
        stable=bool(raw.get("stable", True)),
    )
