"""Candidate payload accessors for source-binding parsing."""

from __future__ import annotations

from fervis.lookup.source_binding.compiler_ir import (
    SourceAppliedFilter,
)
from fervis.lookup.source_binding.evidence_types import evidence_item_can_measure
from fervis.lookup.source_binding.model import (
    BoundSource,
    SourceEvidenceItem,
    SourceField,
    SourceFulfillment,
)
from fervis.lookup.source_binding.candidates.model import SourceCandidate
from fervis.lookup.source_binding.candidates.contracts import (
    CandidateField,
    CandidateKeyEvidence,
    EvidenceItem,
    EntityReferenceEvidence,
    FieldEvidence,
    RowPopulationEvidence,
    ValueEvidence,
    evidence_field_ids,
)


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
]


def candidate_value_is_used_by_bound_source(
    candidate: SourceCandidate,
    bound: BoundSource,
) -> bool:
    source_field_id = candidate.source_field_id
    if not source_field_id:
        return True
    answer_field_ids = {
        item.field_id
        for item in bound.evidence_items
        if item.evidence_id
        in {
            evidence_id
            for fulfillment in bound.fulfillments
            for evidence_id in fulfillment.field_evidence_ids()
        }
        and item.field_id
    }
    return not answer_field_ids or source_field_id in answer_field_ids


def candidate_metric_measure_evidence_ids(
    candidate: SourceCandidate,
) -> tuple[str, ...]:
    available = candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            item.evidence_id
            for item in candidate_evidence_items(candidate)
            if isinstance(item, FieldEvidence)
            and evidence_item_can_measure(item)
            and item.evidence_id
            and item.evidence_id in available
        )
    )


def candidate_row_count_basis_evidence_ids(
    candidate: SourceCandidate,
) -> tuple[str, ...]:
    available = candidate_evidence_ids(candidate)
    return tuple(
        dict.fromkeys(
            item.evidence_id
            for item in candidate_evidence_items(candidate)
            if item.type.lower() == "row_population"
            and item.evidence_id
            and item.evidence_id in available
        )
    )


def candidate_evidence_items(
    candidate: SourceCandidate,
) -> tuple[EvidenceItem, ...]:
    return candidate.evidence_items


def candidate_evidence_ids(candidate: SourceCandidate) -> set[str]:
    return {item.evidence_id for item in candidate.evidence_items if item.evidence_id}


def candidate_cardinality(candidate: SourceCandidate) -> str:
    return candidate.cardinality


def candidate_applied_filters(
    candidate: SourceCandidate,
) -> tuple[SourceAppliedFilter, ...]:
    return candidate.applied_filters


def candidate_field_ids(candidate: SourceCandidate) -> set[str]:
    return {
        item.field_id
        for item in candidate.evidence_items
        if isinstance(item, FieldEvidence) and item.field_id
    }


def candidate_source_fields(
    candidate: SourceCandidate,
    *,
    row_source_id: str = "",
    evidence_items: tuple[SourceEvidenceItem, ...] = (),
    fulfillments: tuple[SourceFulfillment, ...] = (),
    required_field_ids: tuple[str, ...] = (),
    plan_shape: str = "",
) -> tuple[SourceField, ...]:
    selected_evidence_ids = {
        evidence_id
        for fulfillment in fulfillments
        for evidence_id in fulfillment.field_evidence_ids()
    }
    selected_field_ids = {
        item.field_id
        for item in evidence_items
        if item.evidence_id in selected_evidence_ids and item.field_id
    }
    selected_field_ids.update(required_field_ids)
    selected_field_ids.update(
        item.field_id
        for item in candidate.evidence_items
        if isinstance(item, FieldEvidence)
        and item.row_source_id == row_source_id
        and not item.presentation_only
    )
    if plan_shape == "joined_rows":
        selected_field_ids.update(_entity_evidence_field_ids(candidate))
    selected_field_ids.update(
        field_id
        for applied_filter in candidate.applied_filters
        for field_id in applied_filter.predicate_field_ids
        if field_id
    )
    fields = [
        SourceField(
            field_id=item.field_id,
            type=item.type,
            roles=item.roles,
            label=item.label,
            row_cardinality=item.row_cardinality,
        )
        for item in candidate.evidence_items
        if isinstance(item, FieldEvidence)
        and item.field_id in selected_field_ids
        and _field_belongs_to_row_source(item, row_source_id=row_source_id)
        and _candidate_field_selectable_for_planning(item)
    ]
    existing_field_ids = {field.field_id for field in fields}
    fields.extend(
        SourceField(
            field_id=item.field_id,
            type=item.type,
            row_cardinality=item.row_cardinality,
        )
        for item in evidence_items
        if item.evidence_id in selected_evidence_ids
        and item.field_id
        and _field_belongs_to_row_source(item, row_source_id=row_source_id)
        and item.type != "row_population"
        and _field_type_selectable_for_planning(item.type)
        and item.field_id not in existing_field_ids
    )
    existing_field_ids.update(field.field_id for field in fields)
    return tuple(fields)


def _field_belongs_to_row_source(
    field: FieldEvidence | SourceEvidenceItem,
    *,
    row_source_id: str,
) -> bool:
    return not row_source_id or field.row_source_id == row_source_id


def _entity_evidence_field_ids(candidate: SourceCandidate) -> tuple[str, ...]:
    field_ids: list[str] = []
    for item in candidate_evidence_items(candidate):
        if item.type not in {
            "candidate_key",
            "entity_reference",
        }:
            continue
        field_ids.extend(_entity_evidence_component_field_ids(item))
    return tuple(dict.fromkeys(field_ids))


def _entity_evidence_component_field_ids(
    evidence_item: EvidenceItem,
) -> tuple[str, ...]:
    return evidence_field_ids(evidence_item)


def _candidate_field_selectable_for_planning(
    field: CandidateField | FieldEvidence,
) -> bool:
    return _field_type_selectable_for_planning(field.type)


def _field_type_selectable_for_planning(field_type: str) -> bool:
    return field_type.lower() != "object"


def candidate_source_evidence_items(
    candidate: SourceCandidate,
) -> tuple[SourceEvidenceItem, ...]:
    evidence_items = candidate_evidence_items(candidate)
    return tuple(
        source_item
        for item in evidence_items
        for source_item in (_source_evidence_item(item),)
        if source_item is not None
    )


def _source_evidence_item(item: EvidenceItem) -> SourceEvidenceItem | None:
    if isinstance(item, FieldEvidence):
        return SourceEvidenceItem(
            evidence_id=item.evidence_id,
            field_id=item.field_id,
            type=item.type,
            row_cardinality=item.row_cardinality,
            row_source_id=item.row_source_id,
        )
    if isinstance(item, ValueEvidence):
        return SourceEvidenceItem(
            evidence_id=item.evidence_id,
            value_id=item.value_id,
            type=item.type,
        )
    if isinstance(item, RowPopulationEvidence):
        return SourceEvidenceItem(
            evidence_id=item.evidence_id,
            field_id=item.row_path_id,
            type=item.type,
            row_cardinality=item.row_cardinality,
            row_source_id=item.row_source_id,
        )
    if isinstance(item, (CandidateKeyEvidence, EntityReferenceEvidence)):
        return None
    raise AssertionError("unsupported source evidence variant")
