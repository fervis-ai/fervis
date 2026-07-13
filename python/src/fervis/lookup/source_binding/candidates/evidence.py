"""Evidence field payloads for source-binding candidates."""

from collections.abc import Mapping

from ._shared import RowSource, RowSourceField
from fervis.lookup.source_binding.candidates.contracts import (
    CandidateField,
    CandidateKeyEvidence,
    CandidateKeyComponent,
    EvidenceComponent,
    EvidenceItem,
    EntityReferenceEvidence,
    FieldEvidence,
    JsonObject,
    JsonValue,
    ResultGrain,
    ValueEvidence,
    parse_candidate_field,
    parse_result_grain,
)
from .row_population import row_population_evidence_item


def _read_field_payloads(sources: tuple[RowSource, ...]) -> list[JsonObject]:
    fields: list[RowSourceField] = []
    seen: set[str] = set()
    for source in sources:
        for field in source.fields:
            if field.id in seen:
                continue
            seen.add(field.id)
            fields.append(field)
    return [_field_contract(field).payload() for field in fields]


def _field_contract(field: RowSourceField) -> CandidateField:
    return CandidateField(
        field_id=field.id,
        type=field.type,
        field_ref=field.field_ref,
        path=field.path,
        response_path=field.response_path,
        roles=tuple(role.value for role in field.allowed_roles),
        label=field.label if field.label != field.id else "",
    )


def _candidate_with_evidence_items(candidate: JsonObject) -> JsonObject:
    candidate_id = _text(candidate.get("source_candidate_id"))
    if not candidate_id:
        return candidate
    output = dict(candidate)
    raw_grains = _json_objects(candidate.get("result_grains"))
    if raw_grains:
        result_grains, evidence_items = _candidate_grain_evidence(
            raw_grains,
            candidate_id=candidate_id,
        )
        if result_grains:
            output["result_grains"] = _json_array(result_grains)
        if evidence_items:
            output["evidence_items"] = [item.payload() for item in evidence_items]
        return output
    evidence_items = _candidate_evidence_items(output, candidate_id=candidate_id)
    if evidence_items:
        output["evidence_items"] = [item.payload() for item in evidence_items]
    return output


def _candidate_grain_evidence(
    raw_grains: tuple[JsonObject, ...],
    *,
    candidate_id: str,
) -> tuple[list[JsonObject], list[EvidenceItem]]:
    result_grains: list[JsonObject] = []
    evidence_items: list[EvidenceItem] = []
    for raw_grain in raw_grains:
        grain = parse_result_grain(raw_grain)
        field_items = tuple(
            item for item in grain.evidence_items if isinstance(item, FieldEvidence)
        )
        display_field_ids = _display_field_ids(grain)
        entity_field_ids = _entity_field_ids(grain)
        grain_items: list[EvidenceItem] = [
            _field_evidence_item(
                item,
                evidence_id=f"{candidate_id}.{grain.grain_id}.{item.field_id}",
                row_path_id=item.row_path_id or grain.grain_id,
                row_source_id=grain.row_source_id,
                presentation_only=item.field_id in display_field_ids,
                entity_evidence_member=item.field_id in entity_field_ids,
            )
            for item in field_items
            if item.field_id
        ]
        row_population_item = row_population_evidence_item(
            grain.grain_id,
            row_cardinality=grain.cardinality,
            row_source_id=grain.row_source_id,
        )
        grain_items.append(row_population_item)
        grain_items.extend(
            _candidate_key_evidence_items(
                grain,
                candidate_id=candidate_id,
                field_evidence_items=tuple(grain_items),
            )
        )
        grain_items.extend(
            _entity_reference_evidence_items(
                grain,
                candidate_id=candidate_id,
                field_evidence_items=tuple(grain_items),
            )
        )
        grain_output = dict(raw_grain)
        grain_output["evidence_items"] = [item.payload() for item in grain_items]
        result_grains.append(grain_output)
        evidence_items.extend(grain_items)
    return result_grains, evidence_items


def _display_field_ids(grain: ResultGrain) -> frozenset[str]:
    key_fields = {
        field_id
        for declaration in grain.candidate_keys
        for field_id in declaration.context_field_ids
    }
    reference_fields = {
        field_id
        for declaration in grain.entity_references
        for field_id in declaration.context_field_ids
    }
    return frozenset(key_fields | reference_fields)


def _entity_field_ids(grain: ResultGrain) -> frozenset[str]:
    key_fields = {
        component.field_id
        for declaration in grain.candidate_keys
        for component in declaration.components
    }
    reference_fields = {
        component.field_id
        for declaration in grain.entity_references
        for component in declaration.components
    }
    return frozenset(key_fields | reference_fields)


def _candidate_key_evidence_items(
    grain: ResultGrain,
    *,
    candidate_id: str,
    field_evidence_items: tuple[EvidenceItem, ...],
) -> list[EvidenceItem]:
    evidence_ids = _field_evidence_id_index(field_evidence_items)
    items: list[EvidenceItem] = []
    for declaration in grain.candidate_keys:
        if not _declaration_fields_exist(declaration.components, evidence_ids):
            continue
        components = _evidence_components(declaration.components, evidence_ids)
        items.append(
            CandidateKeyEvidence(
                evidence_id=(
                    f"{candidate_id}.{grain.grain_id}.key.{declaration.key_id}"
                ),
                key_id=declaration.key_id,
                entity_kind=declaration.entity_kind,
                components=components,
                row_path_id=grain.grain_id,
                row_source_id=grain.row_source_id,
                primary=declaration.primary,
                stable=declaration.stable,
            )
        )
    return items


def _entity_reference_evidence_items(
    grain: ResultGrain,
    *,
    candidate_id: str,
    field_evidence_items: tuple[EvidenceItem, ...],
) -> list[EvidenceItem]:
    evidence_ids = _field_evidence_id_index(field_evidence_items)
    items: list[EvidenceItem] = []
    for declaration in grain.entity_references:
        if not _declaration_fields_exist(declaration.components, evidence_ids):
            continue
        components = _evidence_components(declaration.components, evidence_ids)
        items.append(
            EntityReferenceEvidence(
                evidence_id=(
                    f"{candidate_id}.{grain.grain_id}.reference."
                    f"{declaration.reference_id}"
                ),
                reference_id=declaration.reference_id,
                target_entity_kind=declaration.target_entity_kind,
                target_key_id=declaration.target_key_id,
                components=components,
                row_path_id=grain.grain_id,
                row_source_id=grain.row_source_id,
            )
        )
    return items


def _declaration_fields_exist(
    components: tuple[CandidateKeyComponent, ...],
    evidence_ids: Mapping[str, str],
) -> bool:
    return bool(components) and all(
        component.field_id in evidence_ids for component in components
    )


def _evidence_components(
    components: tuple[CandidateKeyComponent, ...],
    evidence_ids: Mapping[str, str],
) -> tuple[EvidenceComponent, ...]:
    return tuple(
        EvidenceComponent(
            component_id=component.component_id,
            field_id=component.field_id,
            field_evidence_id=evidence_ids[component.field_id],
        )
        for component in components
    )


def _field_evidence_id_index(items: tuple[EvidenceItem, ...]) -> dict[str, str]:
    return {
        item.field_id: item.evidence_id
        for item in items
        if isinstance(item, FieldEvidence) and item.field_id and item.evidence_id
    }


def _candidate_evidence_items(
    candidate: JsonObject,
    *,
    candidate_id: str,
) -> list[EvidenceItem]:
    fields = tuple(
        parse_candidate_field(item) for item in _json_objects(candidate.get("fields"))
    )
    items: list[EvidenceItem] = [
        _field_evidence_item(
            field,
            evidence_id=f"{candidate_id}_evidence_{index}",
        )
        for index, field in enumerate(fields, start=1)
        if field.field_id
    ]
    if items:
        return items
    value_id = _text(candidate.get("value_id"))
    if not value_id:
        return []
    value_type = _text(candidate.get("type")) or _text(candidate.get("literal_type"))
    return [
        ValueEvidence(
            evidence_id=f"{candidate_id}_value",
            value_id=value_id,
            type=value_type,
            answer_output_ids=_texts(candidate.get("answer_output_ids")),
            prior_answer_output_ids=_texts(candidate.get("prior_answer_output_ids")),
        )
    ]


def _field_evidence_item(
    field: CandidateField | FieldEvidence,
    *,
    evidence_id: str,
    row_path_id: str = "",
    row_source_id: str = "",
    presentation_only: bool = False,
    entity_evidence_member: bool = False,
) -> FieldEvidence:
    return FieldEvidence(
        evidence_id=evidence_id,
        field_id=field.field_id,
        field_ref=field.field_ref,
        path=field.path,
        response_path=field.response_path,
        row_cardinality=field.row_cardinality,
        row_path_id=row_path_id or field.row_path_id,
        row_source_id=row_source_id or field.row_source_id,
        label=field.label,
        description=field.description,
        type=field.type,
        roles=field.roles,
        answer_output_ids=field.answer_output_ids,
        prior_answer_output_ids=field.prior_answer_output_ids,
        presentation_only=presentation_only,
        entity_evidence_member=entity_evidence_member,
    )


def _json_objects(value: JsonValue | None) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _text(value: JsonValue | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _texts(value: JsonValue | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _json_array(values: list[JsonObject]) -> list[JsonValue]:
    return [value for value in values]
