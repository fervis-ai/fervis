"""Typed source-candidate evidence and declaration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing import TypeAlias
from typing_extensions import assert_never


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class EvidenceComponent:
    component_id: str
    field_id: str
    field_evidence_id: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.field_id or not self.field_evidence_id:
            raise ValueError("entity evidence component is incomplete")

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "component_id": self.component_id,
            "field_id": self.field_id,
        }
        if self.field_evidence_id:
            output["field_evidence_id"] = self.field_evidence_id
        return output


@dataclass(frozen=True)
class FieldEvidence:
    evidence_id: str
    field_id: str
    type: str
    field_ref: str = ""
    path: str = ""
    response_path: str = ""
    row_cardinality: str = ""
    row_path_id: str = ""
    row_source_id: str = ""
    label: str = ""
    description: str = ""
    roles: tuple[str, ...] = ()
    answer_output_ids: tuple[str, ...] = ()
    prior_answer_output_ids: tuple[str, ...] = ()
    presentation_only: bool = False
    entity_evidence_member: bool = False

    def payload(self) -> JsonObject:
        output: JsonObject = {"evidence_id": self.evidence_id}
        _put_text_evidence_fields(self, output=output)
        if self.roles:
            output["roles"] = list(self.roles)
        if self.answer_output_ids:
            output["answer_output_ids"] = list(self.answer_output_ids)
        if self.prior_answer_output_ids:
            output["prior_answer_output_ids"] = list(self.prior_answer_output_ids)
        if self.presentation_only:
            output["presentation_only"] = True
        if self.entity_evidence_member:
            output["entity_evidence_member"] = True
        return output


def _put_text_evidence_fields(item: FieldEvidence, *, output: JsonObject) -> None:
    for key, value in (
        ("type", item.type),
        ("field_id", item.field_id),
        ("field_ref", item.field_ref),
        ("path", item.path),
        ("response_path", item.response_path),
        ("row_cardinality", item.row_cardinality),
        ("row_path_id", item.row_path_id),
        ("row_source_id", item.row_source_id),
        ("label", item.label),
        ("description", item.description),
    ):
        if value:
            output[key] = value


@dataclass(frozen=True)
class ValueEvidence:
    evidence_id: str
    value_id: str
    type: str
    answer_output_ids: tuple[str, ...] = ()
    prior_answer_output_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.value_id:
            raise ValueError("value evidence requires evidence and value ids")

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "evidence_id": self.evidence_id,
            "value_id": self.value_id,
            "type": self.type,
        }
        if self.answer_output_ids:
            output["answer_output_ids"] = list(self.answer_output_ids)
        if self.prior_answer_output_ids:
            output["prior_answer_output_ids"] = list(self.prior_answer_output_ids)
        return output


@dataclass(frozen=True)
class RowPopulationEvidence:
    evidence_id: str
    row_path_id: str
    row_source_id: str
    row_cardinality: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.row_path_id:
            raise ValueError("row-population evidence requires evidence and path ids")

    @property
    def type(self) -> str:
        return "row_population"

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "evidence_id": self.evidence_id,
            "type": self.type,
            "field_id": self.row_path_id,
            "label": self.row_path_id,
            "row_path_id": self.row_path_id,
            "row_source_id": self.row_source_id,
            "row_cardinality": self.row_cardinality,
        }
        return output


@dataclass(frozen=True)
class CandidateKeyEvidence:
    evidence_id: str
    key_id: str
    entity_kind: str
    components: tuple[EvidenceComponent, ...]
    row_path_id: str
    row_source_id: str
    primary: bool = False
    stable: bool = True

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.key_id or not self.entity_kind:
            raise ValueError("candidate-key evidence identity is incomplete")
        if not self.components or not self.row_source_id or not self.row_path_id:
            raise ValueError("candidate-key evidence source is incomplete")

    @property
    def type(self) -> str:
        return "candidate_key"

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "evidence_id": self.evidence_id,
            "type": self.type,
            "key_id": self.key_id,
            "entity_kind": self.entity_kind,
            "components": [item.payload() for item in self.components],
            "row_path_id": self.row_path_id,
            "row_source_id": self.row_source_id,
        }
        if self.primary:
            output["primary"] = True
        if not self.stable:
            output["stable"] = False
        return output


@dataclass(frozen=True)
class EntityReferenceEvidence:
    evidence_id: str
    reference_id: str
    target_key_id: str
    target_entity_kind: str
    components: tuple[EvidenceComponent, ...]
    row_path_id: str
    row_source_id: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.reference_id:
            raise ValueError("entity-reference evidence requires ids")
        if not self.target_entity_kind or not self.target_key_id:
            raise ValueError("entity-reference evidence requires target key")
        if not self.components or not self.row_source_id or not self.row_path_id:
            raise ValueError("entity-reference evidence source is incomplete")

    @property
    def type(self) -> str:
        return "entity_reference"

    def payload(self) -> JsonObject:
        return {
            "evidence_id": self.evidence_id,
            "type": self.type,
            "reference_id": self.reference_id,
            "target_key_id": self.target_key_id,
            "target_entity_kind": self.target_entity_kind,
            "components": [item.payload() for item in self.components],
            "row_path_id": self.row_path_id,
            "row_source_id": self.row_source_id,
        }


EntityEvidence: TypeAlias = CandidateKeyEvidence | EntityReferenceEvidence
CountBasisEvidence: TypeAlias = FieldEvidence | RowPopulationEvidence
EvidenceItem: TypeAlias = (
    FieldEvidence | ValueEvidence | RowPopulationEvidence | EntityEvidence
)


def entity_evidence_entity_kind(evidence: EntityEvidence) -> str:
    match evidence:
        case CandidateKeyEvidence(entity_kind=entity_kind):
            return entity_kind
        case EntityReferenceEvidence(target_entity_kind=entity_kind):
            return entity_kind
        case _ as unreachable:
            assert_never(unreachable)


def entity_evidence_key_id(evidence: EntityEvidence) -> str:
    match evidence:
        case CandidateKeyEvidence(key_id=key_id):
            return key_id
        case EntityReferenceEvidence(target_key_id=key_id):
            return key_id
        case _ as unreachable:
            assert_never(unreachable)


def evidence_field_ids(item: EvidenceItem) -> tuple[str, ...]:
    match item:
        case FieldEvidence(field_id=field_id):
            return (field_id,)
        case (
            CandidateKeyEvidence(components=components)
            | EntityReferenceEvidence(components=components)
        ):
            return tuple(component.field_id for component in components)
        case ValueEvidence() | RowPopulationEvidence():
            return ()
        case _ as unreachable:
            assert_never(unreachable)


def evidence_row_path_id(item: EvidenceItem) -> str:
    match item:
        case (
            FieldEvidence(row_path_id=row_path_id)
            | RowPopulationEvidence(row_path_id=row_path_id)
            | CandidateKeyEvidence(row_path_id=row_path_id)
            | EntityReferenceEvidence(row_path_id=row_path_id)
        ):
            return row_path_id
        case ValueEvidence():
            return ""
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class CandidateKeyComponent:
    component_id: str
    field_id: str

    def payload(self) -> JsonObject:
        return {"component_id": self.component_id, "field_id": self.field_id}


@dataclass(frozen=True)
class CandidateKeyDeclaration:
    key_id: str
    entity_kind: str
    components: tuple[CandidateKeyComponent, ...]
    primary: bool = False
    stable: bool = True
    context_field_ids: tuple[str, ...] = ()

    def payload(self) -> JsonObject:
        return {
            "key_id": self.key_id,
            "entity_kind": self.entity_kind,
            "components": [item.payload() for item in self.components],
            "primary": self.primary,
            "stable": self.stable,
            "context_field_ids": list(self.context_field_ids),
        }


@dataclass(frozen=True)
class EntityReferenceDeclaration:
    reference_id: str
    target_entity_kind: str
    target_key_id: str
    components: tuple[CandidateKeyComponent, ...]
    context_field_ids: tuple[str, ...] = ()

    def payload(self) -> JsonObject:
        return {
            "reference_id": self.reference_id,
            "target_entity_kind": self.target_entity_kind,
            "target_key_id": self.target_key_id,
            "components": [item.payload() for item in self.components],
            "context_field_ids": list(self.context_field_ids),
        }


@dataclass(frozen=True)
class ResultGrain:
    grain_id: str
    row_path_id: str
    row_source_id: str
    cardinality: str
    evidence_items: tuple[EvidenceItem, ...] = ()
    candidate_keys: tuple[CandidateKeyDeclaration, ...] = ()
    entity_references: tuple[EntityReferenceDeclaration, ...] = ()

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "grain_id": self.grain_id,
            "row_path_id": self.row_path_id,
            "row_source_id": self.row_source_id,
            "cardinality": self.cardinality,
        }
        if self.evidence_items:
            output["evidence_items"] = [item.payload() for item in self.evidence_items]
        if self.candidate_keys:
            output["candidate_keys"] = [item.payload() for item in self.candidate_keys]
        if self.entity_references:
            output["entity_references"] = [
                item.payload() for item in self.entity_references
            ]
        return output


@dataclass(frozen=True)
class CandidateField:
    field_id: str
    type: str = ""
    field_ref: str = ""
    path: str = ""
    response_path: str = ""
    roles: tuple[str, ...] = ()
    label: str = ""
    description: str = ""
    row_cardinality: str = ""
    row_path_id: str = ""
    row_source_id: str = ""
    answer_output_ids: tuple[str, ...] = ()
    prior_answer_output_ids: tuple[str, ...] = ()

    def payload(self) -> JsonObject:
        output: JsonObject = {"field_id": self.field_id}
        for key, value in (
            ("type", self.type),
            ("field_ref", self.field_ref),
            ("path", self.path),
            ("response_path", self.response_path),
            ("label", self.label),
            ("description", self.description),
            ("row_cardinality", self.row_cardinality),
            ("row_path_id", self.row_path_id),
            ("row_source_id", self.row_source_id),
        ):
            if value:
                output[key] = value
        if self.roles:
            output["roles"] = list(self.roles)
        if self.answer_output_ids:
            output["answer_output_ids"] = list(self.answer_output_ids)
        if self.prior_answer_output_ids:
            output["prior_answer_output_ids"] = list(self.prior_answer_output_ids)
        return output


@dataclass(frozen=True)
class FulfillmentSlot:
    fulfillment_slot_id: str
    answer_output_id: str
    compatibility_basis: str
    answer_output_role: str = ""
    metric_measure_evidence: tuple[FieldEvidence, ...] = ()
    value_evidence: tuple[FieldEvidence, ...] = ()
    row_count_basis_evidence: tuple[CountBasisEvidence, ...] = ()
    entity_evidence: tuple[EntityEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.fulfillment_slot_id:
            raise ValueError("fulfillment slot requires id")

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "fulfillment_slot_id": self.fulfillment_slot_id,
            "answer_output_id": self.answer_output_id,
        }
        if self.compatibility_basis:
            output["compatibility_basis"] = self.compatibility_basis
        if self.answer_output_role:
            output["answer_output_role"] = self.answer_output_role
        for key, items in (
            ("metric_measure_evidence", self.metric_measure_evidence),
            ("value_evidence", self.value_evidence),
            ("row_count_basis_evidence", self.row_count_basis_evidence),
            ("entity_evidence", self.entity_evidence),
        ):
            if items:
                output[key] = [item.payload() for item in items]
        return output


@dataclass(frozen=True)
class FulfillmentSupportSet:
    fulfillment_support_set_id: str
    answer_output_id: str
    fulfillment_slots: tuple[FulfillmentSlot, ...]
    fulfillment_choice_id: str = ""

    def __post_init__(self) -> None:
        if not self.fulfillment_support_set_id:
            raise ValueError("fulfillment support set requires id")
        if not self.fulfillment_slots:
            raise ValueError("fulfillment support set requires slots")

    def payload(self) -> JsonObject:
        output: JsonObject = {
            "fulfillment_support_set_id": self.fulfillment_support_set_id,
            "answer_output_id": self.answer_output_id,
            "fulfillment_slots": [item.payload() for item in self.fulfillment_slots],
        }
        if self.fulfillment_choice_id:
            output["fulfillment_choice_id"] = self.fulfillment_choice_id
        return output


@dataclass(frozen=True)
class EntityTarget:
    entity_kind: str
    key_id: str
    component_id: str

    def payload(self) -> JsonObject:
        return {
            "entity_kind": self.entity_kind,
            "key_id": self.key_id,
            "component_id": self.component_id,
        }


def parse_evidence_item(payload: Mapping[str, JsonValue]) -> EvidenceItem:
    evidence_type = _text(payload, "type")
    evidence_id = _text(payload, "evidence_id")
    components = tuple(
        EvidenceComponent(
            component_id=_text(item, "component_id"),
            field_id=_text(item, "field_id"),
            field_evidence_id=_text(item, "field_evidence_id"),
        )
        for item in _objects(payload, "components")
    )
    if evidence_type == "candidate_key":
        return CandidateKeyEvidence(
            evidence_id=evidence_id,
            key_id=_text(payload, "key_id"),
            entity_kind=_text(payload, "entity_kind"),
            components=components,
            row_path_id=_text(payload, "row_path_id"),
            row_source_id=_text(payload, "row_source_id"),
            primary=_boolean(payload, "primary"),
            stable=_boolean(payload, "stable", default=True),
        )
    if evidence_type == "entity_reference":
        return EntityReferenceEvidence(
            evidence_id=evidence_id,
            reference_id=_text(payload, "reference_id"),
            target_key_id=_text(payload, "target_key_id"),
            target_entity_kind=_text(payload, "target_entity_kind"),
            components=components,
            row_path_id=_text(payload, "row_path_id"),
            row_source_id=_text(payload, "row_source_id"),
        )
    if evidence_type == "row_population":
        return RowPopulationEvidence(
            evidence_id=evidence_id,
            row_path_id=_text(payload, "row_path_id"),
            row_source_id=_text(payload, "row_source_id"),
            row_cardinality=_text(payload, "row_cardinality"),
        )
    value_id = _text(payload, "value_id")
    if value_id:
        return ValueEvidence(
            evidence_id=evidence_id,
            value_id=value_id,
            type=evidence_type,
            answer_output_ids=_texts(payload, "answer_output_ids"),
            prior_answer_output_ids=_texts(payload, "prior_answer_output_ids"),
        )
    return FieldEvidence(
        evidence_id=_text(payload, "evidence_id"),
        field_id=_text(payload, "field_id"),
        type=evidence_type,
        field_ref=_text(payload, "field_ref"),
        path=_text(payload, "path"),
        response_path=_text(payload, "response_path"),
        row_cardinality=_text(payload, "row_cardinality"),
        row_path_id=_text(payload, "row_path_id"),
        row_source_id=_text(payload, "row_source_id"),
        label=_text(payload, "label"),
        description=_text(payload, "description"),
        roles=_texts(payload, "roles"),
        answer_output_ids=_texts(payload, "answer_output_ids"),
        prior_answer_output_ids=_texts(payload, "prior_answer_output_ids"),
        presentation_only=_boolean(payload, "presentation_only"),
        entity_evidence_member=_boolean(payload, "entity_evidence_member"),
    )


def parse_candidate_field(payload: Mapping[str, JsonValue]) -> CandidateField:
    return CandidateField(
        field_id=_text(payload, "field_id") or _text(payload, "id"),
        type=_text(payload, "type"),
        field_ref=_text(payload, "field_ref"),
        path=_text(payload, "path"),
        response_path=_text(payload, "response_path"),
        roles=_texts(payload, "roles"),
        label=_text(payload, "label"),
        description=_text(payload, "description"),
        row_cardinality=_text(payload, "row_cardinality"),
        row_path_id=_text(payload, "row_path_id"),
        row_source_id=_text(payload, "row_source_id"),
        answer_output_ids=_texts(payload, "answer_output_ids"),
        prior_answer_output_ids=_texts(payload, "prior_answer_output_ids"),
    )


def parse_candidate_key(payload: Mapping[str, JsonValue]) -> CandidateKeyDeclaration:
    return CandidateKeyDeclaration(
        key_id=_text(payload, "key_id"),
        entity_kind=_text(payload, "entity_kind"),
        components=_key_components(payload),
        primary=_boolean(payload, "primary"),
        stable=_boolean(payload, "stable", default=True),
        context_field_ids=_texts(payload, "context_field_ids"),
    )


def parse_entity_reference(
    payload: Mapping[str, JsonValue],
) -> EntityReferenceDeclaration:
    return EntityReferenceDeclaration(
        reference_id=_text(payload, "reference_id"),
        target_entity_kind=_text(payload, "target_entity_kind"),
        target_key_id=_text(payload, "target_key_id"),
        components=_key_components(payload),
        context_field_ids=_texts(payload, "context_field_ids"),
    )


def parse_result_grain(payload: Mapping[str, JsonValue]) -> ResultGrain:
    return ResultGrain(
        grain_id=_text(payload, "grain_id") or _text(payload, "row_path_id") or "root",
        row_path_id=_text(payload, "row_path_id"),
        row_source_id=_text(payload, "row_source_id"),
        cardinality=_text(payload, "cardinality"),
        evidence_items=tuple(
            parse_evidence_item(item) for item in _objects(payload, "evidence_items")
        ),
        candidate_keys=tuple(
            parse_candidate_key(item) for item in _objects(payload, "candidate_keys")
        ),
        entity_references=tuple(
            parse_entity_reference(item)
            for item in _objects(payload, "entity_references")
        ),
    )


def parse_fulfillment_slot(
    payload: Mapping[str, JsonValue],
    *,
    answer_output_id: str = "",
) -> FulfillmentSlot:
    metric_items = _evidence_items(payload, "metric_measure_evidence")
    value_items = _evidence_items(payload, "value_evidence")
    count_items = _evidence_items(payload, "row_count_basis_evidence")
    entity_items = _evidence_items(payload, "entity_evidence")
    if any(not isinstance(item, FieldEvidence) for item in metric_items):
        raise ValueError("metric fulfillment requires field evidence")
    if any(not isinstance(item, FieldEvidence) for item in value_items):
        raise ValueError("value fulfillment requires field evidence")
    if any(
        not isinstance(item, (FieldEvidence, RowPopulationEvidence))
        for item in count_items
    ):
        raise ValueError("row-count fulfillment requires field or population evidence")
    if any(
        not isinstance(item, (CandidateKeyEvidence, EntityReferenceEvidence))
        for item in entity_items
    ):
        raise ValueError("entity fulfillment requires entity evidence")
    return FulfillmentSlot(
        fulfillment_slot_id=_text(payload, "fulfillment_slot_id"),
        answer_output_id=_text(payload, "answer_output_id") or answer_output_id,
        compatibility_basis=_text(payload, "compatibility_basis"),
        answer_output_role=_text(payload, "answer_output_role"),
        metric_measure_evidence=tuple(
            item for item in metric_items if isinstance(item, FieldEvidence)
        ),
        value_evidence=tuple(
            item for item in value_items if isinstance(item, FieldEvidence)
        ),
        row_count_basis_evidence=tuple(
            item
            for item in count_items
            if isinstance(item, (FieldEvidence, RowPopulationEvidence))
        ),
        entity_evidence=tuple(
            item
            for item in entity_items
            if isinstance(item, (CandidateKeyEvidence, EntityReferenceEvidence))
        ),
    )


def parse_fulfillment_support_set(
    payload: Mapping[str, JsonValue],
) -> FulfillmentSupportSet:
    answer_output_id = _text(payload, "answer_output_id")
    return FulfillmentSupportSet(
        fulfillment_support_set_id=_text(payload, "fulfillment_support_set_id"),
        answer_output_id=answer_output_id,
        fulfillment_slots=tuple(
            parse_fulfillment_slot(item, answer_output_id=answer_output_id)
            for item in _objects(payload, "fulfillment_slots")
        ),
        fulfillment_choice_id=_text(payload, "fulfillment_choice_id"),
    )


def parse_entity_target(payload: Mapping[str, JsonValue]) -> EntityTarget:
    return EntityTarget(
        entity_kind=_text(payload, "entity_kind"),
        key_id=_text(payload, "key_id"),
        component_id=_text(payload, "component_id"),
    )


def _key_components(
    payload: Mapping[str, JsonValue],
) -> tuple[CandidateKeyComponent, ...]:
    return tuple(
        CandidateKeyComponent(
            component_id=_text(item, "component_id"),
            field_id=_text(item, "field_id"),
        )
        for item in _objects(payload, "components")
    )


def _evidence_items(
    payload: Mapping[str, JsonValue],
    key: str,
) -> tuple[EvidenceItem, ...]:
    return tuple(parse_evidence_item(item) for item in _objects(payload, key))


def _text(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""


def _texts(payload: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _objects(
    payload: Mapping[str, JsonValue],
    key: str,
) -> tuple[dict[str, JsonValue], ...]:
    value = payload.get(key)
    if isinstance(value, dict):
        return (value,)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _boolean(
    payload: Mapping[str, JsonValue],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else default
