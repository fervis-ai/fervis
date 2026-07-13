"""Typed fact addresses for fervis memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from typing import Any

from fervis.memory._serialization import without_empty
from fervis.lookup.canonical_data import (
    EntityKeyComponentValue,
    EntityKeyValue,
    ResultValue,
    runtime_value_from_payload,
    runtime_value_to_payload,
)


class FactAddressKind(StrEnum):
    ENTITY = "entity"
    VALUE = "value"
    RELATION = "relation"
    ROW = "row"
    OUTCOME = "outcome"
    PROOF = "proof"


class RelationSourceKind(StrEnum):
    API_READ = "api_read"
    GENERATED_DOMAIN = "generated_domain"
    CATALOG_DOMAIN = "catalog_domain"
    ACTIVATED_FACT_ADDRESS = "activated_fact_address"
    OPERATION_OUTPUT = "operation_output"


_TERMINAL_OUTCOMES = frozenset(
    {
        "impossible",
        "no_data",
        "undefined",
    }
)


@dataclass(frozen=True)
class EvidenceRef:
    step_ids: tuple[str, ...] = ()
    field_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return without_empty(
            {
                "stepIds": list(self.step_ids),
                "fieldRefs": list(self.field_refs),
            }
        )


def evidence_ref_from_payload(payload: Any) -> EvidenceRef | None:
    if not isinstance(payload, dict):
        return None
    step_ids = tuple(
        str(item).strip() for item in payload.get("stepIds") or () if str(item).strip()
    )
    field_refs = tuple(
        str(item).strip()
        for item in payload.get("fieldRefs") or ()
        if str(item).strip()
    )
    if not step_ids and not field_refs:
        return None
    return EvidenceRef(step_ids=step_ids, field_refs=field_refs)


@dataclass(frozen=True)
class FactAddressValue:
    type: str
    value: ResultValue
    answer_output_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("fact address value requires type")
        if self.type == "entity_key" and not isinstance(self.value, EntityKeyValue):
            raise ValueError("entity-key fact address value requires typed key")
        if self.type != "entity_key" and isinstance(self.value, EntityKeyValue):
            raise ValueError("typed entity key requires entity_key value type")

    def to_dict(self) -> dict[str, Any]:
        value: Any = self.value
        if isinstance(self.value, EntityKeyValue):
            value = {
                "entityKind": self.value.entity_kind,
                "keyId": self.value.key_id,
                "components": {
                    component.component_id: runtime_value_to_payload(component.value)
                    for component in self.value.components
                },
            }
        else:
            value = runtime_value_to_payload(self.value)
        return without_empty(
            {
                "type": self.type,
                "value": value,
                "answer_output_ids": list(self.answer_output_ids),
            }
        )


@dataclass(frozen=True)
class FactAddress:
    address: str
    kind: FactAddressKind
    resource: str = ""
    key_id: str = ""
    reference_text: str = ""
    identity: dict[str, str] = field(default_factory=dict)
    accessor: dict[str, Any] = field(default_factory=dict)
    scalar_value: dict[str, Any] = field(default_factory=dict)
    display: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    derivation: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    grain_keys: tuple[str, ...] = ()
    field_coverage: dict[str, str] = field(default_factory=dict)
    completeness: dict[str, Any] = field(default_factory=dict)
    row_addresses: tuple[str, ...] = ()
    source_relation: str = ""
    grain: dict[str, Any] = field(default_factory=dict)
    values: dict[str, FactAddressValue] = field(default_factory=dict)
    rank: dict[str, Any] = field(default_factory=dict)
    terminal: str = ""
    clarification_questions: tuple[str, ...] = ()
    proof: dict[str, Any] = field(default_factory=dict)
    operation: str = ""
    subject: dict[str, Any] = field(default_factory=dict)
    relations: dict[str, str] = field(default_factory=dict)
    predicate: dict[str, Any] = field(default_factory=dict)
    evidence: EvidenceRef | None = None

    def __post_init__(self) -> None:
        if not self.address:
            raise ValueError("fact address requires address")
        _validate_address_variant(self)

    @classmethod
    def entity(
        cls,
        *,
        address: str,
        resource: str,
        key_id: str,
        reference_text: str,
        identity: dict[str, str],
        accessor: dict[str, Any] | None = None,
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.ENTITY,
            resource=resource,
            key_id=key_id,
            reference_text=reference_text,
            identity=dict(identity),
            accessor=dict(accessor or {}),
            evidence=evidence,
        )

    @classmethod
    def value(
        cls,
        *,
        address: str,
        value: dict[str, Any],
        scope: dict[str, Any] | None = None,
        derivation: dict[str, Any] | None = None,
        display: str = "",
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.VALUE,
            scalar_value=dict(value),
            display=display,
            scope=dict(scope or {}),
            derivation=dict(derivation or {}),
            evidence=evidence,
        )

    @classmethod
    def relation(
        cls,
        *,
        address: str,
        source: dict[str, Any],
        scope: dict[str, Any] | None = None,
        grain_keys: tuple[str, ...] = (),
        field_coverage: dict[str, str] | None = None,
        completeness: dict[str, Any] | None = None,
        row_addresses: tuple[str, ...] = (),
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.RELATION,
            source=dict(source),
            scope=dict(scope or {}),
            grain_keys=tuple(grain_keys),
            field_coverage=dict(field_coverage or {}),
            completeness=dict(completeness or {}),
            row_addresses=tuple(row_addresses),
            evidence=evidence,
        )

    @classmethod
    def row(
        cls,
        *,
        address: str,
        relation: str,
        grain: dict[str, Any] | None = None,
        values: dict[str, FactAddressValue] | None = None,
        identity: dict[str, str] | None = None,
        rank: dict[str, Any] | None = None,
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.ROW,
            source_relation=relation,
            grain=dict(grain or {}),
            values=dict(values or {}),
            identity=dict(identity or {}),
            rank=dict(rank or {}),
            evidence=evidence,
        )

    @classmethod
    def outcome(
        cls,
        *,
        address: str,
        terminal: str,
        scope: dict[str, Any] | None = None,
        clarification_questions: tuple[str, ...] = (),
        proof: dict[str, Any] | None = None,
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.OUTCOME,
            terminal=terminal,
            scope=dict(scope or {}),
            clarification_questions=tuple(clarification_questions),
            proof=dict(proof or {}),
            evidence=evidence,
        )

    @classmethod
    def proof_address(
        cls,
        *,
        address: str,
        operation: str,
        subject: dict[str, Any],
        relations: dict[str, str],
        predicate: dict[str, Any] | None = None,
        completeness: dict[str, Any] | None = None,
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.PROOF,
            operation=operation,
            subject=dict(subject),
            relations=dict(relations),
            predicate=dict(predicate or {}),
            completeness=dict(completeness or {}),
            evidence=evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "address": self.address,
            "kind": self.kind.value,
            "resource": self.resource,
            "keyId": self.key_id,
            "referenceText": self.reference_text,
            "identity": dict(self.identity),
            "accessor": dict(self.accessor),
            "display": self.display,
            "value": _scalar_value_payload(self.scalar_value),
            "scope": dict(self.scope),
            "derivation": dict(self.derivation),
            "source": dict(self.source),
            "grainKeys": list(self.grain_keys),
            "fieldCoverage": dict(self.field_coverage),
            "completeness": dict(self.completeness),
            "rowAddresses": list(self.row_addresses),
            "relation": self.source_relation,
            "grain": dict(self.grain),
            "values": {
                field_id: value.to_dict() for field_id, value in self.values.items()
            },
            "rank": dict(self.rank),
            "terminal": self.terminal,
            "clarificationQuestions": list(self.clarification_questions),
            "proof": dict(self.proof),
            "operation": self.operation,
            "subject": dict(self.subject),
            "relations": dict(self.relations),
            "predicate": dict(self.predicate),
        }
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        return without_empty(payload)


def fact_address_from_payload(payload: Any) -> FactAddress:
    if not isinstance(payload, dict):
        raise ValueError("fact address payload must be an object")
    address = str(payload.get("address") or "").strip()
    kind = _address_kind(payload.get("kind"))
    if not address:
        raise ValueError("fact address requires address")
    if kind is None:
        raise ValueError("fact address requires valid kind")
    if kind == FactAddressKind.ENTITY:
        return FactAddress.entity(
            address=address,
            resource=str(payload.get("resource") or ""),
            key_id=str(payload.get("keyId") or ""),
            reference_text=str(payload.get("referenceText") or ""),
            identity={
                str(key): str(value)
                for key, value in (payload.get("identity") or {}).items()
            },
            accessor=dict(payload.get("accessor") or {}),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    if kind == FactAddressKind.VALUE:
        scalar_value = dict(payload.get("value") or {})
        if "value" in scalar_value:
            scalar_value["value"] = runtime_value_from_payload(scalar_value["value"])
        return FactAddress.value(
            address=address,
            value=scalar_value,
            display=str(payload.get("display") or ""),
            scope=dict(payload.get("scope") or {}),
            derivation=dict(payload.get("derivation") or {}),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    if kind == FactAddressKind.RELATION:
        return FactAddress.relation(
            address=address,
            source=dict(payload.get("source") or {}),
            scope=dict(payload.get("scope") or {}),
            grain_keys=tuple(str(item) for item in payload.get("grainKeys") or ()),
            field_coverage={
                str(key): str(value)
                for key, value in (payload.get("fieldCoverage") or {}).items()
            },
            completeness=dict(payload.get("completeness") or {}),
            row_addresses=tuple(
                str(item) for item in payload.get("rowAddresses") or ()
            ),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    if kind == FactAddressKind.ROW:
        raw_values = payload.get("values") or {}
        if not isinstance(raw_values, dict):
            raise ValueError("row fact address values must be an object")
        return FactAddress.row(
            address=address,
            relation=str(payload.get("relation") or ""),
            grain=dict(payload.get("grain") or {}),
            values={
                str(field_id): fact_address_value_from_payload(value)
                for field_id, value in raw_values.items()
            },
            identity={
                str(key): str(value)
                for key, value in (payload.get("identity") or {}).items()
            },
            rank=dict(payload.get("rank") or {}),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    if kind == FactAddressKind.OUTCOME:
        return FactAddress.outcome(
            address=address,
            terminal=str(payload.get("terminal") or ""),
            scope=dict(payload.get("scope") or {}),
            clarification_questions=tuple(
                str(item) for item in payload.get("clarificationQuestions") or ()
            ),
            proof=dict(payload.get("proof") or {}),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    if kind == FactAddressKind.PROOF:
        return FactAddress.proof_address(
            address=address,
            operation=str(payload.get("operation") or ""),
            subject=dict(payload.get("subject") or {}),
            relations={
                str(key): str(value)
                for key, value in (payload.get("relations") or {}).items()
            },
            predicate=dict(payload.get("predicate") or {}),
            completeness=dict(payload.get("completeness") or {}),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    raise ValueError("fact address requires supported kind")


def fact_address_value_from_payload(payload: object) -> FactAddressValue:
    if not isinstance(payload, dict):
        raise ValueError("fact address value must be an object")
    value_type = str(payload.get("type") or "").strip()
    if not value_type or "value" not in payload:
        raise ValueError("fact address value requires type and value")
    value = (
        _entity_key_value(payload.get("value"))
        if value_type == "entity_key"
        else runtime_value_from_payload(payload.get("value"))
    )
    raw_output_ids = payload.get("answer_output_ids") or ()
    if not isinstance(raw_output_ids, (list, tuple)):
        raise ValueError("fact address value answer_output_ids must be an array")
    return FactAddressValue(
        type=value_type,
        value=value,
        answer_output_ids=tuple(str(item) for item in raw_output_ids if str(item)),
    )


def _entity_key_value(payload: object) -> EntityKeyValue:
    if not isinstance(payload, dict):
        raise ValueError("entity-key fact address value must be an object")
    raw_components = payload.get("components")
    if not isinstance(raw_components, dict):
        raise ValueError("entity-key fact address value requires components")
    return EntityKeyValue(
        entity_kind=str(payload.get("entityKind") or "").strip(),
        key_id=str(payload.get("keyId") or "").strip(),
        components=tuple(
            EntityKeyComponentValue(
                component_id=str(component_id),
                value=runtime_value_from_payload(component_value),
            )
            for component_id, component_value in raw_components.items()
            if str(component_id)
        ),
    )


def _address_kind(value: Any) -> FactAddressKind | None:
    try:
        return FactAddressKind(str(value))
    except ValueError:
        return None


def _scalar_value_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if "value" in payload:
        payload["value"] = runtime_value_to_payload(payload["value"])
    return payload


def _validate_address_variant(address: FactAddress) -> None:
    if address.kind == FactAddressKind.ENTITY:
        if not address.resource or not address.key_id or not address.identity:
            raise ValueError("entity fact address requires a complete candidate key")
        return
    if address.kind == FactAddressKind.VALUE:
        if not address.scalar_value:
            raise ValueError("value fact address requires value")
        return
    if address.kind == FactAddressKind.RELATION:
        if not address.source:
            raise ValueError("relation fact address requires source")
        try:
            RelationSourceKind(str(address.source.get("kind") or ""))
        except ValueError as exc:
            raise ValueError(
                "relation fact address requires valid source.kind"
            ) from exc
        return
    if address.kind == FactAddressKind.ROW:
        if not address.source_relation:
            raise ValueError("row fact address requires relation")
        return
    if address.kind == FactAddressKind.OUTCOME:
        if not address.terminal:
            raise ValueError("outcome fact address requires terminal")
        if address.terminal not in _TERMINAL_OUTCOMES:
            raise ValueError("outcome fact address requires terminal outcome")
        return
    if address.kind == FactAddressKind.PROOF:
        if not address.operation:
            raise ValueError("proof fact address requires operation")
        return
