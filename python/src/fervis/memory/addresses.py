"""Typed fact addresses for fervis memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fervis.memory._serialization import without_empty


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
        "needs_clarification",
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
class FactAddress:
    address: str
    kind: FactAddressKind
    resource: str = ""
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
    values: dict[str, Any] = field(default_factory=dict)
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
        reference_text: str,
        identity: dict[str, str],
        accessor: dict[str, Any] | None = None,
        evidence: EvidenceRef | None = None,
    ) -> "FactAddress":
        return cls(
            address=address,
            kind=FactAddressKind.ENTITY,
            resource=resource,
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
        values: dict[str, Any] | None = None,
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
            "referenceText": self.reference_text,
            "identity": dict(self.identity),
            "accessor": dict(self.accessor),
            "display": self.display,
            "value": dict(self.scalar_value),
            "scope": dict(self.scope),
            "derivation": dict(self.derivation),
            "source": dict(self.source),
            "grainKeys": list(self.grain_keys),
            "fieldCoverage": dict(self.field_coverage),
            "completeness": dict(self.completeness),
            "rowAddresses": list(self.row_addresses),
            "relation": self.source_relation,
            "grain": dict(self.grain),
            "values": dict(self.values),
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
            reference_text=str(payload.get("referenceText") or ""),
            identity={
                str(key): str(value)
                for key, value in (payload.get("identity") or {}).items()
            },
            accessor=dict(payload.get("accessor") or {}),
            evidence=evidence_ref_from_payload(payload.get("evidence")),
        )
    if kind == FactAddressKind.VALUE:
        return FactAddress.value(
            address=address,
            value=dict(payload.get("value") or {}),
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
        return FactAddress.row(
            address=address,
            relation=str(payload.get("relation") or ""),
            grain=dict(payload.get("grain") or {}),
            values=dict(payload.get("values") or {}),
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


def _address_kind(value: Any) -> FactAddressKind | None:
    try:
        return FactAddressKind(str(value))
    except ValueError:
        return None


def _validate_address_variant(address: FactAddress) -> None:
    if address.kind == FactAddressKind.ENTITY:
        if not address.resource or not address.identity:
            raise ValueError("entity fact address requires resource and identity")
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
