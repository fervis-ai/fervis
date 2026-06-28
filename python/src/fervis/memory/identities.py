"""Public memory identity projection primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.memory._serialization import without_empty
from fervis.memory.addresses import FactAddress, FactAddressKind
from fervis.memory.artifacts import FactArtifact


@dataclass(frozen=True)
class MemoryIdentityValue:
    id: str
    identity_type: str
    identity_field: str
    value: str
    lookup_text: str = ""
    display_label: str = ""
    display_label_author: str = "backend"
    source: dict[str, Any] = field(default_factory=dict)
    proof_refs: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "identity"

    def to_dict(self) -> dict[str, Any]:
        return without_empty(
            {
                "id": self.id,
                "kind": self.kind,
                "identity_type": self.identity_type,
                "identity_field": self.identity_field,
                "value": self.value,
                "lookup_text": self.lookup_text,
                "display_label": self.display_label,
                "display_label_author": self.display_label_author,
                "source": dict(self.source),
                "proof_refs": list(self.proof_refs),
            }
        )


@dataclass(frozen=True)
class MemoryIdentitySet:
    id: str
    identity_type: str
    identity_field: str
    values: tuple[str, ...]
    count: int
    display_label: str
    display_label_author: str = "backend"
    source_relation_id: str = ""
    proof_refs: tuple[str, ...] = ()
    completeness: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "identity_set"

    def to_dict(self) -> dict[str, Any]:
        return without_empty(
            {
                "id": self.id,
                "kind": self.kind,
                "identity_type": self.identity_type,
                "identity_field": self.identity_field,
                "count": self.count,
                "display_label": self.display_label,
                "display_label_author": self.display_label_author,
                "source_relation_id": self.source_relation_id,
                "proof_refs": list(self.proof_refs),
                "completeness": dict(self.completeness),
            }
        )


@dataclass(frozen=True)
class MemoryIdentityProjection:
    identity_values: tuple[MemoryIdentityValue, ...] = ()
    identity_sets: tuple[MemoryIdentitySet, ...] = ()


def project_memory_identity_values(
    artifacts: tuple[FactArtifact, ...],
) -> MemoryIdentityProjection:
    values: list[MemoryIdentityValue] = []
    sets: list[MemoryIdentitySet] = []
    for artifact in artifacts:
        rows_by_relation = _rows_by_relation(artifact.addresses)
        relations_by_address = _relations_by_address(artifact.addresses)
        for address in artifact.addresses:
            if address.kind == FactAddressKind.ENTITY:
                values.extend(_entity_identity_values(artifact, address))
            if address.kind == FactAddressKind.ROW:
                values.extend(
                    _row_identity_values(
                        artifact,
                        address,
                        relation=relations_by_address.get(address.source_relation),
                    )
                )
            if address.kind == FactAddressKind.RELATION:
                sets.extend(
                    _relation_identity_sets(
                        artifact,
                        address,
                        rows=rows_by_relation.get(address.address, ()),
                    )
                )
    return MemoryIdentityProjection(
        identity_values=tuple(values),
        identity_sets=tuple(sets),
    )


def _entity_identity_values(
    artifact: FactArtifact,
    address: FactAddress,
) -> tuple[MemoryIdentityValue, ...]:
    if not address.resource:
        return ()
    proof_refs = tuple(address.evidence.step_ids if address.evidence else ())
    lookup_text = str(address.reference_text or "").strip()
    display_label = lookup_text or str(address.display or "").strip()
    return tuple(
        MemoryIdentityValue(
            id=f"mem.{artifact.artifact_id}.{address.address}.identity.{field_id}",
            identity_type=address.resource,
            identity_field=str(field_id),
            value=str(value),
            lookup_text=lookup_text,
            display_label=display_label or f"{address.resource} identity",
            display_label_author="backend",
            source={
                "artifact_id": artifact.artifact_id,
                "address": address.address,
                "field_id": str(field_id),
            },
            proof_refs=proof_refs,
        )
        for field_id, value in address.identity.items()
        if str(field_id).strip() and str(value).strip()
    )


def _row_identity_values(
    artifact: FactArtifact,
    address: FactAddress,
    *,
    relation: FactAddress | None,
) -> tuple[MemoryIdentityValue, ...]:
    if relation is None or not address.identity:
        return ()
    identity_type = _relation_identity_type(relation)
    if not identity_type:
        return ()
    proof_refs = tuple(address.evidence.step_ids if address.evidence else ())
    return tuple(
        MemoryIdentityValue(
            id=f"mem.{artifact.artifact_id}.{address.address}.identity.{field_id}",
            identity_type=identity_type,
            identity_field=str(field_id),
            value=str(value),
            display_label=f"{identity_type} identity from prior answer",
            display_label_author="backend",
            source={
                "artifact_id": artifact.artifact_id,
                "address": address.address,
                "field_id": str(field_id),
            },
            proof_refs=proof_refs,
        )
        for field_id, value in address.identity.items()
        if str(field_id).strip() and str(value).strip()
    )


def _relation_identity_sets(
    artifact: FactArtifact,
    address: FactAddress,
    *,
    rows: tuple[FactAddress, ...],
) -> tuple[MemoryIdentitySet, ...]:
    identity_type = _relation_identity_type(address)
    if not identity_type or not _relation_is_complete(address):
        return ()
    proof_refs = tuple(address.evidence.step_ids if address.evidence else ())
    output: list[MemoryIdentitySet] = []
    for identity_field in _relation_identity_fields(address, rows=rows):
        values = tuple(
            str(row.identity.get(identity_field) or "")
            for row in rows
            if str(row.identity.get(identity_field) or "").strip()
        )
        if len(values) != len(rows):
            continue
        count = len(values)
        output.append(
            MemoryIdentitySet(
                id=(
                    f"mem.{artifact.artifact_id}.{address.address}."
                    f"identity_set.{identity_field}"
                ),
                identity_type=identity_type,
                identity_field=identity_field,
                values=values,
                count=count,
                display_label=_identity_set_display_label(
                    identity_type=identity_type,
                    count=count,
                ),
                display_label_author="backend",
                source_relation_id=f"{artifact.artifact_id}.{address.address}",
                proof_refs=proof_refs,
                completeness=_identity_set_completeness(address, count=count),
            )
        )
    return tuple(output)


def _relations_by_address(
    addresses: tuple[FactAddress, ...],
) -> dict[str, FactAddress]:
    return {
        address.address: address
        for address in addresses
        if address.kind == FactAddressKind.RELATION
    }


def _rows_by_relation(
    addresses: tuple[FactAddress, ...],
) -> dict[str, tuple[FactAddress, ...]]:
    rows_by_address = {
        address.address: address
        for address in addresses
        if address.kind == FactAddressKind.ROW
    }
    output: dict[str, tuple[FactAddress, ...]] = {}
    for relation in (
        address for address in addresses if address.kind == FactAddressKind.RELATION
    ):
        rows = tuple(
            rows_by_address[row_address]
            for row_address in relation.row_addresses
            if row_address in rows_by_address
            and rows_by_address[row_address].source_relation == relation.address
        )
        output[relation.address] = rows
    return output


def _relation_identity_type(address: FactAddress) -> str:
    return str(address.source.get("identityType") or "").strip()


def _relation_identity_fields(
    address: FactAddress,
    *,
    rows: tuple[FactAddress, ...],
) -> tuple[str, ...]:
    if not rows:
        return ()
    fields = tuple(
        str(field_id)
        for field_id in address.grain_keys
        if all(str(row.identity.get(str(field_id)) or "").strip() for row in rows)
    )
    if fields:
        return fields
    return tuple(
        field_id
        for field_id in rows[0].identity
        if all(str(row.identity.get(field_id) or "").strip() for row in rows)
    )


def _relation_is_complete(address: FactAddress) -> bool:
    completeness = dict(address.completeness or {})
    status = str(completeness.get("status") or "").strip()
    pagination = str(completeness.get("pagination") or "").strip()
    return status == "complete" and pagination in {
        "all_pages",
        "not_paginated",
        "terminal",
    }


def _identity_set_completeness(
    address: FactAddress,
    *,
    count: int,
) -> dict[str, Any]:
    completeness = dict(address.completeness or {})
    return without_empty(
        {
            "status": str(completeness.get("status") or ""),
            "pagination": str(completeness.get("pagination") or ""),
            "row_count": completeness.get("rowCount", count),
        }
    )


def _identity_set_display_label(*, identity_type: str, count: int) -> str:
    noun = "identity" if count == 1 else "identities"
    return f"{count} {identity_type} {noun} from prior answer"
