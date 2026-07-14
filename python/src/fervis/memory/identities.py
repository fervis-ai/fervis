"""Public memory identity projection primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.memory._serialization import without_empty
from fervis.memory.addresses import FactAddress, FactAddressKind
from fervis.memory.artifacts import FactArtifact
from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue


@dataclass(frozen=True)
class MemoryIdentityValue:
    id: str
    value: EntityKeyValue
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
                "value": _entity_key_payload(self.value),
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
    keys: tuple[EntityKeyValue, ...]
    display_label: str
    display_label_author: str = "backend"
    source_relation_id: str = ""
    proof_refs: tuple[str, ...] = ()
    completeness: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "identity_set"

    @property
    def entity_kind(self) -> str:
        return self.keys[0].entity_kind

    @property
    def key_id(self) -> str:
        return self.keys[0].key_id

    @property
    def count(self) -> int:
        return len(self.keys)

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("memory identity set requires entity keys")
        key_contracts = {
            (key.entity_kind, key.key_id, _component_ids(key)) for key in self.keys
        }
        if len(key_contracts) != 1:
            raise ValueError("memory identity set mixes candidate-key contracts")

    def to_dict(self) -> dict[str, Any]:
        return without_empty(
            {
                "id": self.id,
                "kind": self.kind,
                "entity_kind": self.entity_kind,
                "key_id": self.key_id,
                "keys": [_entity_key_payload(key) for key in self.keys],
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


@dataclass(frozen=True)
class _RowEntityKey:
    output_id: str
    value: EntityKeyValue


def project_memory_identity_values(
    artifacts: tuple[FactArtifact, ...],
) -> MemoryIdentityProjection:
    values: list[MemoryIdentityValue] = []
    sets: list[MemoryIdentitySet] = []
    for artifact in artifacts:
        rows_by_relation = _rows_by_relation(artifact.addresses)
        for address in artifact.addresses:
            if address.kind == FactAddressKind.ENTITY:
                values.extend(_entity_identity_values(artifact, address))
            if address.kind == FactAddressKind.ROW:
                values.extend(
                    _row_entity_key_values(
                        artifact,
                        address,
                    )
                )
            if address.kind == FactAddressKind.RELATION:
                sets.extend(
                    _relation_entity_key_sets(
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
    components = tuple(
        EntityKeyComponentValue(component_id=str(field_id), value=value)
        for field_id, value in address.identity.items()
        if str(field_id).strip() and str(value).strip()
    )
    if not components:
        return ()
    value = EntityKeyValue(
        entity_kind=address.resource,
        key_id=address.key_id,
        components=components,
    )
    memory_value = MemoryIdentityValue(
        id=f"mem.{artifact.artifact_id}.{address.address}.identity",
        value=value,
        lookup_text=lookup_text,
        display_label=display_label or f"{address.resource} identity",
        display_label_author="backend",
        source={
            "artifact_id": artifact.artifact_id,
            "address": address.address,
        },
        proof_refs=proof_refs,
    )
    return (memory_value,)


def _row_entity_key_values(
    artifact: FactArtifact,
    address: FactAddress,
) -> tuple[MemoryIdentityValue, ...]:
    proof_refs = tuple(address.evidence.step_ids if address.evidence else ())
    return tuple(
        MemoryIdentityValue(
            id=(
                f"mem.{artifact.artifact_id}.{address.address}."
                f"entity_key.{key.output_id}"
            ),
            value=key.value,
            display_label=f"{key.value.entity_kind} identity from prior answer",
            display_label_author="backend",
            source={
                "artifact_id": artifact.artifact_id,
                "address": address.address,
                "answer_output_id": key.output_id,
            },
            proof_refs=proof_refs,
        )
        for key in _row_entity_keys(address)
    )


def _relation_entity_key_sets(
    artifact: FactArtifact,
    address: FactAddress,
    *,
    rows: tuple[FactAddress, ...],
) -> tuple[MemoryIdentitySet, ...]:
    if not rows or not _relation_is_complete(address):
        return ()
    proof_refs = tuple(address.evidence.step_ids if address.evidence else ())
    output: list[MemoryIdentitySet] = []
    for first_key in _row_entity_keys(rows[0]):
        keys = _matching_row_entity_keys(rows, template=first_key)
        if not keys:
            continue
        count = len(keys)
        output.append(
            MemoryIdentitySet(
                id=(
                    f"mem.{artifact.artifact_id}.{address.address}."
                    f"entity_key_set.{first_key.output_id}"
                ),
                keys=keys,
                display_label=_identity_set_display_label(
                    entity_kind=first_key.value.entity_kind,
                    count=count,
                ),
                display_label_author="backend",
                source_relation_id=f"{artifact.artifact_id}.{address.address}",
                proof_refs=proof_refs,
                completeness=_identity_set_completeness(address, count=count),
            )
        )
    return tuple(output)


def _row_entity_keys(address: FactAddress) -> tuple[_RowEntityKey, ...]:
    return tuple(
        _RowEntityKey(
            output_id=output_id,
            value=value.value,
        )
        for output_id, value in address.values.items()
        if isinstance(value.value, EntityKeyValue)
    )


def _matching_row_entity_keys(
    rows: tuple[FactAddress, ...],
    *,
    template: _RowEntityKey,
) -> tuple[EntityKeyValue, ...]:
    keys: list[EntityKeyValue] = []
    for row in rows:
        matches = tuple(
            key.value
            for key in _row_entity_keys(row)
            if key.output_id == template.output_id
            and _same_key_contract(key.value, template.value)
        )
        if len(matches) != 1:
            return ()
        keys.append(matches[0])
    return tuple(keys)


def _same_key_contract(left: EntityKeyValue, right: EntityKeyValue) -> bool:
    return (
        left.entity_kind == right.entity_kind
        and left.key_id == right.key_id
        and _component_ids(left) == _component_ids(right)
    )


def _component_ids(value: EntityKeyValue) -> tuple[str, ...]:
    return tuple(component.component_id for component in value.components)


def _entity_key_payload(value: EntityKeyValue) -> dict[str, Any]:
    return {
        "entity_kind": value.entity_kind,
        "key_id": value.key_id,
        "components": value.component_values(),
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


def _relation_is_complete(address: FactAddress) -> bool:
    completeness = dict(address.completeness or {})
    if completeness.get("truncated") is True:
        return False
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


def _identity_set_display_label(*, entity_kind: str, count: int) -> str:
    noun = "identity" if count == 1 else "identities"
    return f"{count} {entity_kind} {noun} from prior answer"
