"""Current-turn activation of prior fact addresses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fervis.memory._serialization import without_empty
from fervis.memory.addresses import FactAddress, FactAddressKind
from fervis.memory.artifacts import FactArtifact


class UseAs(StrEnum):
    IDENTITY_VALUE_BINDING = "identity_value_binding"
    IDENTITY_SET_BINDING = "identity_set_binding"
    RELATION_ROWS_SOURCE = "relation_rows_source"
    SAME_SCOPE_SOURCE = "same_scope_source"
    SCALAR_VALUE_INPUT = "scalar_value_input"
    CLARIFICATION_FILL = "clarification_fill"


_COMPATIBLE_USES: dict[FactAddressKind, frozenset[UseAs]] = {
    FactAddressKind.ENTITY: frozenset({UseAs.IDENTITY_VALUE_BINDING}),
    FactAddressKind.ROW: frozenset({UseAs.IDENTITY_VALUE_BINDING}),
    FactAddressKind.RELATION: frozenset(
        {
            UseAs.RELATION_ROWS_SOURCE,
            UseAs.SAME_SCOPE_SOURCE,
            UseAs.IDENTITY_SET_BINDING,
        }
    ),
    FactAddressKind.VALUE: frozenset({UseAs.SCALAR_VALUE_INPUT}),
    FactAddressKind.OUTCOME: frozenset({UseAs.CLARIFICATION_FILL}),
}


@dataclass(frozen=True)
class ActivatedInput:
    id: str
    from_artifact_id: str
    from_address: str
    use_as: UseAs
    target_binding_id: str
    requested_scope: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return without_empty(
            {
                "id": self.id,
                "from": {
                    "artifactId": self.from_artifact_id,
                    "address": self.from_address,
                },
                "useAs": self.use_as.value,
                "targetBindingId": self.target_binding_id,
                "requestedScope": dict(self.requested_scope),
            }
        )


@dataclass(frozen=True)
class ActivatedMemory:
    inputs: tuple[ActivatedInput, ...] = ()
    entries: tuple["ActivatedAddress", ...] = ()

    def address(self, address: str) -> FactAddress | None:
        for entry in self.entries:
            if entry.address.address == address:
                return entry.address
        return None

    @property
    def addresses(self) -> tuple[FactAddress, ...]:
        return tuple(entry.address for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return without_empty(
            {
                "activatedInputs": [item.to_dict() for item in self.inputs],
                "addresses": [item.to_dict() for item in self.entries],
            }
        )


@dataclass(frozen=True)
class ActivatedAddress:
    artifact_id: str
    address: FactAddress

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifactId": self.artifact_id,
            **self.address.to_dict(),
        }


def activate_memory(
    *,
    artifacts: tuple[FactArtifact, ...],
    requests: tuple[ActivatedInput, ...],
) -> ActivatedMemory:
    artifacts_by_id = {item.artifact_id: item for item in artifacts}
    entries: list[ActivatedAddress] = []
    seen_addresses: set[tuple[str, str]] = set()
    for request in requests:
        if not request.id:
            raise ValueError("activated input requires id")
        if not request.target_binding_id:
            raise ValueError("activated input requires target_binding_id")
        artifact = artifacts_by_id.get(request.from_artifact_id)
        if artifact is None:
            raise ValueError(
                f"activated input {request.id} references unknown artifact"
            )
        address = artifact.address(request.from_address)
        if address is None:
            raise ValueError(
                f"activated input {request.id} references unknown fact address"
            )
        if request.use_as not in _COMPATIBLE_USES.get(address.kind, frozenset()):
            raise ValueError(
                f"activated input {request.id} uses {address.kind.value} as "
                f"{request.use_as.value}"
            )
        if request.use_as == UseAs.IDENTITY_SET_BINDING:
            _validate_identity_set_address(request, address)
        key = (artifact.artifact_id, address.address)
        if key not in seen_addresses:
            seen_addresses.add(key)
            entries.append(
                ActivatedAddress(
                    artifact_id=artifact.artifact_id,
                    address=address,
                )
            )
    return ActivatedMemory(inputs=tuple(requests), entries=tuple(entries))


def activated_entity_id_rows(memory: ActivatedMemory) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    address_by_id = {
        (entry.artifact_id, entry.address.address): entry.address
        for entry in memory.entries
    }
    for activated_input in memory.inputs:
        if activated_input.use_as != UseAs.IDENTITY_VALUE_BINDING:
            continue
        address = address_by_id.get(
            (activated_input.from_artifact_id, activated_input.from_address)
        )
        if address is None or address.kind != FactAddressKind.ENTITY:
            continue
        for id_field, id_value in address.identity.items():
            rows.append(
                {
                    "entityType": address.resource,
                    "referenceText": address.reference_text,
                    "idField": str(id_field),
                    "idValue": str(id_value),
                }
            )
    return tuple(rows)


def _validate_identity_set_address(
    request: ActivatedInput,
    address: FactAddress,
) -> None:
    completeness = dict(address.completeness or {})
    if completeness.get("status") != "complete":
        raise ValueError(
            f"activated input {request.id} requires complete relation identity set"
        )
    if completeness.get("pagination") not in {
        "all_pages",
        "not_paginated",
        "terminal",
    }:
        raise ValueError(
            f"activated input {request.id} requires complete relation pagination"
        )
