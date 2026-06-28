"""Generic capability contracts for planner validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CapabilityKind(StrEnum):
    ENTITY = "entity"
    METRIC = "metric"
    DIMENSION = "dimension"
    FILTER = "filter"
    AGGREGATION = "aggregation"
    IDENTIFIER = "identifier"
    FIELD = "field"


class CapabilitySource(StrEnum):
    EXPLICIT = "explicit"
    SCHEMA_GENERATED = "schema_generated"
    HOST_ADAPTER = "host_adapter"


@dataclass(frozen=True)
class RequiredCapability:
    kind: CapabilityKind | str
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CapabilityKind(str(self.kind)))
        object.__setattr__(self, "role", _normalize_role(self.role))

    def to_public_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "role": self.role}


@dataclass(frozen=True)
class EndpointCapability:
    kind: CapabilityKind | str
    role: str
    source: CapabilitySource | str = CapabilitySource.SCHEMA_GENERATED

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CapabilityKind(str(self.kind)))
        object.__setattr__(self, "role", _normalize_role(self.role))
        object.__setattr__(self, "source", CapabilitySource(str(self.source)))

    def satisfies(self, required: RequiredCapability) -> bool:
        return self.kind == required.kind and self.role == required.role

    def to_public_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "role": self.role,
            "source": self.source.value,
        }


@dataclass(frozen=True)
class EndpointCapabilities:
    items: tuple[EndpointCapability, ...] = field(default_factory=tuple)

    def has(self, kind: CapabilityKind | str, role: str) -> bool:
        required = RequiredCapability(kind, role)
        return any(item.satisfies(required) for item in self.items)

    def missing(
        self, required_capabilities: tuple[RequiredCapability, ...]
    ) -> tuple[RequiredCapability, ...]:
        return tuple(
            required
            for required in required_capabilities
            if not any(item.satisfies(required) for item in self.items)
        )

    def to_public_dict(self) -> list[dict[str, str]]:
        return [item.to_public_dict() for item in self.items]


def capabilities_from_schema(
    *,
    path_params: tuple[Any, ...],
    query_params: tuple[Any, ...],
    response_fields: tuple[Any, ...],
    primary_key_fields: tuple[str, ...] = (),
) -> EndpointCapabilities:
    items: list[EndpointCapability] = []
    primary_keys = {
        str(field or "").strip() for field in primary_key_fields if str(field).strip()
    }
    for param in (*path_params, *query_params):
        name = str(getattr(param, "name", "") or "")
        if not name:
            continue
        kind = (
            CapabilityKind.IDENTIFIER if name in primary_keys else CapabilityKind.FILTER
        )
        items.append(
            EndpointCapability(
                kind,
                name,
                source=CapabilitySource.SCHEMA_GENERATED,
            )
        )
    for field_contract in response_fields:
        name = str(getattr(field_contract, "name", "") or "")
        path = str(getattr(field_contract, "path", "") or "")
        for role in (name, path):
            if role:
                items.append(
                    EndpointCapability(
                        CapabilityKind.FIELD,
                        role,
                        source=CapabilitySource.SCHEMA_GENERATED,
                    )
                )
        if _field_has_identity(field_contract, primary_keys=primary_keys):
            items.append(
                EndpointCapability(
                    CapabilityKind.IDENTIFIER,
                    name,
                    source=CapabilitySource.SCHEMA_GENERATED,
                )
            )
    return EndpointCapabilities(tuple(dict.fromkeys(items)))


def _field_has_identity(field_contract: Any, *, primary_keys: set[str]) -> bool:
    identity = getattr(field_contract, "identity", None)
    if isinstance(identity, dict) and identity:
        return True
    name = str(getattr(field_contract, "name", "") or "").strip()
    path = str(getattr(field_contract, "path", "") or "").strip()
    leaf = path.rsplit(".", 1)[-1] if path else ""
    return name in primary_keys or leaf in primary_keys


def _normalize_role(value: str) -> str:
    return str(value).strip().lower()
