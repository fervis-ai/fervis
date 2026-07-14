"""Generic capability contracts for planner validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from typing import Protocol


class _NamedContract(Protocol):
    @property
    def name(self) -> str: ...


class _ResponseFieldContract(_NamedContract, Protocol):
    @property
    def path(self) -> str: ...


class _CandidateKeyComponent(Protocol):
    @property
    def field_path(self) -> str: ...


class _CandidateKey(Protocol):
    @property
    def components(self) -> tuple[_CandidateKeyComponent, ...]: ...


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
    kind: CapabilityKind
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _normalize_role(self.role))

    def to_public_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "role": self.role}


@dataclass(frozen=True)
class EndpointCapability:
    kind: CapabilityKind
    role: str
    source: CapabilitySource = CapabilitySource.SCHEMA_GENERATED

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _normalize_role(self.role))

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
        required = RequiredCapability(CapabilityKind(str(kind)), role)
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
    path_params: tuple[_NamedContract, ...],
    query_params: tuple[_NamedContract, ...],
    response_fields: tuple[_ResponseFieldContract, ...],
    candidate_keys: tuple[_CandidateKey, ...] = (),
) -> EndpointCapabilities:
    items: list[EndpointCapability] = []
    identifier_fields = _candidate_key_field_names(candidate_keys)
    for param in (*path_params, *query_params):
        name = str(getattr(param, "name", "") or "")
        if not name:
            continue
        kind = (
            CapabilityKind.IDENTIFIER
            if name in identifier_fields
            else CapabilityKind.FILTER
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
        if _field_is_identifier(field_contract, identifier_fields=identifier_fields):
            items.append(
                EndpointCapability(
                    CapabilityKind.IDENTIFIER,
                    name,
                    source=CapabilitySource.SCHEMA_GENERATED,
                )
            )
    return EndpointCapabilities(tuple(dict.fromkeys(items)))


def _candidate_key_field_names(
    candidate_keys: tuple[_CandidateKey, ...],
) -> set[str]:
    paths = {
        component.field_path.strip()
        for key in candidate_keys
        for component in key.components
        if component.field_path.strip()
    }
    return {*paths, *(path.rsplit(".", 1)[-1] for path in paths)}


def _field_is_identifier(
    field_contract: _ResponseFieldContract,
    *,
    identifier_fields: set[str],
) -> bool:
    name = field_contract.name.strip()
    path = field_contract.path.strip()
    leaf = path.rsplit(".", 1)[-1] if path else ""
    return (
        name in identifier_fields
        or path in identifier_fields
        or leaf in identifier_fields
    )


def _normalize_role(value: str) -> str:
    return str(value).strip().lower()
