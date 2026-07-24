"""Public identity-route contract owned by Grounding."""

from __future__ import annotations

from dataclasses import dataclass
import re

from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    RowSourceParam,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.types.enums import StrEnum


class LookupTextResolutionDecision(StrEnum):
    CAN_RESOLVE_LOOKUP_TEXT = "CAN_RESOLVE_LOOKUP_TEXT"
    CANNOT_RESOLVE_LOOKUP_TEXT = "CANNOT_RESOLVE_LOOKUP_TEXT"


class IdentifierKind(StrEnum):
    PRIMARY_KEY = "PRIMARY_KEY"
    DESCRIPTIVE = "DESCRIPTIVE"


class InputBindingPurpose(StrEnum):
    IDENTITY_VALIDATION = "identity_validation"
    REFERENCE_GROUNDING = "reference_grounding"


class ResourceTypeMatch(StrEnum):
    POSSIBLE_DENOTED_KIND = "POSSIBLE_DENOTED_KIND"
    UNRELATED_RESOURCE_KIND = "UNRELATED_RESOURCE_KIND"


@dataclass(frozen=True)
class ExpectedInputIdentity:
    entity_kind: str
    key_id: str
    key_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.entity_kind or not self.key_id or not self.key_component_ids:
            raise ValueError(
                "expected input identity must name its complete candidate key"
            )


@dataclass(frozen=True)
class InputBindingKeyComponent:
    component_id: str
    field_id: str
    field_ref: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.field_id or not self.field_ref:
            raise ValueError("input binding key component is incomplete")


@dataclass(frozen=True)
class ResolverCandidate:
    known_input_id: str
    resolver_source: RowSource
    entity_kind: str
    key_id: str
    key_components: tuple[InputBindingKeyComponent, ...]
    resolver_resource_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.known_input_id
            or not self.resolver_source.id
            or not self.resolver_source.row_path_id
            or not self.resolver_source.read_id
            or not self.resolver_source.endpoint_name
            or not self.entity_kind
            or not self.key_id
            or not self.key_components
        ):
            raise ValueError("resolver candidate is incomplete")

    @property
    def resolver_read_id(self) -> str:
        return self.resolver_source.read_id

    @property
    def resolver_endpoint_name(self) -> str:
        return self.resolver_source.endpoint_name

    @property
    def identity_validation_request_parameters(self) -> tuple[RowSourceParam, ...]:
        candidate_components = {
            (self.entity_kind, self.key_id, component.component_id)
            for component in self.key_components
        }
        parameters = tuple(
            parameter
            for parameter in self.resolver_source.params
            if str(parameter.source) == "path"
            and parameter.entity_target is not None
            and (
                parameter.entity_target.entity_kind,
                parameter.entity_target.key_id,
                parameter.entity_target.component_id,
            )
            in candidate_components
        )
        parameter_targets = {
            (
                parameter.entity_target.entity_kind,
                parameter.entity_target.key_id,
                parameter.entity_target.component_id,
            )
            for parameter in parameters
            if parameter.entity_target is not None
        }
        return parameters if parameter_targets == candidate_components else ()


@dataclass(frozen=True)
class InputBindingOption:
    id: str
    known_input_id: str
    candidate: ResolverCandidate

    def __post_init__(self) -> None:
        if not self.id or not self.known_input_id:
            raise ValueError("input binding option is incomplete")
        if self.candidate.known_input_id != self.known_input_id:
            raise ValueError("input binding option and candidate have different owners")

    @property
    def purpose(self) -> InputBindingPurpose:
        if self.candidate.identity_validation_request_parameters:
            return InputBindingPurpose.IDENTITY_VALIDATION
        return InputBindingPurpose.REFERENCE_GROUNDING


def reference_binding_options(
    *,
    input_id: str,
    resolver_catalog: RelationCatalog,
    resolver_row_sources: RowSourceCatalog,
    expected_identity: ExpectedInputIdentity | None,
) -> tuple[InputBindingOption, ...]:
    options: list[InputBindingOption] = []
    for source in resolver_row_sources.sources:
        if source.kind is not RowSourceKind.API_READ:
            continue
        read = resolver_catalog.read(source.read_id)
        if read.endpoint_name != source.endpoint_name:
            raise ValueError("resolver row source disagrees with its catalog read")
        for key in source.candidate_keys:
            if not key.primary or not key.stable:
                continue
            component_ids = tuple(component.id for component in key.components)
            if expected_identity is not None and (
                key.entity_kind != expected_identity.entity_kind
                or key.id != expected_identity.key_id
                or component_ids != expected_identity.key_component_ids
            ):
                continue
            candidate = ResolverCandidate(
                known_input_id=input_id,
                resolver_source=source,
                resolver_resource_names=source.resource_names,
                entity_kind=key.entity_kind,
                key_id=key.id,
                key_components=tuple(
                    InputBindingKeyComponent(
                        component_id=component.id,
                        field_id=component.field_id,
                        field_ref=source.field(component.field_id).field_ref,
                    )
                    for component in key.components
                ),
            )
            options.append(
                InputBindingOption(
                    id=(
                        f"bind_{_symbol(input_id)}_{_symbol(source.id)}_"
                        f"{_symbol(key.entity_kind)}_{_symbol(key.id)}"
                    ),
                    known_input_id=input_id,
                    candidate=candidate,
                )
            )
    return tuple(options)


def _symbol(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return re.sub(r"_+", "_", text).strip("_").lower() or "value"


__all__ = [
    "ExpectedInputIdentity",
    "IdentifierKind",
    "InputBindingPurpose",
    "InputBindingKeyComponent",
    "InputBindingOption",
    "LookupTextResolutionDecision",
    "ResolverCandidate",
    "ResourceTypeMatch",
    "reference_binding_options",
]
