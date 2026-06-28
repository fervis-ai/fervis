"""Framework-neutral relation catalog model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ParamSource(StrEnum):
    PATH = "path"
    QUERY = "query"
    HEADER = "header"


class RowCardinality(StrEnum):
    ONE = "one"
    MANY = "many"


class PaginationMode(StrEnum):
    NONE = "none"
    PAGE_NUMBER = "page_number"
    CURSOR = "cursor"
    LIMIT_OFFSET = "limit_offset"


class CompletenessPolicy(StrEnum):
    COMPLETE = "complete"
    ALL_PAGES = "all_pages"
    BOUNDED = "bounded"
    INCOMPLETE = "incomplete"


class CatalogFactAvailability(StrEnum):
    AVAILABLE = "available"
    NOT_READABLE = "not_readable"
    POLICY_BLOCKED = "policy_blocked"
    NOT_COLLECTED = "not_collected"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class IdentityMetadata:
    entity_ref: str = ""
    identity_field: str = ""
    primary_key: bool = False
    stable: bool = True
    display_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldRequirement:
    param_ref: str
    value: Any


@dataclass(frozen=True)
class CatalogParam:
    ref: str
    name: str
    source: ParamSource
    type: str
    description: str = ""
    required: bool = False
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] | None = None
    default: Any = None
    identity: IdentityMetadata | None = None
    semantics: str = ""


@dataclass(frozen=True)
class RowPath:
    id: str
    path: str
    cardinality: RowCardinality
    parent_path: str = ""


@dataclass(frozen=True)
class ResponseEnvelopeMetadata:
    results_path: str = ""
    count_path: str = ""
    has_more_path: str = ""
    next_path: str = ""


@dataclass(frozen=True)
class PaginationMetadata:
    mode: PaginationMode = PaginationMode.NONE
    default_page_size: int = 0
    max_page_size: int = 0
    completeness_policy: CompletenessPolicy = CompletenessPolicy.COMPLETE


@dataclass(frozen=True)
class CatalogField:
    ref: str
    type: str
    path: str = ""
    row_path_id: str = ""
    nullable: bool = False
    choices: tuple[str, ...] = ()
    identity: IdentityMetadata | None = None
    requirements: tuple[FieldRequirement, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CatalogFact:
    ref: str
    availability: CatalogFactAvailability = CatalogFactAvailability.AVAILABLE
    field_ref: str = ""
    read_id: str = ""
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogEndpointMetadata:
    catalog_endpoint_key: str
    endpoint_name: str
    framework_kind: str
    source_namespace_kind: str
    source_namespace_path: tuple[str, ...]
    route_method: str
    route_path_template: str
    handler_ref: str
    route_name: str = ""
    api_schema_operation_id: str = ""
    domain_resource_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class EndpointRead:
    id: str
    endpoint_name: str
    method: str = "GET"
    path: str = ""
    resource_names: tuple[str, ...] = ()
    params: tuple[CatalogParam, ...] = ()
    row_paths: tuple[RowPath, ...] = ()
    fields: tuple[CatalogField, ...] = ()
    facts: tuple[CatalogFact, ...] = ()
    response_envelope: ResponseEnvelopeMetadata = ResponseEnvelopeMetadata()
    pagination: PaginationMetadata | None = PaginationMetadata()
    access: tuple[str, ...] = ()
    catalog_endpoint: CatalogEndpointMetadata | None = None
    source_metadata: dict[str, Any] | None = None

    @property
    def fields_by_path(self) -> dict[str, CatalogField]:
        return {item.path: item for item in self.fields}


@dataclass(frozen=True)
class RelationCatalog:
    reads: tuple[EndpointRead, ...] = ()
    facts: tuple[CatalogFact, ...] = ()

    def read(self, read_id: str) -> EndpointRead:
        for item in self.reads:
            if item.id == read_id:
                return item
        raise KeyError(read_id)
