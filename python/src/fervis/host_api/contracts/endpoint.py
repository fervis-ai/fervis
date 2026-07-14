"""DTOs for public GET endpoint contracts exposed to the fervis."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fervis.types.enums import StrEnum
from uuid import NAMESPACE_URL, uuid5

from fervis.host_api.contracts.capabilities import EndpointCapabilities
from fervis.host_api.contracts.pagination import PaginationContract
from fervis.host_api.contracts.values import ContractValue


def _public_values(values: Iterable[ContractValue]) -> list[ContractValue]:
    return list(values)


class FrameworkKind(StrEnum):
    DJANGO_DRF = "django_drf"
    FASTAPI = "fastapi"
    FLASK = "flask"


class SourceNamespaceKind(StrEnum):
    DJANGO_APP = "django_app"
    FASTAPI_APP = "fastapi_app"
    FLASK_BLUEPRINT = "flask_blueprint"
    PYTHON_MODULE = "python_module"


def make_catalog_endpoint_key(endpoint: "EndpointContract") -> str:
    """Human-readable deterministic key for the endpoint contract snapshot."""

    catalog_endpoint = endpoint.catalog_endpoint
    if catalog_endpoint is None:
        raise ValueError("endpoint contract is missing catalog endpoint metadata")
    payload = _catalog_endpoint_key_payload(endpoint)
    digest = uuid5(
        NAMESPACE_URL,
        f"fervis:catalog_endpoint:{payload}",
    ).hex[:12]
    namespace = (
        catalog_endpoint.source_namespace_path[-1]
        if catalog_endpoint.source_namespace_path
        else "root"
    )
    readable = "_".join(
        _catalog_endpoint_key_segment(item)
        for item in (
            catalog_endpoint.framework_kind,
            namespace,
            endpoint.endpoint_name,
        )
        if item
    )
    return f"{readable[:96]}:{digest}"


def _catalog_endpoint_key_payload(endpoint: "EndpointContract") -> str:
    catalog_endpoint = endpoint.catalog_endpoint
    if catalog_endpoint is None:
        raise ValueError("endpoint contract is missing catalog endpoint metadata")
    payload = {
        "endpoint_name": endpoint.endpoint_name,
        "framework_kind": catalog_endpoint.framework_kind,
        "handler_ref": catalog_endpoint.handler_ref,
        "route_method": endpoint.method.upper(),
        "route_path_template": endpoint.path_template,
        "source_namespace_kind": catalog_endpoint.source_namespace_kind,
        "source_namespace_path": list(catalog_endpoint.source_namespace_path),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _catalog_endpoint_key_segment(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


@dataclass(frozen=True)
class EntityKeyComponentTargetContract:
    entity_kind: str
    key_id: str
    component_id: str

    def __post_init__(self) -> None:
        if not self.entity_kind or not self.key_id or not self.component_id:
            raise ValueError("entity key component target is incomplete")

    def to_public_dict(self) -> dict[str, ContractValue]:
        return {
            "entityKind": self.entity_kind,
            "keyId": self.key_id,
            "componentId": self.component_id,
        }


@dataclass(frozen=True)
class ParameterContract:
    name: str
    type: str
    required: bool = False
    description: str = ""
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] = field(default_factory=dict)
    default: ContractValue = None
    source: str = "query"
    entity_target: EntityKeyComponentTargetContract | None = None
    semantics: str = ""

    def to_public_dict(self) -> dict[str, ContractValue]:
        payload: dict[str, ContractValue] = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "source": self.source,
        }
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.choice_labels:
            payload["choiceLabels"] = dict(self.choice_labels)
        if self.default is not None:
            payload["default"] = self.default
        if self.entity_target is not None:
            payload["entityTarget"] = self.entity_target.to_public_dict()
        if self.semantics:
            payload["semantics"] = self.semantics
        return payload


@dataclass(frozen=True)
class ResponseFieldContract:
    name: str
    type: str
    path: str
    description: str = ""
    choices: tuple[str, ...] = ()
    requires: dict[str, ContractValue] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, ContractValue]:
        payload: dict[str, ContractValue] = {
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "description": self.description,
        }
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.requires:
            payload["requires"] = dict(self.requires)
        return payload


@dataclass(frozen=True)
class CatalogEndpointContract:
    framework_kind: str
    source_namespace_kind: str
    source_namespace_path: tuple[str, ...]
    handler_ref: str
    route_name: str = ""
    api_schema_operation_id: str = ""
    domain_resource_names: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, ContractValue]:
        payload: dict[str, ContractValue] = {
            "frameworkKind": self.framework_kind,
            "sourceNamespaceKind": self.source_namespace_kind,
            "sourceNamespacePath": list(self.source_namespace_path),
            "handlerRef": self.handler_ref,
        }
        if self.route_name:
            payload["routeName"] = self.route_name
        if self.api_schema_operation_id:
            payload["apiSchemaOperationId"] = self.api_schema_operation_id
        if self.domain_resource_names:
            payload["domainResourceNames"] = list(self.domain_resource_names)
        return payload


@dataclass(frozen=True)
class CandidateKeyComponentContract:
    component_id: str
    field_path: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.field_path:
            raise ValueError("candidate key component requires id and field")

    def to_public_dict(self) -> dict[str, ContractValue]:
        return {
            "componentId": self.component_id,
            "fieldPath": self.field_path,
        }


@dataclass(frozen=True)
class CandidateKeyContract:
    key_id: str
    entity_kind: str
    components: tuple[CandidateKeyComponentContract, ...]
    primary: bool = False
    stable: bool = True
    context_field_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key_id or not self.entity_kind or not self.components:
            raise ValueError("candidate key requires id, entity kind, and fields")
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("candidate key component ids must be unique")

    def to_public_dict(self) -> dict[str, ContractValue]:
        components = _public_values(item.to_public_dict() for item in self.components)
        return {
            "keyId": self.key_id,
            "entityKind": self.entity_kind,
            "components": components,
            "primary": self.primary,
            "stable": self.stable,
            "contextFieldPaths": list(self.context_field_paths),
        }


@dataclass(frozen=True)
class CandidateKeyAuthorityComponentContract:
    component_id: str
    type: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.type:
            raise ValueError("candidate key authority component is incomplete")

    def to_public_dict(self) -> dict[str, ContractValue]:
        return {"componentId": self.component_id, "type": self.type}


@dataclass(frozen=True)
class CandidateKeyAuthorityContract:
    key_id: str
    entity_kind: str
    components: tuple[CandidateKeyAuthorityComponentContract, ...]
    primary: bool = False
    stable: bool = True

    def __post_init__(self) -> None:
        if not self.key_id or not self.entity_kind or not self.components:
            raise ValueError("candidate key authority is incomplete")
        component_ids = tuple(component.component_id for component in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("candidate key authority component ids must be unique")

    def to_public_dict(self) -> dict[str, ContractValue]:
        components = _public_values(item.to_public_dict() for item in self.components)
        return {
            "keyId": self.key_id,
            "entityKind": self.entity_kind,
            "components": components,
            "primary": self.primary,
            "stable": self.stable,
        }


@dataclass(frozen=True)
class EntityReferenceComponentContract:
    target_component_id: str
    local_field_path: str

    def __post_init__(self) -> None:
        if not self.target_component_id or not self.local_field_path:
            raise ValueError("entity reference component requires target and field")

    def to_public_dict(self) -> dict[str, ContractValue]:
        return {
            "targetComponentId": self.target_component_id,
            "localFieldPath": self.local_field_path,
        }


@dataclass(frozen=True)
class EntityReferenceContract:
    reference_id: str
    target_entity_kind: str
    target_key_id: str
    components: tuple[EntityReferenceComponentContract, ...]
    context_field_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.reference_id
            or not self.target_entity_kind
            or not self.target_key_id
        ):
            raise ValueError("entity reference requires id, entity kind, and key")
        if not self.components:
            raise ValueError("entity reference requires components")
        component_ids = tuple(item.target_component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("entity reference target components must be unique")

    def to_public_dict(self) -> dict[str, ContractValue]:
        components = _public_values(item.to_public_dict() for item in self.components)
        return {
            "referenceId": self.reference_id,
            "targetEntityKind": self.target_entity_kind,
            "targetKeyId": self.target_key_id,
            "components": components,
            "contextFieldPaths": list(self.context_field_paths),
        }


def relation_metadata_from_public_value(
    value: ContractValue,
) -> tuple[tuple[CandidateKeyContract, ...], tuple[EntityReferenceContract, ...]]:
    payload = _public_mapping(value, label="x-fervis")
    candidate_keys = tuple(
        _candidate_key_from_public_value(item)
        for item in _public_list(
            payload.get("candidateKeys", []), label="candidateKeys"
        )
    )
    entity_references = tuple(
        _entity_reference_from_public_value(item)
        for item in _public_list(
            payload.get("entityReferences", []),
            label="entityReferences",
        )
    )
    return candidate_keys, entity_references


def _candidate_key_from_public_value(value: ContractValue) -> CandidateKeyContract:
    payload = _public_mapping(value, label="candidate key")
    components = tuple(
        _candidate_key_component_from_public_value(item)
        for item in _public_list(payload.get("components"), label="components")
    )
    return CandidateKeyContract(
        key_id=_public_text(payload.get("keyId"), label="keyId"),
        entity_kind=_public_text(payload.get("entityKind"), label="entityKind"),
        components=components,
        primary=_public_bool(payload.get("primary", False), label="primary"),
        stable=_public_bool(payload.get("stable", True), label="stable"),
        context_field_paths=_public_texts(
            payload.get("contextFieldPaths", []),
            label="contextFieldPaths",
        ),
    )


def _candidate_key_component_from_public_value(
    value: ContractValue,
) -> CandidateKeyComponentContract:
    payload = _public_mapping(value, label="candidate key component")
    return CandidateKeyComponentContract(
        component_id=_public_text(payload.get("componentId"), label="componentId"),
        field_path=_public_text(payload.get("fieldPath"), label="fieldPath"),
    )


def _entity_reference_from_public_value(
    value: ContractValue,
) -> EntityReferenceContract:
    payload = _public_mapping(value, label="entity reference")
    components = tuple(
        _entity_reference_component_from_public_value(item)
        for item in _public_list(payload.get("components"), label="components")
    )
    return EntityReferenceContract(
        reference_id=_public_text(payload.get("referenceId"), label="referenceId"),
        target_entity_kind=_public_text(
            payload.get("targetEntityKind"),
            label="targetEntityKind",
        ),
        target_key_id=_public_text(payload.get("targetKeyId"), label="targetKeyId"),
        components=components,
        context_field_paths=_public_texts(
            payload.get("contextFieldPaths", []),
            label="contextFieldPaths",
        ),
    )


def _entity_reference_component_from_public_value(
    value: ContractValue,
) -> EntityReferenceComponentContract:
    payload = _public_mapping(value, label="entity reference component")
    return EntityReferenceComponentContract(
        target_component_id=_public_text(
            payload.get("targetComponentId"),
            label="targetComponentId",
        ),
        local_field_path=_public_text(
            payload.get("localFieldPath"),
            label="localFieldPath",
        ),
    )


def _public_mapping(
    value: ContractValue,
    *,
    label: str,
) -> dict[str, ContractValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _public_list(value: ContractValue, *, label: str) -> list[ContractValue]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _public_text(value: ContractValue, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _public_texts(value: ContractValue, *, label: str) -> tuple[str, ...]:
    return tuple(
        _public_text(item, label=label) for item in _public_list(value, label=label)
    )


def _public_bool(value: ContractValue, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True)
class EndpointContract:
    endpoint_name: str
    url_name: str
    method: str
    path_template: str
    docstring: str
    view_class: str
    path_params: tuple[ParameterContract, ...] = ()
    query_params: tuple[ParameterContract, ...] = ()
    response_fields: tuple[ResponseFieldContract, ...] = ()
    response_schema: Mapping[str, ContractValue] = field(default_factory=dict)
    capabilities: EndpointCapabilities = field(default_factory=EndpointCapabilities)
    capability_sources: tuple[str, ...] = ()
    agent_access: bool = False
    staff_access: bool = False
    admin_access: bool = True
    public_access: bool = False
    pagination: PaginationContract | None = None
    query_schema_source: str = "missing"
    response_schema_source: str = "missing"
    response_cardinality: str = "one"
    tags: tuple[str, ...] = field(default_factory=tuple)
    resource_names: tuple[str, ...] = field(default_factory=tuple)
    candidate_keys: tuple[CandidateKeyContract, ...] = field(default_factory=tuple)
    candidate_key_authorities: tuple[CandidateKeyAuthorityContract, ...] = field(
        default_factory=tuple
    )
    entity_references: tuple[EntityReferenceContract, ...] = field(
        default_factory=tuple
    )
    catalog_endpoint: CatalogEndpointContract | None = None

    def __post_init__(self) -> None:
        key_identities = tuple(
            (key.entity_kind, key.key_id) for key in self.candidate_keys
        )
        if len(set(key_identities)) != len(key_identities):
            raise ValueError("endpoint candidate key identities must be unique")
        authority_identities = tuple(
            (authority.entity_kind, authority.key_id)
            for authority in self.candidate_key_authorities
        )
        if len(set(authority_identities)) != len(authority_identities):
            raise ValueError("endpoint candidate key authorities must be unique")
        reference_ids = tuple(item.reference_id for item in self.entity_references)
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("endpoint entity reference ids must be unique")
        field_paths = {str(field.path or field.name) for field in self.response_fields}
        for key in self.candidate_keys:
            key_field_paths = tuple(item.field_path for item in key.components)
            unknown_paths = (
                set((*key_field_paths, *key.context_field_paths)) - field_paths
            )
            if unknown_paths:
                raise ValueError("endpoint candidate key references unknown field")
        for reference in self.entity_references:
            local_field_paths = tuple(
                item.local_field_path for item in reference.components
            )
            unknown_paths = (
                set((*local_field_paths, *reference.context_field_paths)) - field_paths
            )
            if unknown_paths:
                raise ValueError("endpoint entity reference references unknown field")

    def supports_lookup_read(self) -> bool:
        return bool(self.response_fields)

    def to_public_dict(
        self,
        *,
        include_response_fields: bool = True,
    ) -> dict[str, ContractValue]:
        payload: dict[str, ContractValue] = {
            "endpointName": self.endpoint_name,
            "urlName": self.url_name,
            "method": self.method,
            "path": self.path_template,
            "docstring": self.docstring,
            "viewClass": self.view_class,
            "pathParams": [item.to_public_dict() for item in self.path_params],
            "queryParams": [item.to_public_dict() for item in self.query_params],
            "access": {
                "admin": self.admin_access,
                "staff": self.staff_access,
                "agent": self.agent_access,
                "public": self.public_access,
            },
            "pagination": (
                None
                if self.pagination is None
                else self.pagination.to_public_dict()
            ),
            "schemaSources": {
                "query": self.query_schema_source,
                "response": self.response_schema_source,
            },
            "responseSchema": self.response_schema,
            "capabilities": _public_values(self.capabilities.to_public_dict()),
            "capabilitySources": list(self.capability_sources),
            "tags": list(self.tags),
            "resourceNames": list(self.resource_names),
            "candidateKeys": _public_values(
                key.to_public_dict() for key in self.candidate_keys
            ),
            "candidateKeyAuthorities": _public_values(
                authority.to_public_dict()
                for authority in self.candidate_key_authorities
            ),
            "entityReferences": _public_values(
                reference.to_public_dict() for reference in self.entity_references
            ),
        }
        if self.catalog_endpoint is not None:
            payload["catalogEndpoint"] = self.catalog_endpoint.to_public_dict()
        if include_response_fields:
            payload["responseFields"] = [
                item.to_public_dict() for item in self.response_fields
            ]
        else:
            payload["responseFieldNames"] = [item.name for item in self.response_fields]
            payload["responseFieldPaths"] = [item.path for item in self.response_fields]
        return payload
