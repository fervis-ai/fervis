"""DTOs for public GET endpoint contracts exposed to the fervis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fervis.host_api.contracts.capabilities import EndpointCapabilities


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
class ParameterContract:
    name: str
    type: str
    required: bool = False
    description: str = ""
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] = field(default_factory=dict)
    default: Any = None
    source: str = "query"
    identity: dict[str, Any] = field(default_factory=dict)
    semantics: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.identity:
            payload["identity"] = dict(self.identity)
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
    requires: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "description": self.description,
        }
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.requires:
            payload["requires"] = dict(self.requires)
        if self.identity:
            payload["identity"] = dict(self.identity)
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

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
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
    response_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: EndpointCapabilities = field(default_factory=EndpointCapabilities)
    capability_sources: tuple[str, ...] = ()
    agent_access: bool = False
    staff_access: bool = False
    admin_access: bool = True
    public_access: bool = False
    paginated: bool = False
    query_schema_source: str = "missing"
    response_schema_source: str = "missing"
    response_cardinality: str = "one"
    tags: tuple[str, ...] = field(default_factory=tuple)
    resource_names: tuple[str, ...] = field(default_factory=tuple)
    primary_key_fields: tuple[str, ...] = field(default_factory=tuple)
    catalog_endpoint: CatalogEndpointContract | None = None

    def is_detail_endpoint(self) -> bool:
        return bool(self.path_params) and not self._returns_collection_payload()

    def is_collection_endpoint(self) -> bool:
        return not self.is_detail_endpoint()

    def supports_lookup_read(self) -> bool:
        return bool(self.response_fields)

    def _returns_collection_payload(self) -> bool:
        if self.paginated:
            return True
        return any(
            str(field.path or field.name or "") == "data"
            or str(field.path or field.name or "").startswith("data.")
            for field in self.response_fields
        )

    def to_public_dict(self, *, include_response_fields: bool = True) -> dict[str, Any]:
        payload = {
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
            "pagination": {"paginated": self.paginated},
            "schemaSources": {
                "query": self.query_schema_source,
                "response": self.response_schema_source,
            },
            "responseSchema": self.response_schema,
            "capabilities": self.capabilities.to_public_dict(),
            "capabilitySources": list(self.capability_sources),
            "tags": list(self.tags),
            "resourceNames": list(self.resource_names),
            "primaryKeyFields": list(self.primary_key_fields),
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
