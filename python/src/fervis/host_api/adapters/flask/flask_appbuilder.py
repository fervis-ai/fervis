"""Locate Flask-AppBuilder OpenAPI metadata for Fervis endpoint contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.host_api.adapters.openapi import endpoint_evidence_from_openapi
from fervis.host_api.contracts import EndpointContract


def enrich_contract_from_flask_appbuilder_metadata(
    contract: EndpointContract,
    *,
    view: object | None,
) -> EndpointContract:
    if contract.response_fields:
        return contract
    operation = _get_operation_metadata(_view_resource(view), contract=contract)
    if operation is None:
        return contract
    evidence = next(
        iter(
            endpoint_evidence_from_openapi(
                {
                    "openapi": "3.1.0",
                    "paths": {
                        contract.path_template: {
                            "get": {
                                "operationId": contract.endpoint_name,
                                **operation,
                            }
                        }
                    },
                },
                path_prefixes=("/",),
            )
        ),
        None,
    )
    if evidence is None or not evidence.response_fields:
        return contract
    return replace(
        contract,
        query_params=evidence.query_params or contract.query_params,
        response_fields=evidence.response_fields,
        response_schema=evidence.response_schema,
        response_schema_source="flask_appbuilder",
        response_cardinality=evidence.response_cardinality,
        query_schema_source=(
            "flask_appbuilder"
            if evidence.query_params
            else contract.query_schema_source
        ),
    )


def _view_resource(view: object | None) -> object | None:
    if view is None:
        return None
    bound_self = getattr(view, "__self__", None)
    if bound_self is not None:
        return bound_self
    return getattr(view, "view_class", None) or view


def _get_operation_metadata(
    resource: object | None,
    *,
    contract: EndpointContract,
) -> dict[str, Any] | None:
    methods = getattr(resource, "openapi_spec_methods", None)
    if not isinstance(methods, dict):
        return None
    route_name = (
        contract.catalog_endpoint.route_name
        if contract.catalog_endpoint is not None
        else contract.endpoint_name
    )
    candidates = (
        "get",
        "GET",
        contract.endpoint_name,
        route_name,
        route_name.rsplit(".", 1)[-1],
    )
    for candidate in candidates:
        operation = methods.get(candidate)
        if isinstance(operation, dict):
            return operation
    return None
