"""FastAPI/OpenAPI endpoint catalog translation."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute

from fervis.host_api.contracts import (
    EndpointContract,
    FrameworkKind,
    SourceNamespaceKind,
)
from fervis.host_api.contracts import CatalogEndpointContract
from fervis.project.integration import FastAPIAppSource
from fervis.project.source_paths import (
    normalize_source_path_prefixes,
    source_path_matches,
)
from fervis.host_api.adapters.resource_names import endpoint_resource_names

from .loading import import_fastapi_app
from .routes import effective_api_routes
from .schema_introspection import (
    fastapi_detail_path_parameters,
    fastapi_route_parameters,
    inspect_fastapi_response,
)


def get_fastapi_endpoint_contracts(
    *,
    sources: tuple[FastAPIAppSource, ...],
    project_root: Path,
) -> tuple[EndpointContract, ...]:
    contracts: list[EndpointContract] = []
    for source in sources:
        for import_path in source.import_paths:
            app = import_fastapi_app(import_path, project_root=project_root)
            contracts.extend(
                endpoint_contracts_from_fastapi_app(
                    app,
                    source=source,
                    import_path=import_path,
                )
            )
    return tuple(sorted(contracts, key=lambda item: item.path_template))


def _normalized_path_prefixes(path_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    try:
        return normalize_source_path_prefixes(path_prefixes)
    except ValueError as exc:
        raise ValueError(f"FastAPIAppSource.path_prefixes invalid: {exc}") from exc


def endpoint_contracts_from_fastapi_app(
    app: FastAPI,
    *,
    source: FastAPIAppSource,
    import_path: str,
) -> tuple[EndpointContract, ...]:
    path_prefixes = _normalized_path_prefixes(tuple(source.path_prefixes))
    return tuple(
        _endpoint_contract(
            route,
            source_name=source.name,
            import_path=import_path,
        )
        for route in effective_api_routes(app)
        if "GET" in (route.methods or set())
        and source_path_matches(route.path, path_prefixes)
    )


def _endpoint_contract(
    route: APIRoute,
    *,
    source_name: str,
    import_path: str,
) -> EndpointContract:
    response = inspect_fastapi_response(route)
    operation_id = str(route.operation_id or route.unique_id or route.name)
    tags = tuple(str(tag) for tag in route.tags)
    resource_names = endpoint_resource_names(
        tags=tags,
        operation_id=operation_id,
        path_template=route.path,
    ) or (source_name,)
    endpoint = route.endpoint
    handler_ref = f"{endpoint.__module__}:{endpoint.__qualname__}"
    return EndpointContract(
        endpoint_name=operation_id,
        url_name=operation_id,
        method="GET",
        path_template=route.path,
        docstring=str(route.description or ""),
        view_class=handler_ref,
        path_params=fastapi_detail_path_parameters(route, response=response),
        query_params=fastapi_route_parameters(route, source="query"),
        response_fields=response.fields,
        response_schema=response.schema,
        response_schema_source="fastapi_response_model",
        response_cardinality=response.cardinality,
        query_schema_source="fastapi_dependencies",
        tags=tags,
        resource_names=resource_names,
        candidate_keys=response.candidate_keys,
        candidate_key_authorities=response.candidate_key_authorities,
        entity_references=response.entity_references,
        catalog_endpoint=CatalogEndpointContract(
            framework_kind=FrameworkKind.FASTAPI.value,
            source_namespace_kind=SourceNamespaceKind.FASTAPI_APP.value,
            source_namespace_path=(source_name,),
            handler_ref=import_path,
            route_name=route.name,
            api_schema_operation_id=operation_id,
            domain_resource_names=resource_names,
        ),
    )
