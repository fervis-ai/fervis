"""Execute configured FastAPI GET endpoint contracts in-process."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass

from fervis.host_api.adapters.get_execution import (
    execute_prepared_get,
    prepare_get_endpoint,
)
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)
from fervis.project.integration import FastAPIAppSource
from fervis.project.importing import import_object, project_import_context

from . import catalog
from .loading import load_fastapi_app
from ..response_body import response_body
from ..runtime_output import suppress_host_output


@dataclass(frozen=True)
class FastAPIDependencyOverride:
    dependency: str
    resolver: str
    key: str | None
    tenant_id: str | None = None
    principal_id_attr: str = "id"


def execute_get_endpoint(
    *,
    endpoint_name: str,
    sources: tuple[FastAPIAppSource, ...],
    project_root: Path,
    path_params: dict[str, Any] | None = None,
    query_params: dict[str, Any] | None = None,
    page_policy: dict[str, Any] | None = None,
    dependency_override: FastAPIDependencyOverride | None = None,
    transport_overlay: ReadTransportOverlay | None = None,
) -> EndpointExecutionResult:
    contract = _endpoint_contract(
        endpoint_name,
        sources=sources,
        project_root=project_root,
    )
    if contract is None:
        raise EndpointExecutionError(f"Unknown endpoint: {endpoint_name}")

    import_path = _app_import_path(contract, sources=sources)
    with project_import_context(project_root):
        app = load_fastapi_app(import_path)
        overrides = _resolved_overrides(dependency_override)
        with _dependency_overrides(app, overrides):
            client = _test_client(app)
            prepared = prepare_get_endpoint(
                contract=contract,
                path_params=path_params,
                query_params=query_params,
                page_policy=page_policy,
                transport_overlay=transport_overlay,
            )
            return execute_prepared_get(
                contract=contract,
                prepared=prepared,
                page_policy=page_policy,
                get_page=lambda url, params: _get_page(
                    client,
                    url,
                    params,
                    headers=prepared.headers or {},
                    cookies=prepared.cookies or {},
                ),
            )


def _get_page(
    client: Any,
    url: str,
    query_params: dict[str, Any],
    *,
    headers: dict[str, str],
    cookies: dict[str, str],
) -> tuple[int, Any]:
    with suppress_host_output():
        response = client.get(
            url, params=query_params, headers=headers, cookies=cookies
        )
    return response.status_code, response_body(response)


def _endpoint_contract(
    endpoint_name: str,
    *,
    sources: tuple[FastAPIAppSource, ...],
    project_root: Path,
) -> EndpointContract | None:
    return next(
        (
            contract
            for contract in catalog.get_fastapi_endpoint_contracts(
                sources=sources,
                project_root=project_root,
            )
            if contract.endpoint_name == endpoint_name
        ),
        None,
    )


def _test_client(app: object) -> Any:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        raise RuntimeError("FastAPI endpoint execution requires fastapi.") from exc
    return TestClient(app)


def _app_import_path(
    contract: EndpointContract,
    *,
    sources: tuple[FastAPIAppSource, ...],
) -> str:
    source = _source_for_contract(contract, sources=sources)
    import_path = (
        contract.catalog_endpoint.handler_ref
        if contract.catalog_endpoint is not None
        else ""
    )
    if import_path not in source.import_paths:
        raise EndpointExecutionError(
            f"Endpoint {contract.endpoint_name} is not in configured FastAPI sources."
        )
    return import_path


@contextmanager
def _dependency_overrides(app: object, overrides: dict[Any, Any]):
    if not overrides:
        yield
        return
    existing = getattr(app, "dependency_overrides", None)
    if not isinstance(existing, dict):
        raise EndpointExecutionError(
            "FastAPI app does not expose dependency_overrides."
        )
    original = dict(existing)
    try:
        existing.update(overrides)
        yield
    finally:
        existing.clear()
        existing.update(original)


def _resolved_overrides(
    override: FastAPIDependencyOverride | None,
) -> dict[Any, Any]:
    if override is None:
        return {}
    dependency = import_object(override.dependency)
    resolver = import_object(override.resolver)
    principal = resolver(override.key, override.tenant_id)
    _validate_resolved_principal(principal, override=override)
    return {dependency: lambda: principal}


def _validate_resolved_principal(
    principal: object,
    *,
    override: FastAPIDependencyOverride,
) -> None:
    if principal is None:
        raise EndpointExecutionError(
            f"FastAPI resolver could not resolve principal: {override.key}"
        )
    actual = getattr(principal, override.principal_id_attr, None)
    if actual is None and isinstance(principal, dict):
        actual = principal.get(override.principal_id_attr)
    if str(actual or "") != str(override.key or ""):
        raise EndpointExecutionError(
            "FastAPI resolver returned a different principal than requested."
        )


def _source_for_contract(
    contract: EndpointContract,
    *,
    sources: tuple[FastAPIAppSource, ...],
) -> FastAPIAppSource:
    namespace = (
        contract.catalog_endpoint.source_namespace_path
        if contract.catalog_endpoint is not None
        else ()
    )
    source_name = namespace[0] if namespace else ""
    for source in sources:
        if source.name == source_name:
            return source
    raise EndpointExecutionError(f"Unknown FastAPI source for {contract.endpoint_name}")
