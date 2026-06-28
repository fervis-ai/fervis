"""Execute configured Flask GET endpoint contracts in-process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from fervis.project.integration import FlaskAppSource
from fervis.project.importing import project_import_context

from .catalog import get_flask_endpoint_contracts
from .loading import import_flask_app
from .principal import FlaskPrincipalOverride, resolve_flask_principal
from .transport import FlaskInProcessReadTransport


def execute_get_endpoint(
    *,
    endpoint_name: str,
    sources: tuple[FlaskAppSource, ...],
    project_root: Path,
    path_params: dict[str, Any] | None = None,
    query_params: dict[str, Any] | None = None,
    page_policy: dict[str, Any] | None = None,
    principal_override: FlaskPrincipalOverride | None = None,
    transport_overlay: ReadTransportOverlay | None = None,
) -> EndpointExecutionResult:
    with project_import_context(project_root):
        contract = _endpoint_contract(
            endpoint_name,
            sources=sources,
            project_root=project_root,
        )
        if contract is None:
            raise EndpointExecutionError(f"Unknown endpoint: {endpoint_name}")
        source = _source_for_contract(contract, sources=sources)
        prepared = prepare_get_endpoint(
            contract=contract,
            path_params=path_params,
            query_params=query_params,
            page_policy=page_policy,
            transport_overlay=transport_overlay,
        )
        app = import_flask_app(
            source.app,
            project_root=project_root,
            app_args=tuple(source.app_args),
            app_kwargs=dict(source.app_kwargs),
        )
        principal = (
            None
            if principal_override is None
            else resolve_flask_principal(principal_override)
        )
        transport = FlaskInProcessReadTransport(app)
        return execute_prepared_get(
            contract=contract,
            prepared=prepared,
            page_policy=page_policy,
            get_page=lambda url, params: transport.get(
                url,
                params,
                principal=principal,
                headers=prepared.headers or {},
                cookies=prepared.cookies or {},
            ),
        )


def _endpoint_contract(
    endpoint_name: str,
    *,
    sources: tuple[FlaskAppSource, ...],
    project_root: Path,
) -> EndpointContract | None:
    return next(
        (
            contract
            for contract in get_flask_endpoint_contracts(
                sources=sources,
                project_root=project_root,
            )
            if contract.endpoint_name == endpoint_name
        ),
        None,
    )


def _source_for_contract(
    contract: EndpointContract,
    *,
    sources: tuple[FlaskAppSource, ...],
) -> FlaskAppSource:
    namespace = (
        contract.catalog_endpoint.source_namespace_path
        if contract.catalog_endpoint is not None
        else ()
    )
    source_name = namespace[0] if namespace else ""
    for source in sources:
        if source.name == source_name:
            return source
    raise EndpointExecutionError(f"Unknown Flask source for {contract.endpoint_name}")
