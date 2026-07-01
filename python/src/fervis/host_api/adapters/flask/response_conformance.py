"""Flask runtime probes for declared response contract conformance."""

from __future__ import annotations

from pathlib import Path

from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.probe import (
    belongs_to_source,
    is_probeable_get_contract,
)
from fervis.host_api.contracts.response_conformance import (
    ResponseConformanceResult,
    check_response_conformance,
)
from fervis.project.integration import FlaskAppSource

from .loading import import_flask_app
from .transport import FlaskInProcessReadTransport


def check_flask_response_conformance(
    *,
    sources: tuple[FlaskAppSource, ...],
    project_root: Path,
    contracts: tuple[EndpointContract, ...],
) -> tuple[ResponseConformanceResult, ...]:
    results: list[ResponseConformanceResult] = []
    for source in sources:
        source_contracts = tuple(
            contract
            for contract in contracts
            if contract.response_fields
            and belongs_to_source(contract, source_name=source.name)
        )
        if not source_contracts:
            continue
        app = import_flask_app(
            source.app,
            project_root=project_root,
            app_args=tuple(source.app_args),
            app_kwargs=dict(source.app_kwargs),
        )
        transport = FlaskInProcessReadTransport(app)
        for contract in source_contracts:
            if not is_probeable_get_contract(contract):
                results.append(_skipped_required_params(contract))
                continue
            status, body = transport.get(contract.path_template, {})
            if status in {401, 403}:
                results.append(_skipped_auth_required(contract, status=status))
                continue
            if status != 200:
                results.append(_failed_status(contract, status=status))
                continue
            results.append(check_response_conformance(contract, body))
    return tuple(results)


def _skipped_required_params(contract: EndpointContract) -> ResponseConformanceResult:
    return ResponseConformanceResult(
        endpoint_name=contract.endpoint_name,
        path_template=contract.path_template,
        status="skipped",
        reason="required_params",
        message=(
            f"GET {contract.path_template} requires path or query params; "
            "Fervis will not invent sample values for response shape probes."
        ),
    )


def _skipped_auth_required(
    contract: EndpointContract,
    *,
    status: int,
) -> ResponseConformanceResult:
    return ResponseConformanceResult(
        endpoint_name=contract.endpoint_name,
        path_template=contract.path_template,
        status="skipped",
        reason="auth_required",
        message=(
            f"GET {contract.path_template} returned HTTP {status}; response "
            "shape conformance requires configured host read credentials."
        ),
    )


def _failed_status(
    contract: EndpointContract,
    *,
    status: int,
) -> ResponseConformanceResult:
    return ResponseConformanceResult(
        endpoint_name=contract.endpoint_name,
        path_template=contract.path_template,
        status="failed",
        reason="http_status",
        message=(
            f"GET {contract.path_template} returned HTTP {status}; Fervis "
            "could not verify the declared response shape."
        ),
    )
