"""Shared rules for safe read probes against declared endpoint contracts."""

from __future__ import annotations

from .endpoint import EndpointContract


def belongs_to_source(contract: EndpointContract, *, source_name: str) -> bool:
    catalog_endpoint = contract.catalog_endpoint
    if catalog_endpoint is None:
        return False
    namespace = catalog_endpoint.source_namespace_path
    return bool(namespace and namespace[0] == source_name)


def is_probeable_get_contract(
    contract: EndpointContract,
    *,
    source_name: str | None = None,
) -> bool:
    if source_name is not None and not belongs_to_source(
        contract, source_name=source_name
    ):
        return False
    return (
        str(contract.method).upper() == "GET"
        and not contract.path_params
        and not any(param.required for param in contract.query_params)
    )
