"""Compile selected host API reads into executable requests."""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import quote

from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.execution import (
    CompiledReadRequest,
    ReadTransportOverlay,
)
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.ports import EndpointExecutionError

_PATH_PLACEHOLDER = re.compile(r"{([^{}]+)}")
_UNSAFE_PATH_FRAGMENTS = ("/", "\\", "?", "#")


def compile_read_request(
    *,
    contract: EndpointContract,
    invocation: ReadInvocation,
    transport_overlay: ReadTransportOverlay | None = None,
) -> CompiledReadRequest:
    overlay = transport_overlay or ReadTransportOverlay()
    _validate_pagination(contract, invocation.page_policy)
    path_params = dict(invocation.path_params)
    query_params = _normalize_query_params(dict(invocation.query_params))
    _validate_path_params(contract, path_params)
    _validate_query_params(contract, query_params)
    _validate_overlay(query_params, overlay)
    return CompiledReadRequest(
        url=_build_url(contract, path_params),
        query_params=query_params,
        transport_query_params=dict(overlay.query_params),
        headers=dict(overlay.headers),
        cookies=dict(overlay.cookies),
    )


def _validate_pagination(
    contract: EndpointContract,
    page_policy: Mapping[str, Any] | None,
) -> None:
    policy = dict(page_policy or {})
    if (
        str(policy.get("mode") or "single_page") == "all_pages"
        and not contract.paginated
    ):
        raise EndpointExecutionError(
            f"Endpoint {contract.endpoint_name} is not paginated."
        )


def _validate_path_params(
    contract: EndpointContract,
    path_params: Mapping[str, Any],
) -> None:
    declared = {item.name for item in contract.path_params}
    placeholders = set(_PATH_PLACEHOLDER.findall(contract.path_template))
    allowed = declared | placeholders
    unknown = sorted(set(path_params) - allowed)
    if unknown:
        raise EndpointExecutionError(
            f"Unknown path params for {contract.endpoint_name}: {', '.join(unknown)}"
        )
    required = {
        item.name for item in contract.path_params if item.required
    } | placeholders
    missing = sorted(name for name in required if _blank(path_params.get(name)))
    if missing:
        raise EndpointExecutionError(
            f"Missing required path params: {', '.join(missing)}"
        )
    for name, value in path_params.items():
        _validate_path_value(name, value)


def _validate_path_value(name: str, value: Any) -> None:
    text = str(value)
    if text in {".", ".."} or any(
        fragment in text for fragment in _UNSAFE_PATH_FRAGMENTS
    ):
        raise EndpointExecutionError(
            f"Unsafe path param {name}: path param values must not contain URL structure."
        )


def _validate_query_params(
    contract: EndpointContract,
    query_params: Mapping[str, Any],
) -> None:
    allowed = {item.name for item in contract.query_params}
    unknown = sorted(set(query_params) - allowed)
    if unknown:
        raise EndpointExecutionError(
            f"Unknown query params for {contract.endpoint_name}: {', '.join(unknown)}"
        )
    required = {item.name for item in contract.query_params if item.required}
    missing = sorted(name for name in required if _blank(query_params.get(name)))
    if missing:
        raise EndpointExecutionError(
            f"Missing required query params: {', '.join(missing)}"
        )


def _validate_overlay(
    selected_query_params: Mapping[str, Any],
    overlay: ReadTransportOverlay,
) -> None:
    protected = set(selected_query_params) | {"limit", "offset"}
    overlap = sorted(protected & set(overlay.query_params))
    if overlap:
        raise EndpointExecutionError(
            "HTTP request overlay must not overlap selected query params: "
            + ", ".join(overlap)
        )
    allowed = set(overlay.allowed_query_params)
    unknown = sorted(set(overlay.query_params) - allowed)
    if unknown:
        raise EndpointExecutionError(
            "HTTP request overlay query params are not allowed: " + ", ".join(unknown)
        )


def _build_url(contract: EndpointContract, path_params: Mapping[str, Any]) -> str:
    url = contract.path_template
    for key, value in path_params.items():
        url = url.replace("{" + key + "}", quote(str(value), safe=""))
    if "{" in url or "}" in url:
        raise EndpointExecutionError(
            f"Unresolved path params for {contract.endpoint_name}: {url}"
        )
    return url


def _normalize_query_params(query_params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(query_params)
    if "limit" in normalized:
        try:
            normalized["limit"] = max(1, min(int(normalized["limit"]), 200))
        except (TypeError, ValueError):
            raise EndpointExecutionError(
                "Invalid pagination query param limit: expected integer."
            ) from None
    if "offset" in normalized:
        try:
            normalized["offset"] = max(0, int(normalized["offset"]))
        except (TypeError, ValueError):
            raise EndpointExecutionError(
                "Invalid pagination query param offset: expected integer."
            ) from None
    return normalized


def _blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""
