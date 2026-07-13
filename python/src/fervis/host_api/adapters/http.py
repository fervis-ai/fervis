"""Shared HTTP GET execution for host API adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import requests

from fervis.host_api.adapters.get_execution import (
    execute_prepared_get,
    prepare_get_endpoint,
)
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.authority import ReadAuthority
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)
from fervis.project.importing import import_object
from fervis.host_api.credentials import (
    CapturedHeaderCredentialPolicy,
    credential_policy_from_auth_schema,
    overlay_from_header_credential,
)
from .response_body import response_body


@dataclass(frozen=True)
class HttpReadExecutionConfig:
    base_url_env: str
    request_overlay_source: str | None = None
    timeout_seconds: int = 30
    auth_query_params: tuple[str, ...] = ()
    credential_policy: CapturedHeaderCredentialPolicy | None = None


@dataclass(frozen=True)
class HttpRequestOverlay:
    headers: Mapping[str, str] = field(default_factory=dict)
    query_params: Mapping[str, str] = field(default_factory=dict)
    cookies: Mapping[str, str] = field(default_factory=dict)


def http_read_config_from_auth_schema(
    schema: dict[str, object] | None,
) -> HttpReadExecutionConfig | None:
    transport = _mapping((schema or {}).get("transport"))
    if transport.get("mode") != "http":
        return None
    return HttpReadExecutionConfig(
        base_url_env=str(transport.get("base_url_env") or ""),
        request_overlay_source=str(transport.get("request_overlay_source") or ""),
        auth_query_params=_string_sequence(transport.get("auth_query_params")),
        credential_policy=credential_policy_from_auth_schema(schema),
    )


def execute_http_read(
    *,
    contract: EndpointContract,
    authority: ReadAuthority,
    invocation: ReadInvocation,
    config: HttpReadExecutionConfig,
) -> EndpointExecutionResult:
    overlay = _merge_overlays(
        overlay_from_header_credential(
            authority.delegated_credential,
            policy=config.credential_policy,
        ),
        _request_overlay(
            config,
            authority=authority,
            invocation=invocation,
        ),
    )
    transport_overlay = ReadTransportOverlay(
        headers=overlay.headers,
        query_params=overlay.query_params,
        cookies=overlay.cookies,
        allowed_query_params=config.auth_query_params,
    )
    prepared = prepare_get_endpoint(
        contract=contract,
        path_params=dict(invocation.path_params),
        query_params=dict(invocation.query_params),
        page_policy=(
            None if invocation.page_policy is None else dict(invocation.page_policy)
        ),
        transport_overlay=transport_overlay,
    )
    base_url = _base_url(config.base_url_env)
    return execute_prepared_get(
        contract=contract,
        prepared=prepared,
        page_policy=(
            None if invocation.page_policy is None else dict(invocation.page_policy)
        ),
        get_page=lambda url, params: _get_page(
            base_url=base_url,
            url=url,
            query_params=params,
            prepared=prepared,
            timeout_seconds=config.timeout_seconds,
        ),
    )


def _get_page(
    *,
    base_url: str,
    url: str,
    query_params: dict[str, Any],
    prepared,
    timeout_seconds: int,
) -> tuple[int, Any]:
    response = _get(
        _join_url(base_url, url),
        params={**query_params, **dict(prepared.transport_query_params or {})},
        headers=dict(prepared.headers or {}),
        cookies=dict(prepared.cookies or {}),
        timeout=timeout_seconds,
    )
    return response.status_code, response_body(response)


def _request_overlay(
    config: HttpReadExecutionConfig,
    *,
    authority: ReadAuthority,
    invocation: ReadInvocation,
) -> HttpRequestOverlay:
    if not config.request_overlay_source:
        return HttpRequestOverlay()
    source = import_object(config.request_overlay_source)
    if not callable(source):
        raise EndpointExecutionError(
            f"HTTP request overlay source is not callable: "
            f"{config.request_overlay_source}"
        )
    return _coerce_overlay(source(authority, invocation))


def _merge_overlays(
    first: ReadTransportOverlay,
    second: HttpRequestOverlay,
) -> HttpRequestOverlay:
    _reject_overlap(
        "headers",
        first.headers,
        second.headers,
        normalize=lambda value: value.lower(),
    )
    _reject_overlap("query params", first.query_params, second.query_params)
    _reject_overlap("cookies", first.cookies, second.cookies)
    return HttpRequestOverlay(
        headers={**dict(first.headers), **dict(second.headers)},
        query_params={**dict(first.query_params), **dict(second.query_params)},
        cookies={**dict(first.cookies), **dict(second.cookies)},
    )


def _reject_overlap(
    label: str,
    first: Mapping[str, str],
    second: Mapping[str, str],
    *,
    normalize=lambda value: value,
) -> None:
    left = {normalize(str(key)): str(key) for key in first}
    right = {normalize(str(key)): str(key) for key in second}
    overlap = sorted(left[key] for key in set(left) & set(right))
    if overlap:
        raise EndpointExecutionError(
            f"HTTP request overlay {label} overlap: " + ", ".join(overlap)
        )


def _coerce_overlay(value: object) -> HttpRequestOverlay:
    if isinstance(value, HttpRequestOverlay):
        return value
    if not isinstance(value, Mapping):
        raise EndpointExecutionError("HTTP request overlay must be an object.")
    return HttpRequestOverlay(
        headers=_string_mapping(value.get("headers"), label="headers"),
        query_params=_string_mapping(
            value.get("query_params"),
            label="query_params",
        ),
        cookies=_string_mapping(value.get("cookies"), label="cookies"),
    )


def _string_mapping(value: object, *, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EndpointExecutionError(f"HTTP request overlay {label} must be an object.")
    return {str(key): str(item) for key, item in value.items()}


def _base_url(env_name: str) -> str:
    base_url = os.getenv(env_name, "").strip()
    if not base_url:
        raise EndpointExecutionError(
            f"HTTP execution base URL is missing from {env_name}."
        )
    return base_url


def _join_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _get(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    cookies: dict[str, str],
    timeout: int,
):
    return requests.get(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        timeout=timeout,
    )


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item or "").strip())
