"""Execute public GET endpoint contracts through DRF, never ORM shortcuts."""

from __future__ import annotations

import json
from typing import Any

from django.core.serializers.json import DjangoJSONEncoder
from rest_framework.test import APIClient

from fervis.host_api.adapters.get_execution import (
    execute_prepared_get,
    prepare_get_endpoint,
)
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)
from fervis.project.source_scope import DjangoSourceScope

from .catalog import get_endpoint_contract


def execute_get_endpoint(
    *,
    endpoint_name: str,
    user: Any,
    sources: tuple[DjangoSourceScope, ...],
    path_params: dict[str, Any] | None = None,
    query_params: dict[str, Any] | None = None,
    page_policy: dict[str, Any] | None = None,
    transport_overlay: ReadTransportOverlay | None = None,
) -> EndpointExecutionResult:
    contract = get_endpoint_contract(
        endpoint_name,
        sources=sources,
    )
    if contract is None:
        raise EndpointExecutionError(f"Unknown endpoint: {endpoint_name}")

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
            user=user,
            url=url,
            query_params=params,
            headers=prepared.headers or {},
            cookies=prepared.cookies or {},
        ),
    )


def _get_page(
    *,
    user: Any,
    url: str,
    query_params: dict[str, Any],
    headers: dict[str, str],
    cookies: dict[str, str],
) -> tuple[int, Any]:
    client = _client_for(user)
    for name, value in cookies.items():
        client.cookies[name] = value
    response = client.get(url, query_params, **_django_header_kwargs(headers))
    body = response.data if hasattr(response, "data") else {}
    return response.status_code, _json_safe(body)


def _client_for(user: Any) -> APIClient:
    client = APIClient()
    client.force_authenticate(user=user)
    return client


def _django_header_kwargs(headers: dict[str, str]) -> dict[str, str]:
    return {
        "HTTP_" + name.upper().replace("-", "_"): value
        for name, value in headers.items()
        if name.lower() not in {"content-type", "content-length"}
    }


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, cls=DjangoJSONEncoder))
