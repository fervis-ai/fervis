"""Shared GET endpoint execution mechanics for host framework adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fervis.host_api.compilation import compile_read_request
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.response_envelope import (
    PAGINATION_FIELD,
    TOTAL_COUNT_FIELD,
    has_more_value,
)
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)


PageGetter = Callable[[str, dict[str, Any]], tuple[int, Any]]


@dataclass(frozen=True)
class PreparedGet:
    url: str
    query_params: dict[str, Any]
    transport_query_params: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    cookies: dict[str, str] | None = None


def prepare_get_endpoint(
    contract: EndpointContract,
    *,
    path_params: dict[str, Any] | None,
    query_params: dict[str, Any] | None,
    page_policy: dict[str, Any] | None = None,
    transport_overlay: ReadTransportOverlay | None = None,
) -> PreparedGet:
    compiled = compile_read_request(
        contract=contract,
        invocation=ReadInvocation(
            endpoint_name=contract.endpoint_name,
            path_params=dict(path_params or {}),
            query_params=dict(query_params or {}),
            page_policy=page_policy,
        ),
        transport_overlay=transport_overlay,
    )
    return PreparedGet(
        url=compiled.url,
        query_params=compiled.query_params,
        transport_query_params=compiled.transport_query_params,
        headers=compiled.headers,
        cookies=compiled.cookies,
    )


def execute_prepared_get(
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
    get_page: PageGetter,
) -> EndpointExecutionResult:
    policy = dict(page_policy or {})
    if str(policy.get("mode") or "single_page") == "all_pages":
        return execute_all_pages(
            contract=contract,
            prepared=prepared,
            page_policy=policy,
            get_page=get_page,
        )
    return execute_single_page(
        contract=contract,
        prepared=prepared,
        get_page=get_page,
    )


def execute_single_page(
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    get_page: PageGetter,
) -> EndpointExecutionResult:
    status, body = get_page(prepared.url, prepared.query_params)
    return EndpointExecutionResult(
        endpoint_name=contract.endpoint_name,
        request_url=prepared.url,
        query_params=prepared.query_params,
        response_status=status,
        response_body=body,
    )


def execute_all_pages(
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    page_policy: dict[str, Any],
    get_page: PageGetter,
) -> EndpointExecutionResult:
    limit = max(
        1,
        min(
            int(page_policy.get("limit") or prepared.query_params.get("limit") or 200),
            200,
        ),
    )
    max_pages = max(1, min(int(page_policy.get("max_pages") or 50), 50))
    merged: list[Any] = []
    last_status = 200
    page_count = 0
    final_has_more = False
    for page in range(max_pages):
        params = {**prepared.query_params, "limit": limit, "offset": page * limit}
        status, body = get_page(prepared.url, params)
        page_count += 1
        last_status = status
        if status >= 400:
            return EndpointExecutionResult(
                endpoint_name=contract.endpoint_name,
                request_url=prepared.url,
                query_params=params,
                response_status=status,
                response_body=body,
                page_count=page_count,
            )
        rows, has_more = rows_and_has_more(body)
        final_has_more = has_more
        merged.extend(rows)
        if not has_more:
            break
    return EndpointExecutionResult(
        endpoint_name=contract.endpoint_name,
        request_url=prepared.url,
        query_params={**prepared.query_params, "limit": limit, "offset": 0},
        response_status=last_status,
        response_body={
            "data": merged,
            PAGINATION_FIELD: {
                TOTAL_COUNT_FIELD: len(merged),
                "limit": limit,
                "offset": 0,
                "has_more": final_has_more,
            },
        },
        page_count=page_count,
        truncated=final_has_more,
    )


def rows_and_has_more(body: Any) -> tuple[list[Any], bool]:
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        return list(body["data"]), bool(has_more_value(body))
    if isinstance(body, list):
        return list(body), False
    raise EndpointExecutionError("Paginated endpoint response requires a row list")
