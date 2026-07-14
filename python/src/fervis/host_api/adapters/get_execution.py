"""Shared GET endpoint execution mechanics for host framework adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Any

from fervis.host_api.compilation import compile_read_request
from fervis.host_api.contracts import EndpointContract, PaginationKind
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.response_envelope import (
    PAGINATION_FIELD,
    TOTAL_COUNT_FIELD,
)
from fervis.host_api.contracts.ports import (
    EndpointExecutionError,
    EndpointExecutionResult,
)


PageGetter = Callable[[str, dict[str, Any]], tuple[int, Any]]
AsyncPageGetter = Callable[[str, dict[str, Any]], Awaitable[tuple[int, Any]]]

_DEFAULT_MAX_PAGES = 10
_DEFAULT_MAX_ROWS = 2_000
_HARD_MAX_PAGES = 50
_HARD_MAX_ROWS = 10_000


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
    page_policy: dict[str, Any] | None = None,
    get_page: PageGetter,
) -> EndpointExecutionResult:
    if contract.pagination is not None and _all_pages_requested(page_policy):
        return execute_all_pages(
            contract=contract,
            prepared=prepared,
            page_policy=page_policy,
            get_page=get_page,
        )
    return execute_single_page(
        contract=contract,
        prepared=prepared,
        page_policy=page_policy,
        get_page=get_page,
    )


def execute_single_page(
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
    get_page: PageGetter,
) -> EndpointExecutionResult:
    query_params, page_size = _single_page_request(
        contract,
        prepared=prepared,
        page_policy=page_policy,
    )
    status, body = get_page(prepared.url, query_params)
    return _single_page_result(
        contract,
        prepared=prepared,
        status=status,
        body=body,
        page_size=page_size,
    )


def execute_all_pages(
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
    get_page: PageGetter,
) -> EndpointExecutionResult:
    traversal = _page_traversal(
        contract,
        prepared=prepared,
        page_policy=page_policy,
    )
    return _drive_pages(traversal, url=prepared.url, get_page=get_page)


async def execute_prepared_get_async(
    *,
    contract: EndpointContract,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
    get_page: AsyncPageGetter,
) -> EndpointExecutionResult:
    if contract.pagination is None or not _all_pages_requested(page_policy):
        query_params, page_size = _single_page_request(
            contract,
            prepared=prepared,
            page_policy=page_policy,
        )
        status, body = await get_page(prepared.url, query_params)
        return _single_page_result(
            contract,
            prepared=prepared,
            status=status,
            body=body,
            page_size=page_size,
        )
    traversal = _page_traversal(
        contract,
        prepared=prepared,
        page_policy=page_policy,
    )
    query_params = next(traversal)
    while True:
        response = await get_page(prepared.url, query_params)
        try:
            query_params = traversal.send(response)
        except StopIteration as completed:
            return completed.value


PageTraversal = Generator[
    dict[str, Any],
    tuple[int, Any],
    EndpointExecutionResult,
]


def _drive_pages(
    traversal: PageTraversal,
    *,
    url: str,
    get_page: PageGetter,
) -> EndpointExecutionResult:
    query_params = next(traversal)
    while True:
        response = get_page(url, query_params)
        try:
            query_params = traversal.send(response)
        except StopIteration as completed:
            return completed.value


def _page_traversal(
    contract: EndpointContract,
    *,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
) -> PageTraversal:
    max_pages, max_rows = _traversal_limits(page_policy)
    page_size = _page_size(contract, page_policy=page_policy)
    query_params = first_page_request(
        contract,
        prepared=prepared,
        page_size=page_size,
    )
    rows: list[Any] = []
    total: int | None = None
    last_status = 200
    has_more = True
    page_count = 0
    while has_more and page_count < max_pages and len(rows) < max_rows:
        status, body = yield query_params
        page_count += 1
        last_status = status
        if status >= 400:
            return _failed_page_result(
                contract,
                prepared=prepared,
                query_params=query_params,
                status=status,
                body=body,
                page_count=page_count,
            )
        page_rows, page_total, continuation = parse_page(contract, body=body)
        total = _consistent_total(total, page_total)
        remaining = max_rows - len(rows)
        rows.extend(page_rows[:remaining])
        _validate_collected_total(total=total, collected_rows=len(rows))
        has_more = len(page_rows) > remaining or _page_has_more(
            total=total,
            collected_rows=len(rows),
            continuation=continuation,
            returned_rows=len(page_rows),
            page_size=page_size,
        )
        if has_more and page_rows:
            query_params = next_page_request(
                contract,
                previous=query_params,
                page_size=page_size,
                returned_rows=len(page_rows),
            )
        elif has_more:
            break
    return _page_collection_result(
        contract,
        prepared=prepared,
        page_size=page_size,
        rows=rows,
        total=total,
        last_status=last_status,
        page_count=page_count,
        truncated=has_more,
    )


def first_page_request(
    contract: EndpointContract,
    *,
    prepared: PreparedGet,
    page_size: int,
) -> dict[str, Any]:
    pagination = contract.pagination
    if pagination is None:
        raise EndpointExecutionError(
            f"Endpoint {contract.endpoint_name} has no pagination contract."
        )
    params = dict(prepared.query_params)
    if pagination.page_size_query_param:
        params[pagination.page_size_query_param] = page_size
    params[pagination.position_query_param] = (
        0 if pagination.kind is PaginationKind.OFFSET else 1
    )
    return params


def parse_page(
    contract: EndpointContract,
    *,
    body: Any,
) -> tuple[list[Any], int | None, bool | None]:
    pagination = contract.pagination
    if pagination is None:
        raise EndpointExecutionError(
            f"Endpoint {contract.endpoint_name} has no pagination contract."
        )
    rows = _rows_at_path(body, path=pagination.results_path)
    total = _total(body, path=pagination.total_path)
    continuation = _continuation(body, path=pagination.continuation_path)
    if total is not None and total < len(rows):
        raise EndpointExecutionError(
            "Paginated endpoint total is smaller than its returned row count."
        )
    return rows, total, None if continuation is None else bool(continuation)


def next_page_request(
    contract: EndpointContract,
    *,
    previous: dict[str, Any],
    page_size: int,
    returned_rows: int,
) -> dict[str, Any]:
    pagination = contract.pagination
    if pagination is None:
        raise EndpointExecutionError(
            f"Endpoint {contract.endpoint_name} has no pagination contract."
        )
    position = int(previous[pagination.position_query_param])
    params = dict(previous)
    if pagination.kind is PaginationKind.OFFSET:
        params[pagination.position_query_param] = position + returned_rows
    elif pagination.kind is PaginationKind.PAGE_NUMBER:
        params[pagination.position_query_param] = position + 1
    else:
        raise EndpointExecutionError("Unsupported pagination kind.")
    if pagination.page_size_query_param:
        params[pagination.page_size_query_param] = page_size
    return params


def _page_collection_result(
    contract: EndpointContract,
    *,
    prepared: PreparedGet,
    page_size: int,
    rows: list[Any],
    total: int | None,
    last_status: int,
    page_count: int,
    truncated: bool,
) -> EndpointExecutionResult:
    return EndpointExecutionResult(
        endpoint_name=contract.endpoint_name,
        request_url=prepared.url,
        query_params=prepared.query_params,
        response_status=last_status,
        response_body={
            "data": rows,
            PAGINATION_FIELD: {
                TOTAL_COUNT_FIELD: total if total is not None else len(rows),
                "page_size": page_size,
                "has_more": truncated,
            },
        },
        page_count=page_count,
        truncated=truncated,
    )


def _failed_page_result(
    contract: EndpointContract,
    *,
    prepared: PreparedGet,
    query_params: dict[str, Any],
    status: int,
    body: Any,
    page_count: int,
) -> EndpointExecutionResult:
    return EndpointExecutionResult(
        endpoint_name=contract.endpoint_name,
        request_url=prepared.url,
        query_params=query_params,
        response_status=status,
        response_body=body,
        page_count=page_count,
    )


def _traversal_limits(page_policy: dict[str, Any] | None) -> tuple[int, int]:
    policy = dict(page_policy or {})
    max_pages = max(
        1,
        min(int(policy.get("max_pages") or _DEFAULT_MAX_PAGES), _HARD_MAX_PAGES),
    )
    max_rows = max(
        1,
        min(int(policy.get("max_rows") or _DEFAULT_MAX_ROWS), _HARD_MAX_ROWS),
    )
    return max_pages, max_rows


def _all_pages_requested(page_policy: dict[str, Any] | None) -> bool:
    return str(dict(page_policy or {}).get("mode") or "single_page") == "all_pages"


def _single_page_request(
    contract: EndpointContract,
    *,
    prepared: PreparedGet,
    page_policy: dict[str, Any] | None,
) -> tuple[dict[str, Any], int | None]:
    if contract.pagination is None:
        return prepared.query_params, None
    page_size = _page_size(contract, page_policy=page_policy)
    return (
        first_page_request(contract, prepared=prepared, page_size=page_size),
        page_size,
    )


def _single_page_result(
    contract: EndpointContract,
    *,
    prepared: PreparedGet,
    status: int,
    body: Any,
    page_size: int | None,
) -> EndpointExecutionResult:
    truncated = False
    if status < 400 and page_size is not None:
        rows, total, continuation = parse_page(contract, body=body)
        truncated = _page_has_more(
            total=total,
            collected_rows=len(rows),
            continuation=continuation,
            returned_rows=len(rows),
            page_size=page_size,
        )
    return EndpointExecutionResult(
        endpoint_name=contract.endpoint_name,
        request_url=prepared.url,
        query_params=prepared.query_params,
        response_status=status,
        response_body=body,
        truncated=truncated,
    )


def _page_size(
    contract: EndpointContract,
    *,
    page_policy: dict[str, Any] | None,
) -> int:
    pagination = contract.pagination
    if pagination is None:
        raise EndpointExecutionError(
            f"Endpoint {contract.endpoint_name} has no pagination contract."
        )
    requested = int(dict(page_policy or {}).get("page_size") or pagination.page_size)
    return max(1, min(requested, pagination.max_page_size))


def _consistent_total(previous: int | None, current: int | None) -> int | None:
    if previous is not None and current is not None and previous != current:
        raise EndpointExecutionError("Paginated endpoint total changed during traversal.")
    return current if current is not None else previous


def _page_has_more(
    *,
    total: int | None,
    collected_rows: int,
    continuation: bool | None,
    returned_rows: int,
    page_size: int,
) -> bool:
    if continuation is True:
        return True
    if total is not None and total > collected_rows:
        return True
    if continuation is False:
        return False
    return returned_rows >= page_size


def _validate_collected_total(*, total: int | None, collected_rows: int) -> None:
    if total is not None and collected_rows > total:
        raise EndpointExecutionError(
            "Paginated endpoint returned more rows than its declared total."
        )


def _rows_at_path(body: Any, *, path: str) -> list[Any]:
    value = _value_at_path(body, path=path)
    if not isinstance(value, list):
        raise EndpointExecutionError(
            "Paginated endpoint response row path must contain an array."
        )
    return list(value)


def _total(body: Any, *, path: str) -> int | None:
    if not path:
        return None
    value = _value_at_path(body, path=path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EndpointExecutionError(
            "Paginated endpoint total path must contain an integer."
        )
    return value


def _continuation(body: Any, *, path: str) -> Any:
    if not path:
        return None
    return _value_at_path(body, path=path)


def _value_at_path(body: Any, *, path: str) -> Any:
    value = body
    for segment in path.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise EndpointExecutionError(
                "Paginated endpoint response does not match its declared row path."
            )
        value = value[segment]
    return value
