from __future__ import annotations

import pytest

from fervis.host_api.contracts import (
    EndpointContract,
    PaginationContract,
    PaginationKind,
    ParameterContract,
)
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.ports import EndpointExecutionError
from fervis.host_api.adapters.get_execution import prepare_get_endpoint


def test_http_read_executor_returns_endpoint_execution_result(monkeypatch) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )

    calls = []

    def fake_get(url, *, params, headers, cookies, timeout):
        calls.append(
            {
                "method": "GET",
                "url": url,
                "params": dict(params),
                "headers": dict(headers),
                "cookies": dict(cookies),
                "timeout": timeout,
            }
        )
        return _HttpResponse(200, {"data": [{"id": "ord_1"}]})

    monkeypatch.setattr(
        "fervis.host_api.adapters.http._get",
        fake_get,
    )
    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")

    result = execute_http_read(
        contract=_orders_contract(),
        authority=_authority(),
        invocation=ReadInvocation(
            endpoint_name="list_orders",
            query_params={"status": "open"},
        ),
        config=HttpReadExecutionConfig(
            base_url_env="FERVIS_HOST_API_BASE_URL",
            request_overlay_source=(
                "tests.host_api.test_http_read_execution:http_request_overlay"
            ),
            auth_query_params=("auth_token",),
        ),
    )

    assert result.endpoint_name == "list_orders"
    assert result.request_url == "/api/orders/"
    assert result.query_params == {"status": "open"}
    assert result.response_status == 200
    assert result.response_body == {"data": [{"id": "ord_1"}]}
    assert calls == [
        {
            "method": "GET",
            "url": "https://api.example.test/api/orders/",
            "params": {"status": "open", "auth_token": "token_user_1"},
            "headers": {"X-Fervis-Subject": "user_1"},
            "cookies": {"sessionid": "session_user_1"},
            "timeout": 30,
        }
    ]


def test_http_read_executor_rejects_auth_query_param_collision(monkeypatch) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )

    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")

    with pytest.raises(EndpointExecutionError, match="overlap selected query params"):
        execute_http_read(
            contract=_orders_contract(pagination=_pagination()),
            authority=_authority(),
            invocation=ReadInvocation(
                endpoint_name="list_orders",
                query_params={"status": "open"},
            ),
            config=HttpReadExecutionConfig(
                base_url_env="FERVIS_HOST_API_BASE_URL",
                request_overlay_source=(
                    "tests.host_api.test_http_read_execution:"
                    "colliding_http_request_overlay"
                ),
            ),
        )


def test_http_read_executor_rejects_pagination_query_param_collision(
    monkeypatch,
) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )

    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")

    with pytest.raises(EndpointExecutionError, match="limit"):
        execute_http_read(
            contract=_orders_contract(pagination=_pagination()),
            authority=_authority(),
            invocation=ReadInvocation(
                endpoint_name="list_orders",
                page_policy={"mode": "all_pages"},
            ),
            config=HttpReadExecutionConfig(
                base_url_env="FERVIS_HOST_API_BASE_URL",
                request_overlay_source=(
                    "tests.host_api.test_http_read_execution:"
                    "pagination_colliding_http_request_overlay"
                ),
            ),
        )


def test_http_read_executor_rejects_undeclared_overlay_query_params(
    monkeypatch,
) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )

    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(
        "fervis.host_api.adapters.http._get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe overlay query params must be rejected before I/O")
        ),
    )

    with pytest.raises(EndpointExecutionError, match="overlay query"):
        execute_http_read(
            contract=_orders_contract(),
            authority=_authority(),
            invocation=ReadInvocation(
                endpoint_name="list_orders",
                query_params={"status": "open"},
            ),
            config=HttpReadExecutionConfig(
                base_url_env="FERVIS_HOST_API_BASE_URL",
                request_overlay_source=(
                    "tests.host_api.test_http_read_execution:"
                    "row_scope_mutating_http_request_overlay"
                ),
            ),
        )


def test_get_execution_rejects_missing_required_query_params() -> None:
    with pytest.raises(EndpointExecutionError, match="Missing required query params"):
        prepare_get_endpoint(
            _orders_contract(
                query_params=(
                    ParameterContract(
                        name="customer_id",
                        type="string",
                        required=True,
                    ),
                )
            ),
            path_params={},
            query_params={},
        )


def test_get_execution_rejects_path_param_traversal_and_query_smuggling() -> None:
    contract = EndpointContract(
        endpoint_name="get_order",
        url_name="order",
        method="GET",
        path_template="/api/orders/{order_id}/",
        docstring="Order detail.",
        view_class="OrderView",
        path_params=(
            ParameterContract(
                name="order_id",
                type="string",
                required=True,
                source="path",
            ),
        ),
    )

    for unsafe_value in ("../admin/users", "ord_1?include_deleted=true"):
        with pytest.raises(EndpointExecutionError, match="path param"):
            prepare_get_endpoint(
                contract,
                path_params={"order_id": unsafe_value},
                query_params={},
            )


def test_get_execution_allows_dotted_path_param_identifiers() -> None:
    contract = EndpointContract(
        endpoint_name="get_order",
        url_name="order",
        method="GET",
        path_template="/api/orders/{order_id}/",
        docstring="Order detail.",
        view_class="OrderView",
        path_params=(
            ParameterContract(
                name="order_id",
                type="string",
                required=True,
                source="path",
            ),
        ),
    )

    prepared = prepare_get_endpoint(
        contract,
        path_params={"order_id": "sku..2026"},
        query_params={},
    )

    assert prepared.url == "/api/orders/sku..2026/"


def test_http_read_executor_preserves_non_json_response_body(monkeypatch) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )

    monkeypatch.setattr(
        "fervis.host_api.adapters.http._get",
        lambda *args, **kwargs: _TextHttpResponse(
            403,
            "forbidden",
            "text/plain; charset=utf-8",
        ),
    )
    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")

    result = execute_http_read(
        contract=_orders_contract(),
        authority=_authority(),
        invocation=ReadInvocation(endpoint_name="list_orders"),
        config=HttpReadExecutionConfig(base_url_env="FERVIS_HOST_API_BASE_URL"),
    )

    assert result.response_status == 403
    assert result.response_body == {
        "contentType": "text/plain; charset=utf-8",
        "text": "forbidden",
    }


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    (
        ("Authorization", "Bearer jwt-token"),
        ("API-TOKEN", "api-token-value"),
        ("X-API-Key", "api-key-value"),
    ),
)
def test_captured_header_credential_replays_standard_auth_headers(
    monkeypatch,
    header_name: str,
    header_value: str,
) -> None:
    from fervis.host_api.credentials import (
        CapturedHeaderCredentialPolicy,
        capture_header_credential,
        overlay_from_header_credential,
    )

    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")
    credential = capture_header_credential(
        request_headers={header_name: header_value, "X-Other": "ignored"},
        policy=CapturedHeaderCredentialPolicy(
            headers=(header_name,),
            encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
        ),
    )

    assert credential is not None
    overlay = overlay_from_header_credential(
        credential,
        policy=CapturedHeaderCredentialPolicy(
            headers=(header_name,),
            encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
        ),
    )

    assert overlay.headers == {header_name: header_value}


def test_captured_header_credential_fails_when_configured_headers_are_missing(
    monkeypatch,
) -> None:
    from fervis.host_api.credentials import (
        CapturedHeaderCredentialPolicy,
        capture_header_credential,
    )

    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")

    with pytest.raises(EndpointExecutionError, match="missing configured auth headers"):
        capture_header_credential(
            request_headers={"X-Other": "ignored"},
            policy=CapturedHeaderCredentialPolicy(
                headers=("Authorization", "API-TOKEN"),
                encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
            ),
        )


def test_captured_header_credential_requires_every_configured_header(
    monkeypatch,
) -> None:
    from fervis.host_api.credentials import (
        CapturedHeaderCredentialPolicy,
        capture_header_credential,
    )

    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")

    with pytest.raises(EndpointExecutionError, match="X-Tenant"):
        capture_header_credential(
            request_headers={"Authorization": "Bearer token"},
            policy=CapturedHeaderCredentialPolicy(
                headers=("Authorization", "X-Tenant"),
                encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
            ),
        )


def test_delegated_credential_replay_requires_every_configured_header(
    monkeypatch,
) -> None:
    from fervis.host_api.credentials import (
        CapturedHeaderCredentialPolicy,
        capture_header_credential,
        overlay_from_header_credential,
    )

    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")
    credential = capture_header_credential(
        request_headers={"Authorization": "Bearer token"},
        policy=CapturedHeaderCredentialPolicy(
            headers=("Authorization",),
            encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
        ),
    )

    with pytest.raises(EndpointExecutionError, match="X-Tenant"):
        overlay_from_header_credential(
            credential,
            policy=CapturedHeaderCredentialPolicy(
                headers=("Authorization", "X-Tenant"),
                encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
            ),
        )


def test_http_read_executor_applies_delegated_credential_before_request(
    monkeypatch,
) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )
    from fervis.host_api.credentials import (
        CapturedHeaderCredentialPolicy,
        capture_header_credential,
    )

    calls = []
    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")
    monkeypatch.setattr(
        "fervis.host_api.adapters.http._get",
        lambda url, *, params, headers, cookies, timeout: (
            calls.append(
                {
                    "url": url,
                    "params": dict(params),
                    "headers": dict(headers),
                    "cookies": dict(cookies),
                    "timeout": timeout,
                }
            )
            or _HttpResponse(200, {"data": [{"id": "ord_1"}]})
        ),
    )

    credential = capture_header_credential(
        request_headers={"Authorization": "Bearer jwt-token"},
        policy=CapturedHeaderCredentialPolicy(
            headers=("Authorization",),
            encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
        ),
    )

    result = execute_http_read(
        contract=_orders_contract(),
        authority=ReadAuthority(
            tenant_id="tenant_1",
            read_context_ref=ReadContextRef(scheme="flask_principal", key="user_1"),
            delegated_credential=credential,
        ),
        invocation=ReadInvocation(endpoint_name="list_orders"),
        config=HttpReadExecutionConfig(
            base_url_env="FERVIS_HOST_API_BASE_URL",
            credential_policy=CapturedHeaderCredentialPolicy(
                headers=("Authorization",),
                encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
            ),
        ),
    )

    assert result.response_status == 200
    assert calls == [
        {
            "url": "https://api.example.test/api/orders/",
            "params": {},
            "headers": {"Authorization": "Bearer jwt-token"},
            "cookies": {},
            "timeout": 30,
        }
    ]


def test_http_read_executor_rejects_delegated_header_overlay_collision(
    monkeypatch,
) -> None:
    from fervis.host_api.adapters.http import (
        HttpReadExecutionConfig,
        execute_http_read,
    )
    from fervis.host_api.credentials import (
        CapturedHeaderCredentialPolicy,
        capture_header_credential,
    )

    monkeypatch.setenv("FERVIS_HOST_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")
    credential = capture_header_credential(
        request_headers={"Authorization": "Bearer jwt-token"},
        policy=CapturedHeaderCredentialPolicy(
            headers=("Authorization",),
            encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
        ),
    )

    with pytest.raises(EndpointExecutionError, match="overlay headers overlap"):
        execute_http_read(
            contract=_orders_contract(),
            authority=ReadAuthority(
                tenant_id="tenant_1",
                read_context_ref=ReadContextRef(
                    scheme="flask_principal",
                    key="user_1",
                ),
                delegated_credential=credential,
            ),
            invocation=ReadInvocation(endpoint_name="list_orders"),
            config=HttpReadExecutionConfig(
                base_url_env="FERVIS_HOST_API_BASE_URL",
                request_overlay_source=(
                    "tests.host_api.test_http_read_execution:"
                    "authorization_http_request_overlay"
                ),
                credential_policy=CapturedHeaderCredentialPolicy(
                    headers=("Authorization",),
                    encryption_key_env="FERVIS_TEST_CREDENTIAL_KEY",
                ),
            ),
        )


def http_request_overlay(authority, invocation):
    assert invocation.endpoint_name == "list_orders"
    read_context_ref = authority.read_context_ref
    return {
        "headers": {"X-Fervis-Subject": read_context_ref.key},
        "query_params": {"auth_token": f"token_{read_context_ref.key}"},
        "cookies": {"sessionid": f"session_{read_context_ref.key}"},
    }


def colliding_http_request_overlay(authority, invocation):
    del authority, invocation
    return {"query_params": {"status": "closed"}}


def authorization_http_request_overlay(authority, invocation):
    del authority, invocation
    return {"headers": {"Authorization": "Bearer custom"}}


def pagination_colliding_http_request_overlay(authority, invocation):
    del authority, invocation
    return {"query_params": {"limit": "999"}}


def row_scope_mutating_http_request_overlay(authority, invocation):
    del authority, invocation
    return {"query_params": {"include_deleted": "true"}}


def _authority() -> ReadAuthority:
    return ReadAuthority(
        tenant_id="tenant_1",
        read_context_ref=ReadContextRef(
            scheme="delegated_capability",
            key="user_1",
            tenant_key="tenant_1",
        ),
    )


def _orders_contract(
    *,
    query_params: tuple[ParameterContract, ...] | None = None,
    pagination: PaginationContract | None = None,
) -> EndpointContract:
    return EndpointContract(
        endpoint_name="list_orders",
        url_name="orders",
        method="GET",
        path_template="/api/orders/",
        docstring="Orders.",
        view_class="OrdersView",
        query_params=query_params
        if query_params is not None
        else (ParameterContract(name="status", type="string"),),
        pagination=pagination,
    )


def _pagination() -> PaginationContract:
    return PaginationContract(
        kind=PaginationKind.OFFSET,
        position_query_param="offset",
        page_size_query_param="limit",
        results_path="data",
        page_size=50,
        max_page_size=200,
        continuation_path="pagination.has_more",
    )


class _HttpResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _TextHttpResponse:
    def __init__(self, status_code: int, text: str, content_type: str):
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}

    def json(self):
        raise ValueError("not json")
