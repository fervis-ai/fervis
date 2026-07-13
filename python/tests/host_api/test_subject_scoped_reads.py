from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path

import pytest

from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.read import ReadInvocation


def test_read_context_ref_is_small_json_safe_execution_handle() -> None:
    ref = ReadContextRef(
        scheme="django_principal",
        key="user_1",
        tenant_key="tenant_1",
    )

    assert ref.to_storage_dict() == {
        "scheme": "django_principal",
        "key": "user_1",
        "tenant_key": "tenant_1",
    }
    assert ReadContextRef.from_storage_dict(ref.to_storage_dict()) == ref


def test_read_context_ref_rejects_raw_auth_material() -> None:
    with pytest.raises(ValueError, match="unexpected ReadContextRef keys"):
        ReadContextRef.from_storage_dict(
            {
                "scheme": "django_principal",
                "key": "user_1",
                "headers": {"Authorization": "Bearer secret"},
            }
        )


def test_read_context_ref_rejects_removed_unmigrated_scheme() -> None:
    with pytest.raises(ValueError, match="unsupported ReadContextRef scheme"):
        ReadContextRef.from_storage_dict({"scheme": "unmigrated"})


def test_framework_adapters_do_not_import_host_application_packages() -> None:
    fervis_root = Path(__file__).resolve().parents[3] / "apps" / "fervis"
    adapter_roots = (
        fervis_root / "host_api" / "adapters",
        fervis_root / "interfaces",
    )
    forbidden = ("apps.accounts", "apps.retail_ops", "common.pagination")

    offenders = {
        path.relative_to(fervis_root).as_posix(): token
        for root in adapter_roots
        for path in root.rglob("*.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }

    assert offenders == {}


def test_host_api_runtime_has_one_configured_context_home() -> None:
    fervis_root = Path(__file__).resolve().parents[3] / "apps" / "fervis"
    offenders = {
        path.relative_to(fervis_root).as_posix()
        for path in fervis_root.rglob("*.py")
        if "configure_host_api_adapter" in path.read_text(encoding="utf-8")
    }

    assert offenders == set()


def test_read_invocation_preserves_selected_read_params() -> None:
    invocation = ReadInvocation(
        endpoint_name="get_order",
        path_params={"order_id": "ord_1"},
        query_params={"store": "abc"},
        page_policy={"mode": "single_page"},
    )

    assert invocation.to_execution_kwargs() == {
        "endpoint_name": "get_order",
        "path_params": {"order_id": "ord_1"},
        "query_params": {"store": "abc"},
        "page_policy": {"mode": "single_page"},
    }


def test_django_adapter_captures_request_user_as_read_context_ref() -> None:
    from fervis.host_api.adapters.django.adapter import DjangoHostApiAdapter

    adapter = DjangoHostApiAdapter(sources=())

    assert adapter.capture_read_context(
        SimpleNamespace(user=SimpleNamespace(pk=42))
    ) == (ReadContextRef(scheme="django_principal", key="42"))


def test_django_adapter_executes_read_with_resolved_subject(monkeypatch) -> None:
    import fervis.host_api.adapters.django.adapter as django_adapter_module

    calls: list[dict[str, object]] = []
    resolved_user = object()

    monkeypatch.setattr(
        django_adapter_module,
        "resolve_django_read_context_ref",
        lambda read_context_ref: resolved_user,
    )
    monkeypatch.setattr(
        django_adapter_module,
        "execute_get_endpoint",
        lambda **kwargs: _record_execution(calls, kwargs),
    )

    adapter = django_adapter_module.DjangoHostApiAdapter(sources=())
    result = adapter.execute_read(
        authority=_authority(scheme="django_principal", key="user_1"),
        invocation=ReadInvocation(
            endpoint_name="get_order",
            path_params={"order_id": "ord_1"},
            query_params={"store": "abc"},
            page_policy={"mode": "single_page"},
        ),
    )

    assert result.response_status == 200
    assert calls == [
        {
            "endpoint_name": "get_order",
            "user": resolved_user,
            "sources": (),
            "path_params": {"order_id": "ord_1"},
            "query_params": {"store": "abc"},
            "page_policy": {"mode": "single_page"},
            "transport_overlay": ReadTransportOverlay(),
        }
    ]


def test_django_adapter_http_mode_uses_shared_http_executor(monkeypatch) -> None:
    import fervis.host_api.adapters.django.adapter as django_adapter_module

    calls: list[dict[str, object]] = []
    contract = _fastapi_contract()
    monkeypatch.setattr(
        django_adapter_module,
        "get_endpoint_contract",
        lambda *args, **kwargs: contract,
    )
    monkeypatch.setattr(
        django_adapter_module,
        "execute_http_read",
        lambda **kwargs: _record_execution(calls, kwargs),
    )
    monkeypatch.setattr(
        django_adapter_module,
        "resolve_django_read_context_ref",
        lambda read_context_ref: (_ for _ in ()).throw(
            AssertionError("in-process subject resolution should not run")
        ),
    )

    adapter = django_adapter_module.DjangoHostApiAdapter(
        sources=(),
        auth_schema=_http_auth_schema(),
    )

    result = adapter.execute_read(
        authority=_authority(scheme="django_principal", key="user_1"),
        invocation=ReadInvocation(endpoint_name="get_order"),
    )

    assert result.response_status == 200
    assert calls == [
        {
            "contract": contract,
            "authority": _authority(scheme="django_principal", key="user_1"),
            "invocation": ReadInvocation(endpoint_name="get_order"),
            "config": _http_read_config(),
        }
    ]


def test_fastapi_adapter_fails_closed_for_non_anonymous_subject(tmp_path) -> None:
    from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter

    adapter = FastAPIHostApiAdapter(sources=(), project_root=tmp_path)

    with pytest.raises(ValueError, match="principal reauthorization"):
        adapter.execute_read(
            authority=_authority(scheme="fastapi_principal", key="user_1"),
            invocation=ReadInvocation(endpoint_name="list_orders"),
        )


def test_fastapi_adapter_executes_read_with_configured_principal_dependency(
    tmp_path,
) -> None:
    from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
    from fervis.project.integration import FastAPIAppSource

    _write_fastapi_subject_app(tmp_path)
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="default",
                import_paths=["app.main:app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema={
            "schema_version": "v0.1",
            "framework": "fastapi",
            "security": {"mode": "principal_reauthorization"},
            "transport": {"mode": "in_process"},
            "principal": {
                "source": "fastapi_dependency",
                "dependency": "app.deps:get_current_user",
                "id_attr": "id",
                "resolver": "app.users:get_user_by_id",
            },
        },
    )

    result = adapter.execute_read(
        authority=_authority(scheme="fastapi_principal", key="user_7"),
        invocation=ReadInvocation(endpoint_name="get_current_account_api_account__get"),
    )

    assert result.response_status == 200
    assert result.response_body == {"owner_id": "user_7"}


def test_concurrent_fastapi_reads_keep_each_principal_isolated(
    tmp_path,
) -> None:
    from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
    from fervis.project.integration import FastAPIAppSource

    _write_fastapi_subject_app(tmp_path)
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="default",
                import_paths=["app.main:app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema={
            "schema_version": "v0.1",
            "framework": "fastapi",
            "security": {"mode": "principal_reauthorization"},
            "transport": {"mode": "in_process"},
            "principal": {
                "source": "fastapi_dependency",
                "dependency": "app.deps:get_current_user",
                "id_attr": "id",
                "resolver": "app.users:get_user_by_id",
            },
        },
    )

    def read_as(principal_id: str) -> str:
        result = adapter.execute_read(
            authority=_authority(
                scheme="fastapi_principal",
                key=principal_id,
            ),
            invocation=ReadInvocation(
                endpoint_name="get_current_account_api_account__get"
            ),
        )
        return str(result.response_body["owner_id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed_principals = tuple(executor.map(read_as, ("user_a", "user_b")))

    assert observed_principals == ("user_a", "user_b")


def test_fastapi_adapter_executes_read_with_submitted_tenant_authority(
    tmp_path,
) -> None:
    from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
    from fervis.project.integration import FastAPIAppSource

    _write_fastapi_tenant_subject_app(tmp_path)
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="default",
                import_paths=["app.main:app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema={
            "schema_version": "v0.1",
            "framework": "fastapi",
            "security": {"mode": "principal_reauthorization"},
            "transport": {"mode": "in_process"},
            "principal": {
                "source": "fastapi_dependency",
                "dependency": "app.deps:get_current_user",
                "id_attr": "id",
                "resolver": "app.users:get_user_by_id",
            },
        },
    )

    result = adapter.execute_read(
        authority=ReadAuthority(
            tenant_id="tenant_a",
            read_context_ref=ReadContextRef(
                scheme="fastapi_principal",
                key="user_7",
            ),
        ),
        invocation=ReadInvocation(endpoint_name="get_current_account_api_account__get"),
    )

    assert result.response_status == 200
    assert result.response_body == {"owner_id": "user_7", "tenant_id": "tenant_a"}


def test_fastapi_adapter_uses_authority_tenant_not_read_context_tenant_key(
    tmp_path,
) -> None:
    from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
    from fervis.project.integration import FastAPIAppSource

    _write_fastapi_tenant_subject_app(tmp_path)
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="default",
                import_paths=["app.main:app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema={
            "schema_version": "v0.1",
            "framework": "fastapi",
            "security": {"mode": "principal_reauthorization"},
            "transport": {"mode": "in_process"},
            "principal": {
                "source": "fastapi_dependency",
                "dependency": "app.deps:get_current_user",
                "id_attr": "id",
                "resolver": "app.users:get_user_by_id",
            },
        },
    )

    result = adapter.execute_read(
        authority=ReadAuthority(
            tenant_id="tenant_b",
            read_context_ref=ReadContextRef(
                scheme="fastapi_principal",
                key="user_7",
                tenant_key="stale-or-host-specific",
            ),
        ),
        invocation=ReadInvocation(endpoint_name="get_current_account_api_account__get"),
    )

    assert result.response_status == 200
    assert result.response_body == {"owner_id": "user_7", "tenant_id": "tenant_b"}


def test_fastapi_adapter_rejects_unresolved_dependency_principal(tmp_path) -> None:
    from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
    from fervis.project.integration import FastAPIAppSource

    _write_fastapi_subject_app(tmp_path)
    (tmp_path / "app" / "users.py").write_text(
        "class User:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    del tenant_id\n"
        "    return None\n",
        encoding="utf-8",
    )
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="default",
                import_paths=["app.main:app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema={
            "schema_version": "v0.1",
            "framework": "fastapi",
            "security": {"mode": "principal_reauthorization"},
            "transport": {"mode": "in_process"},
            "principal": {
                "source": "fastapi_dependency",
                "dependency": "app.deps:get_current_user",
                "id_attr": "id",
                "resolver": "app.users:get_user_by_id",
            },
        },
    )

    with pytest.raises(ValueError, match="could not resolve principal"):
        adapter.execute_read(
            authority=_authority(scheme="fastapi_principal", key="missing"),
            invocation=ReadInvocation(
                endpoint_name="get_current_account_api_account__get"
            ),
        )


def test_flask_adapter_captures_g_current_user_as_read_context_ref(tmp_path) -> None:
    from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter
    from fervis.project.integration import FlaskAppSource

    _write_flask_login_subject_app(tmp_path)
    adapter = FlaskHostApiAdapter(
        sources=(
            FlaskAppSource(
                name="default",
                app="flask_subject.main:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema=_flask_auth_schema(),
    )
    app = _load_flask_app(tmp_path, "flask_subject.main:app")

    with app.test_request_context("/fervis/questions/"):
        from flask import g

        g.current_user = SimpleNamespace(id="user_7")

        read_context_ref = adapter.capture_read_context(request=None)

    assert read_context_ref == ReadContextRef(scheme="flask_principal", key="user_7")


def test_flask_adapter_executes_read_with_resolved_current_user(tmp_path) -> None:
    from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter
    from fervis.project.integration import FlaskAppSource

    _write_flask_login_subject_app(tmp_path)
    adapter = FlaskHostApiAdapter(
        sources=(
            FlaskAppSource(
                name="default",
                app="flask_subject.main:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema=_flask_auth_schema(),
    )

    result = adapter.execute_read(
        authority=_authority(scheme="flask_principal", key="user_7"),
        invocation=ReadInvocation(endpoint_name="get_current_account"),
    )

    assert result.response_status == 200
    assert result.response_body == {"owner_id": "user_7"}


def test_flask_adapter_executes_read_with_submitted_tenant_authority(
    tmp_path,
) -> None:
    from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter
    from fervis.project.integration import FlaskAppSource

    _write_flask_login_tenant_subject_app(tmp_path)
    adapter = FlaskHostApiAdapter(
        sources=(
            FlaskAppSource(
                name="default",
                app="flask_subject.main:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema=_flask_auth_schema(),
    )

    result = adapter.execute_read(
        authority=ReadAuthority(
            tenant_id="tenant_b",
            read_context_ref=ReadContextRef(
                scheme="flask_principal",
                key="user_7",
                tenant_key="stale-or-host-specific",
            ),
        ),
        invocation=ReadInvocation(endpoint_name="get_current_account"),
    )

    assert result.response_status == 200
    assert result.response_body == {"owner_id": "user_7", "tenant_id": "tenant_b"}


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    (
        ("Authorization", "Bearer jwt-token"),
        ("API-TOKEN", "api-token-value"),
        ("X-API-Key", "api-key-value"),
    ),
)
def test_flask_adapter_replays_captured_auth_headers_for_in_process_reads(
    monkeypatch,
    tmp_path,
    header_name: str,
    header_value: str,
) -> None:
    from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter
    from fervis.project.integration import FlaskAppSource

    monkeypatch.setenv("FERVIS_TEST_CREDENTIAL_KEY", "test-key")
    _write_flask_header_auth_app(tmp_path, header_name=header_name)
    adapter = FlaskHostApiAdapter(
        sources=(
            FlaskAppSource(
                name="default",
                app="flask_header_auth.main:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema=_flask_header_auth_schema(header_name),
    )
    credential = adapter.capture_delegated_credential(
        SimpleNamespace(headers={header_name: header_value})
    )

    result = adapter.execute_read(
        authority=ReadAuthority(
            tenant_id="tenant_1",
            read_context_ref=ReadContextRef(scheme="anonymous"),
            delegated_credential=credential,
        ),
        invocation=ReadInvocation(endpoint_name="list_orders"),
    )

    assert result.response_status == 200
    assert result.response_body == {"orders": [{"id": "ord_1"}]}


def test_flask_adapter_callable_capture_receives_request(tmp_path) -> None:
    from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter

    hooks_dir = tmp_path / "app"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text("", encoding="utf-8")
    (hooks_dir / "auth_hooks.py").write_text(
        "def capture(request):\n"
        "    return {'scheme': 'flask_principal', 'key': request.headers['X-User']}\n",
        encoding="utf-8",
    )
    adapter = FlaskHostApiAdapter(
        sources=(),
        project_root=tmp_path,
        auth_schema={
            "schema_version": "v0.1",
            "framework": "flask",
            "security": {"mode": "principal_reauthorization"},
            "transport": {"mode": "in_process"},
            "principal": {
                "source": "callable",
                "capture": "app.auth_hooks:capture",
                "resolver": "app.auth_hooks:capture",
            },
        },
    )

    with _project_import_context(tmp_path):
        read_context_ref = adapter.capture_read_context(
            SimpleNamespace(headers={"X-User": "user_9"})
        )

    assert read_context_ref == ReadContextRef(scheme="flask_principal", key="user_9")


def test_flask_adapter_http_mode_uses_shared_http_executor(
    monkeypatch, tmp_path
) -> None:
    import fervis.host_api.adapters.flask.adapter as flask_adapter_module

    calls: list[dict[str, object]] = []
    contract = _fastapi_contract()
    monkeypatch.setattr(
        flask_adapter_module.FlaskHostApiAdapter,
        "_endpoint_contract",
        lambda self, endpoint_name: contract,
    )
    monkeypatch.setattr(
        flask_adapter_module,
        "execute_http_read",
        lambda **kwargs: _record_execution(calls, kwargs),
    )
    monkeypatch.setattr(
        flask_adapter_module,
        "execute_get_endpoint",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("in-process Flask execution should not run")
        ),
    )

    adapter = flask_adapter_module.FlaskHostApiAdapter(
        sources=(),
        project_root=tmp_path,
        auth_schema={**_http_auth_schema(), "framework": "flask"},
    )

    result = adapter.execute_read(
        authority=_authority(scheme="flask_principal", key="user_1"),
        invocation=ReadInvocation(endpoint_name="list_orders"),
    )

    assert result.response_status == 200
    assert calls == [
        {
            "contract": contract,
            "authority": _authority(scheme="flask_principal", key="user_1"),
            "invocation": ReadInvocation(endpoint_name="list_orders"),
            "config": _http_read_config(),
        }
    ]


def test_fastapi_adapter_http_mode_uses_shared_http_executor(
    monkeypatch, tmp_path
) -> None:
    import fervis.host_api.adapters.fastapi.adapter as fastapi_adapter_module
    from fastapi import FastAPI
    from fervis.project.integration import FastAPIAppSource

    calls: list[dict[str, object]] = []
    contract = _fastapi_contract()
    monkeypatch.setattr(
        fastapi_adapter_module,
        "load_fastapi_app",
        lambda import_path: FastAPI(),
    )
    monkeypatch.setattr(
        fastapi_adapter_module.catalog,
        "endpoint_contracts_from_fastapi_app",
        lambda *args, **kwargs: (contract,),
    )
    monkeypatch.setattr(
        fastapi_adapter_module,
        "execute_http_read",
        lambda **kwargs: _record_execution(calls, kwargs),
    )
    adapter = fastapi_adapter_module.FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="default",
                import_paths=["app.main:app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
        auth_schema=_http_auth_schema(),
    )

    result = adapter.execute_read(
        authority=_authority(scheme="fastapi_principal", key="user_1"),
        invocation=ReadInvocation(endpoint_name="list_orders"),
    )

    assert result.response_status == 200
    assert calls == [
        {
            "contract": contract,
            "authority": _authority(scheme="fastapi_principal", key="user_1"),
            "invocation": ReadInvocation(endpoint_name="list_orders"),
            "config": _http_read_config(),
        }
    ]


def _execution_result():
    from fervis.host_api.contracts.ports import EndpointExecutionResult

    return EndpointExecutionResult(
        endpoint_name="get_order",
        request_url="/orders/ord_1/",
        query_params={"store": "abc"},
        response_status=200,
        response_body={"id": "ord_1"},
    )


def _record_execution(calls, kwargs):
    calls.append(kwargs)
    return _execution_result()


def _http_auth_schema() -> dict[str, object]:
    return {
        "schema_version": "v0.1",
        "framework": "fastapi",
        "security": {"mode": "principal_reauthorization"},
        "transport": {
            "mode": "http",
            "base_url_env": "FERVIS_HOST_API_BASE_URL",
            "request_overlay_source": "config.fervis_host_hooks:http_request_overlay",
        },
        "principal": {
            "source": "fastapi_dependency",
            "dependency": "app.api.deps:get_current_user",
            "id_attr": "id",
        },
    }


def _http_read_config():
    from fervis.host_api.adapters.http import HttpReadExecutionConfig

    return HttpReadExecutionConfig(
        base_url_env="FERVIS_HOST_API_BASE_URL",
        request_overlay_source="config.fervis_host_hooks:http_request_overlay",
    )


def _fastapi_contract() -> EndpointContract:
    return EndpointContract(
        endpoint_name="list_orders",
        url_name="orders",
        method="GET",
        path_template="/api/orders/",
        docstring="Orders.",
        view_class="OrdersView",
    )


def test_configured_catalog_is_not_filtered_by_subject() -> None:
    from fervis.host_api.context import HostApiContext

    public_contract = EndpointContract(
        endpoint_name="list_public_orders",
        url_name="public-orders",
        method="GET",
        path_template="/orders/public/",
        docstring="Public orders.",
        view_class="PublicOrders",
        public_access=True,
    )
    private_contract = EndpointContract(
        endpoint_name="list_private_orders",
        url_name="private-orders",
        method="GET",
        path_template="/orders/private/",
        docstring="Private orders.",
        view_class="PrivateOrders",
        public_access=False,
        admin_access=False,
        staff_access=False,
        agent_access=False,
    )

    context = HostApiContext(
        adapter=_ContractOnlyHostApiAdapter(
            public_contract,
            private_contract,
        )
    )

    assert [item.endpoint_name for item in context.describe_sources()] == [
        "list_public_orders",
        "list_private_orders",
    ]


class _ContractOnlyHostApiAdapter:
    def __init__(self, *contracts):
        self.contracts = tuple(contracts)

    def describe_sources(self):
        return self.contracts

    def capture_read_context(self, request):
        del request
        return ReadContextRef(scheme="delegated_capability", key="user_1")

    def capture_delegated_credential(self, request):
        del request
        return None

    def execute_read(self, *, authority, invocation):
        raise AssertionError((authority, invocation))


def _authority(
    *,
    scheme,
    key,
    tenant_id: str = "tenant_1",
    tenant_key: str | None = "tenant_1",
) -> ReadAuthority:
    return ReadAuthority(
        tenant_id=tenant_id,
        read_context_ref=ReadContextRef(
            scheme=scheme,
            key=key,
            tenant_key=tenant_key,
        ),
    )


def _write_fastapi_subject_app(root) -> None:
    app_dir = root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    del tenant_id\n"
        "    return User(user_id)\n",
        encoding="utf-8",
    )
    (app_dir / "deps.py").write_text(
        "from app.users import User\n\n"
        "def get_current_user():\n"
        "    return User('anonymous')\n",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        "from fastapi import Depends, FastAPI\n"
        "from pydantic import BaseModel\n"
        "from app.deps import get_current_user\n\n"
        "class AccountResponse(BaseModel):\n"
        "    owner_id: str\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/api/account/', response_model=AccountResponse)\n"
        "def get_current_account(user=Depends(get_current_user)):\n"
        "    return AccountResponse(owner_id=user.id)\n",
        encoding="utf-8",
    )


def _write_fastapi_tenant_subject_app(root) -> None:
    app_dir = root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    def __init__(self, id, tenant_id):\n"
        "        self.id = id\n"
        "        self.tenant_id = tenant_id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    return User(user_id, tenant_id)\n",
        encoding="utf-8",
    )
    (app_dir / "deps.py").write_text(
        "from app.users import User\n\n"
        "def get_current_user():\n"
        "    return User('anonymous', 'anonymous_tenant')\n",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        "from fastapi import Depends, FastAPI\n"
        "from pydantic import BaseModel\n"
        "from app.deps import get_current_user\n\n"
        "class AccountResponse(BaseModel):\n"
        "    owner_id: str\n"
        "    tenant_id: str\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/api/account/', response_model=AccountResponse)\n"
        "def get_current_account(user=Depends(get_current_user)):\n"
        "    return AccountResponse(owner_id=user.id, tenant_id=user.tenant_id)\n",
        encoding="utf-8",
    )


def _flask_auth_schema() -> dict[str, object]:
    return {
        "schema_version": "v0.1",
        "framework": "flask",
        "security": {"mode": "principal_reauthorization"},
        "transport": {"mode": "in_process"},
        "principal": {
            "source": "flask_g",
            "id_attr": "id",
            "resolver": "flask_subject.users:get_user_by_id",
        },
    }


def _flask_header_auth_schema(header_name: str) -> dict[str, object]:
    schema = _flask_auth_schema()
    schema["credentials"] = {
        "source": "captured_request_headers",
        "headers": [header_name],
        "ttl_seconds": 900,
        "encryption_key_env": "FERVIS_TEST_CREDENTIAL_KEY",
    }
    return schema


def _load_flask_app(root, app_target):
    from fervis.host_api.adapters.flask.loading import import_flask_app

    return import_flask_app(app_target, project_root=root)


def _project_import_context(root):
    from fervis.project.importing import project_import_context

    return project_import_context(root)


def _write_flask_login_subject_app(root) -> None:
    app_dir = root / "flask_subject"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    is_active = True\n"
        "    is_authenticated = True\n"
        "    is_anonymous = False\n"
        "    def __init__(self, id, tenant_id='tenant_1'):\n"
        "        self.id = id\n"
        "        self.tenant_id = tenant_id\n"
        "    def get_id(self):\n"
        "        return self.id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    del tenant_id\n"
        "    return User(user_id)\n",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        "from flask import Flask, g, jsonify\n"
        "from flask_subject.users import get_user_by_id\n\n"
        "app = Flask(__name__)\n"
        "@app.get('/api/account/')\n"
        "def get_current_account():\n"
        "    return jsonify({'owner_id': g.current_user.id})\n",
        encoding="utf-8",
    )


def _write_flask_login_tenant_subject_app(root) -> None:
    app_dir = root / "flask_subject"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    is_active = True\n"
        "    is_authenticated = True\n"
        "    is_anonymous = False\n"
        "    def __init__(self, id, tenant_id):\n"
        "        self.id = id\n"
        "        self.tenant_id = tenant_id\n"
        "    def get_id(self):\n"
        "        return self.id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    return User(user_id, tenant_id)\n",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        "from flask import Flask, g, jsonify\n"
        "from flask_subject.users import get_user_by_id\n\n"
        "app = Flask(__name__)\n"
        "@app.get('/api/account/')\n"
        "def get_current_account():\n"
        "    return jsonify({"
        "'owner_id': g.current_user.id, "
        "'tenant_id': g.current_user.tenant_id"
        "})\n",
        encoding="utf-8",
    )


def _write_flask_header_auth_app(root, *, header_name: str) -> None:
    app_dir = root / "flask_header_auth"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "users.py").write_text(
        "class User:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n\n"
        "def get_user_by_id(user_id, tenant_id):\n"
        "    del tenant_id\n"
        "    return User(user_id)\n",
        encoding="utf-8",
    )
    (app_dir / "main.py").write_text(
        "from flask import Flask, jsonify, request\n\n"
        "app = Flask(__name__)\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        f"    if not request.headers.get({header_name!r}):\n"
        "        return jsonify({'error': 'missing auth'}), 401\n"
        "    return jsonify({'orders': [{'id': 'ord_1'}]})\n",
        encoding="utf-8",
    )
