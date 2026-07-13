from __future__ import annotations

import asyncio
from pathlib import Path
from threading import get_ident

from fastapi import FastAPI, Request

from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
from fervis.host_api.adapters.fastapi.executor import FastAPIApplicationRuntime
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.execution import ReadTransportOverlay
from fervis.host_api.contracts.read import ReadInvocation
from fervis.project.integration import FastAPIAppSource


def test_fastapi_executor_calls_configured_get_endpoint_with_query_params(
    tmp_path: Path,
) -> None:
    _write_fastapi_app(tmp_path)
    adapter = _adapter(tmp_path)

    result = _execute(
        adapter,
        endpoint_name="get_order_count",
        query_params={"store": "ABC"},
    )
    adapter.close()

    assert result.endpoint_name == "get_order_count"
    assert result.request_url == "/api/orders/count/"
    assert result.query_params == {"store": "ABC"}
    assert result.response_status == 200
    assert result.response_body == {"count": 42, "store": "ABC"}


def test_fastapi_executor_calls_configured_get_endpoint_with_path_params(
    tmp_path: Path,
) -> None:
    _write_fastapi_app(tmp_path)
    adapter = _adapter(tmp_path)

    result = _execute(
        adapter,
        endpoint_name="get_order",
        path_params={"order_id": "ord_1"},
    )
    adapter.close()

    assert result.endpoint_name == "get_order"
    assert result.request_url == "/api/orders/ord_1/"
    assert result.response_status == 200
    assert result.response_body == {"id": "ord_1"}


def test_fastapi_executor_preserves_non_json_response_body(tmp_path: Path) -> None:
    _write_fastapi_app(tmp_path)
    adapter = _adapter(tmp_path)

    result = _execute(
        adapter,
        endpoint_name="get_plain_error",
    )
    adapter.close()

    assert result.endpoint_name == "get_plain_error"
    assert result.response_status == 403
    assert result.response_body == {
        "contentType": "text/plain; charset=utf-8",
        "text": "forbidden",
    }


def test_fastapi_executor_runs_the_host_application_lifespan(tmp_path: Path) -> None:
    _write_fastapi_app(tmp_path)
    adapter = _adapter(tmp_path)

    result = _execute(
        adapter,
        endpoint_name="get_runtime_state",
    )
    adapter.close()

    assert result.response_status == 200
    assert result.response_body == {"status": "ready"}


def test_fastapi_adapter_owns_one_application_lifespan_until_closed(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.txt"
    _write_lifecycle_app(tmp_path, events_path=events_path)
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="runtime",
                import_paths=["runtime_app.main:create_app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert not events_path.exists()

    adapter.describe_sources()
    adapter.describe_sources()

    assert events_path.read_text(encoding="utf-8").splitlines() == ["factory"]

    first = _execute(adapter, endpoint_name="runtime_state")
    second = _execute(adapter, endpoint_name="runtime_state")

    assert first.response_body == {"status": "ready", "request_count": 1}
    assert second.response_body == {"status": "ready", "request_count": 2}
    assert events_path.read_text(encoding="utf-8").splitlines() == [
        "factory",
        "startup",
        "request",
        "request",
    ]

    adapter.close()
    adapter.close()

    assert events_path.read_text(encoding="utf-8").splitlines() == [
        "factory",
        "startup",
        "request",
        "request",
        "shutdown",
    ]


def test_closing_unused_fastapi_adapter_does_not_start_application(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.txt"
    _write_lifecycle_app(tmp_path, events_path=events_path)
    adapter = FastAPIHostApiAdapter(
        sources=(
            FastAPIAppSource(
                name="runtime",
                import_paths=["runtime_app.main:create_app"],
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    adapter.close()

    assert not events_path.exists()


def test_fastapi_runtime_does_not_carry_cookies_between_reads(tmp_path: Path) -> None:
    app = FastAPI()

    @app.get("/api/session/", operation_id="read_session")
    def read_session(request: Request) -> dict[str, str]:
        return {"session": request.cookies.get("session", "")}

    runtime = FastAPIApplicationRuntime(app, project_root=tmp_path)
    contract = EndpointContract(
        endpoint_name="read_session",
        url_name="read_session",
        method="GET",
        path_template="/api/session/",
        docstring="",
        view_class="",
    )

    first = runtime.execute_get(
        contract=contract,
        transport_overlay=ReadTransportOverlay(cookies={"session": "principal-a"}),
    )
    second = runtime.execute_get(contract=contract)
    runtime.close()

    assert first.response_body == {"session": "principal-a"}
    assert second.response_body == {"session": ""}


def test_fastapi_runtime_keeps_lifespan_output_out_of_cli_envelopes(
    tmp_path: Path,
    capsys,
) -> None:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        print("host startup")
        yield
        print("host shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/value/", operation_id="read_value")
    def read_value() -> dict[str, int]:
        print("host request")
        return {"value": 1}

    runtime = FastAPIApplicationRuntime(app, project_root=tmp_path)
    contract = EndpointContract(
        endpoint_name="read_value",
        url_name="read_value",
        method="GET",
        path_template="/api/value/",
        docstring="",
        view_class="",
    )

    result = runtime.execute_get(contract=contract)
    runtime.close()

    assert result.response_body == {"value": 1}
    assert capsys.readouterr().out == ""


def test_fastapi_runtime_runs_lifespan_on_the_calling_worker_thread(
    tmp_path: Path,
) -> None:
    from contextlib import asynccontextmanager

    lifecycle_threads: list[int] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        lifecycle_threads.append(get_ident())
        yield
        lifecycle_threads.append(get_ident())

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/value/", operation_id="read_value")
    async def read_value() -> dict[str, int]:
        return {"value": 1}

    runtime = FastAPIApplicationRuntime(app, project_root=tmp_path)
    contract = EndpointContract(
        endpoint_name="read_value",
        url_name="read_value",
        method="GET",
        path_template="/api/value/",
        docstring="",
        view_class="",
    )
    worker_thread = get_ident()

    runtime.execute_get(contract=contract)
    runtime.close()

    assert lifecycle_threads == [worker_thread, worker_thread]


def test_fastapi_runtime_uses_python_3_10_event_loop_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delattr(asyncio, "Runner", raising=False)
    app = FastAPI()

    @app.get("/api/value/", operation_id="read_value")
    async def read_value() -> dict[str, int]:
        return {"value": 1}

    runtime = FastAPIApplicationRuntime(app, project_root=tmp_path)
    contract = EndpointContract(
        endpoint_name="read_value",
        url_name="read_value",
        method="GET",
        path_template="/api/value/",
        docstring="",
        view_class="",
    )

    result = runtime.execute_get(contract=contract)
    runtime.close()

    assert result.response_body == {"value": 1}


def _sources() -> tuple[FastAPIAppSource, ...]:
    return (
        FastAPIAppSource(
            name="commerce",
            import_paths=["service.main:app"],
            path_prefixes=["/api/"],
        ),
    )


def _adapter(root: Path) -> FastAPIHostApiAdapter:
    return FastAPIHostApiAdapter(sources=_sources(), project_root=root)


def _execute(
    adapter: FastAPIHostApiAdapter,
    *,
    endpoint_name: str,
    path_params: dict[str, object] | None = None,
    query_params: dict[str, object] | None = None,
):
    authority = ReadAuthority(
        tenant_id="test",
        read_context_ref=ReadContextRef(scheme="anonymous"),
    )
    return adapter.execute_read(
        authority=authority,
        invocation=ReadInvocation(
            endpoint_name=endpoint_name,
            path_params=path_params or {},
            query_params=query_params or {},
        ),
    )


def _write_lifecycle_app(root: Path, *, events_path: Path) -> None:
    package = root / "runtime_app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from contextlib import asynccontextmanager\n"
        "from pathlib import Path\n\n"
        "from fastapi import FastAPI\n\n"
        f"EVENTS_PATH = Path({str(events_path)!r})\n\n"
        "def record(event: str) -> None:\n"
        "    with EVENTS_PATH.open('a', encoding='utf-8') as stream:\n"
        "        stream.write(event + '\\n')\n\n"
        "def create_app() -> FastAPI:\n"
        "    record('factory')\n"
        "    @asynccontextmanager\n"
        "    async def lifespan(app: FastAPI):\n"
        "        app.state.status = 'ready'\n"
        "        app.state.request_count = 0\n"
        "        record('startup')\n"
        "        yield\n"
        "        record('shutdown')\n"
        "    app = FastAPI(lifespan=lifespan)\n"
        "    @app.get('/api/runtime/', operation_id='runtime_state')\n"
        "    def runtime_state():\n"
        "        app.state.request_count += 1\n"
        "        record('request')\n"
        "        return {\n"
        "            'status': app.state.status,\n"
        "            'request_count': app.state.request_count,\n"
        "        }\n"
        "    return app\n",
        encoding="utf-8",
    )


def _write_fastapi_app(root: Path) -> None:
    package = root / "service"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from contextlib import asynccontextmanager\n"
        "from fastapi.responses import PlainTextResponse\n"
        "from pydantic import BaseModel\n\n\n"
        "class OrderCountResponse(BaseModel):\n"
        "    count: int\n"
        "    store: str\n\n\n"
        "class OrderResponse(BaseModel):\n"
        "    id: str\n\n\n"
        "@asynccontextmanager\n"
        "async def lifespan(app):\n"
        "    app.state.runtime_status = 'ready'\n"
        "    yield\n\n\n"
        "app = FastAPI(lifespan=lifespan)\n\n\n"
        "@app.get(\n"
        "    '/api/orders/count/',\n"
        "    operation_id='get_order_count',\n"
        "    tags=['orders'],\n"
        "    response_model=OrderCountResponse,\n"
        ")\n"
        "def get_order_count(store: str) -> OrderCountResponse:\n"
        "    return OrderCountResponse(count=42, store=store)\n\n\n"
        "@app.get(\n"
        "    '/api/orders/{order_id}/',\n"
        "    operation_id='get_order',\n"
        "    tags=['orders'],\n"
        "    response_model=OrderResponse,\n"
        ")\n"
        "def get_order(order_id: str) -> OrderResponse:\n"
        "    return OrderResponse(id=order_id)\n\n\n"
        "@app.get('/api/plain-error/', operation_id='get_plain_error')\n"
        "def get_plain_error():\n"
        "    return PlainTextResponse('forbidden', status_code=403)\n\n\n"
        "@app.get('/api/runtime-state/', operation_id='get_runtime_state')\n"
        "def get_runtime_state():\n"
        "    return {'status': app.state.runtime_status}\n",
        encoding="utf-8",
    )
