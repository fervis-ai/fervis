from __future__ import annotations

from pathlib import Path

from fervis.host_api.adapters.fastapi.executor import execute_get_endpoint
from fervis.project.integration import FastAPIAppSource


def test_fastapi_executor_calls_configured_get_endpoint_with_query_params(
    tmp_path: Path,
) -> None:
    _write_fastapi_app(tmp_path)

    result = execute_get_endpoint(
        endpoint_name="get_order_count",
        sources=_sources(),
        project_root=tmp_path,
        query_params={"store": "ABC"},
    )

    assert result.endpoint_name == "get_order_count"
    assert result.request_url == "/api/orders/count/"
    assert result.query_params == {"store": "ABC"}
    assert result.response_status == 200
    assert result.response_body == {"count": 42, "store": "ABC"}


def test_fastapi_executor_calls_configured_get_endpoint_with_path_params(
    tmp_path: Path,
) -> None:
    _write_fastapi_app(tmp_path)

    result = execute_get_endpoint(
        endpoint_name="get_order",
        sources=_sources(),
        project_root=tmp_path,
        path_params={"order_id": "ord_1"},
    )

    assert result.endpoint_name == "get_order"
    assert result.request_url == "/api/orders/ord_1/"
    assert result.response_status == 200
    assert result.response_body == {"id": "ord_1"}


def test_fastapi_executor_preserves_non_json_response_body(tmp_path: Path) -> None:
    _write_fastapi_app(tmp_path)

    result = execute_get_endpoint(
        endpoint_name="get_plain_error",
        sources=_sources(),
        project_root=tmp_path,
    )

    assert result.endpoint_name == "get_plain_error"
    assert result.response_status == 403
    assert result.response_body == {
        "contentType": "text/plain; charset=utf-8",
        "text": "forbidden",
    }


def _sources() -> tuple[FastAPIAppSource, ...]:
    return (
        FastAPIAppSource(
            name="commerce",
            import_paths=["service.main:app"],
            path_prefixes=["/api/"],
        ),
    )


def _write_fastapi_app(root: Path) -> None:
    package = root / "service"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "from fastapi.responses import PlainTextResponse\n"
        "from pydantic import BaseModel\n\n\n"
        "class OrderCountResponse(BaseModel):\n"
        "    count: int\n"
        "    store: str\n\n\n"
        "class OrderResponse(BaseModel):\n"
        "    id: str\n\n\n"
        "app = FastAPI()\n\n\n"
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
        "    return PlainTextResponse('forbidden', status_code=403)\n",
        encoding="utf-8",
    )
