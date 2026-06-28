from __future__ import annotations

from pathlib import Path

import pytest

from fervis.host_api.adapters.flask.executor import execute_get_endpoint
from fervis.host_api.adapters.flask.principal import FlaskPrincipalOverride
from fervis.host_api.contracts.ports import EndpointExecutionError
from fervis.project.integration import FlaskAppSource
from fervis.project.importing import import_object, project_import_context


def test_flask_executor_calls_configured_get_endpoint(tmp_path: Path) -> None:
    _write_flask_app(tmp_path)

    result = execute_get_endpoint(
        endpoint_name="order_count",
        sources=_sources(),
        project_root=tmp_path,
    )

    assert result.endpoint_name == "order_count"
    assert result.request_url == "/api/orders/count/"
    assert result.query_params == {}
    assert result.response_status == 200
    assert result.response_body == {"count": 42}


def test_flask_executor_calls_configured_get_endpoint_with_path_params(
    tmp_path: Path,
) -> None:
    _write_flask_app(tmp_path)

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


def test_flask_executor_rejects_undeclared_query_params(tmp_path: Path) -> None:
    _write_flask_app(tmp_path)

    with pytest.raises(EndpointExecutionError, match="Unknown query params"):
        execute_get_endpoint(
            endpoint_name="order_count",
            sources=_sources(),
            project_root=tmp_path,
            query_params={"status": "open"},
        )


def test_flask_executor_runs_wsgi_request_hooks_with_principal(tmp_path: Path) -> None:
    _write_flask_app(tmp_path)

    result = execute_get_endpoint(
        endpoint_name="current_user",
        sources=_sources(),
        project_root=tmp_path,
        principal_override=FlaskPrincipalOverride(
            source="flask_g",
            resolver="service.users:get_user_by_id",
            key="user_1",
        ),
    )

    assert result.response_status == 200
    assert result.response_body == {"id": "user_1", "role": "manager"}


def test_flask_executor_does_not_mutate_app_request_hooks(tmp_path: Path) -> None:
    _write_flask_app(tmp_path)

    execute_get_endpoint(
        endpoint_name="current_user",
        sources=_sources(),
        project_root=tmp_path,
        principal_override=FlaskPrincipalOverride(
            source="flask_g",
            resolver="service.users:get_user_by_id",
            key="user_1",
        ),
    )

    with project_import_context(tmp_path):
        app = import_object("service.main:app")
    assert len(app.before_request_funcs[None]) == 1


def test_flask_executor_returns_non_json_body_without_crashing(tmp_path: Path) -> None:
    _write_flask_app(tmp_path)

    result = execute_get_endpoint(
        endpoint_name="plain_text_status",
        sources=_sources(),
        project_root=tmp_path,
    )

    assert result.response_status == 200
    assert result.response_body == {
        "contentType": "text/plain; charset=utf-8",
        "text": "ready",
    }


def _sources() -> tuple[FlaskAppSource, ...]:
    return (
        FlaskAppSource(
            name="commerce",
            app="service.main:app",
            path_prefixes=["/api/"],
        ),
    )


def _write_flask_app(root: Path) -> None:
    package = root / "service"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from flask import Flask, g, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/count/')\n"
        "def order_count():\n"
        "    return jsonify({'count': 42})\n\n"
        "@app.get('/api/orders/<order_id>/')\n"
        "def get_order(order_id):\n"
        "    return jsonify({'id': order_id})\n\n"
        "@app.before_request\n"
        "def require_current_user_if_present():\n"
        "    if hasattr(g, 'current_user'):\n"
        "        g.user = g.current_user\n\n"
        "@app.get('/api/current-user/')\n"
        "def current_user():\n"
        "    return jsonify({'id': g.user.id, 'role': g.user.role})\n\n"
        "@app.get('/api/plain-text-status/')\n"
        "def plain_text_status():\n"
        "    return 'ready', 200, {'content-type': 'text/plain; charset=utf-8'}\n",
        encoding="utf-8",
    )
    (package / "users.py").write_text(
        "from types import SimpleNamespace\n\n"
        "def get_user_by_id(key, tenant_id=None):\n"
        "    return SimpleNamespace(id=key, role='manager')\n",
        encoding="utf-8",
    )
