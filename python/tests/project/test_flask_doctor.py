from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from fervis.interfaces.cli.dispatch import run_doctor_command
from fervis.project import discover_project


def test_fervis_doctor_reports_route_only_flask_endpoints_not_ready(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root)
    (root / "app.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n"
        "from fervis import configured_fervis\n"
        "configured_fervis().init_app(app)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return []\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    envelope = json.loads(stdout.getvalue())
    checks = _checks(envelope)
    assert exit_code == 2
    assert checks["source.catalog"]["status"] == "failed"
    assert "no lookup-readable GET endpoints" in checks["source.catalog"]["message"]


def test_fervis_doctor_accepts_openapi_backed_flask_endpoint(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root)
    (root / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n"
        "from fervis import configured_fervis\n"
        "configured_fervis().init_app(app)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([])\n\n"
        "@app.get('/openapi.json')\n"
        "def openapi():\n"
        "    return jsonify({'paths': {'/api/orders/': {'get': {\n"
        "        'operationId': 'list_orders',\n"
        "        'responses': {'200': {'content': {'application/json': {\n"
        "            'schema': {'type': 'array', 'items': {'type': 'object', "
        "'properties': {'id': {'type': 'string'}}}}\n"
        "        }}}}\n"
        "    }}}})\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    checks = _checks(json.loads(stdout.getvalue()))
    assert checks["source.catalog"]["status"] == "passed"
    assert checks["source.response_schema"]["status"] == "passed"
    assert checks["source.response_conformance"]["status"] == "passed"


def test_fervis_doctor_rejects_flask_endpoint_when_schema_cardinality_is_wrong(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root)
    (root / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n"
        "from fervis import configured_fervis\n"
        "configured_fervis().init_app(app)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([{'id': 'order_1'}])\n\n"
        "@app.get('/openapi.json')\n"
        "def openapi():\n"
        "    return jsonify({'paths': {'/api/orders/': {'get': {\n"
        "        'operationId': 'list_orders',\n"
        "        'responses': {'200': {'content': {'application/json': {\n"
        "            'schema': {'type': 'object', 'properties': "
        "{'id': {'type': 'string'}}}\n"
        "        }}}}\n"
        "    }}}})\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    checks = _checks(json.loads(stdout.getvalue()))
    assert exit_code == 2
    assert checks["source.catalog"]["status"] == "passed"
    assert checks["source.response_schema"]["status"] == "passed"
    assert checks["source.response_conformance"]["status"] == "failed"
    assert (
        "GET /api/orders/ is declared as one object, but returned a JSON array."
        in checks["source.response_conformance"]["message"]
    )
    fix = checks["source.response_conformance"]["fix"]
    assert isinstance(fix, dict)
    assert fix["kind"] == "fix_schema_cardinality"


def test_fervis_doctor_skips_shape_probe_for_flask_detail_route(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root)
    (root / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n"
        "from fervis import configured_fervis\n"
        "configured_fervis().init_app(app)\n\n"
        "@app.get('/api/orders/<order_id>')\n"
        "def get_order(order_id):\n"
        "    return jsonify({'id': order_id})\n\n"
        "@app.get('/openapi.json')\n"
        "def openapi():\n"
        "    return jsonify({'paths': {'/api/orders/{order_id}': {'get': {\n"
        "        'operationId': 'get_order',\n"
        "        'parameters': [{'name': 'order_id', 'in': 'path', "
        "'required': True, 'schema': {'type': 'string'}}],\n"
        "        'responses': {'200': {'content': {'application/json': {\n"
        "            'schema': {'type': 'object', 'properties': "
        "{'id': {'type': 'string'}}}\n"
        "        }}}}\n"
        "    }}}})\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    checks = _checks(json.loads(stdout.getvalue()))
    assert checks["source.catalog"]["status"] == "passed"
    assert checks["source.response_schema"]["status"] == "passed"
    assert checks["source.response_conformance"]["status"] == "skipped"
    assert (
        "No response shape probes could run"
        in checks["source.response_conformance"]["message"]
    )


def test_fervis_doctor_marks_shape_probe_skipped_when_flask_auth_blocks_all_probes(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root)
    (root / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n"
        "from fervis import configured_fervis\n"
        "configured_fervis().init_app(app)\n\n"
        "@app.before_request\n"
        "def require_auth():\n"
        "    from flask import request\n"
        "    if request.path == '/openapi.json':\n"
        "        return None\n"
        "    return ('Unauthorized', 401)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([{'id': 'order_1'}])\n\n"
        "@app.get('/openapi.json')\n"
        "def openapi():\n"
        "    return jsonify({'paths': {'/api/orders/': {'get': {\n"
        "        'operationId': 'list_orders',\n"
        "        'responses': {'200': {'content': {'application/json': {\n"
        "            'schema': {'type': 'array', 'items': {'type': 'object', "
        "'properties': {'id': {'type': 'string'}}}}\n"
        "        }}}}\n"
        "    }}}})\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    checks = _checks(json.loads(stdout.getvalue()))
    assert checks["source.response_conformance"]["status"] == "skipped"
    assert (
        "requires configured host read credentials"
        in checks["source.response_conformance"]["message"]
    )
    fix = checks["source.response_conformance"]["fix"]
    assert isinstance(fix, dict)
    assert fix["kind"] == "command"
    assert "fervis auth configure" in str(fix["command"])


def test_fervis_doctor_certifies_flask_factory_mount_from_runtime_app(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root, app_target="app:create_app")
    (root / "app.py").write_text(
        "from flask import Flask, jsonify\n"
        "from fervis import configured_fervis\n\n"
        "def create_app():\n"
        "    app = Flask(__name__)\n\n"
        "    @app.get('/api/orders/')\n"
        "    def list_orders():\n"
        "        return jsonify([])\n\n"
        "    @app.get('/openapi.json')\n"
        "    def openapi():\n"
        "        return jsonify({'paths': {'/api/orders/': {'get': {\n"
        "            'operationId': 'list_orders',\n"
        "            'responses': {'200': {'content': {'application/json': {\n"
        "                'schema': {'type': 'array', 'items': {'type': 'object', "
        "'properties': {'id': {'type': 'string'}}}}\n"
        "            }}}}\n"
        "        }}}})\n\n"
        "    configured_fervis().init_app(app)\n"
        "    return app\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    checks = _checks(json.loads(stdout.getvalue()))
    assert checks["framework.flask.mount"]["status"] == "passed"


def test_fervis_doctor_fails_flask_source_when_any_configured_route_lacks_contract(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(root)
    (root / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n"
        "from fervis import configured_fervis\n"
        "configured_fervis().init_app(app)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([])\n\n"
        "@app.get('/api/products/')\n"
        "def list_products():\n"
        "    return jsonify([])\n\n"
        "@app.get('/openapi.json')\n"
        "def openapi():\n"
        "    return jsonify({'paths': {'/api/orders/': {'get': {\n"
        "        'operationId': 'list_orders',\n"
        "        'responses': {'200': {'content': {'application/json': {\n"
        "            'schema': {'type': 'array', 'items': {'type': 'object', "
        "'properties': {'id': {'type': 'string'}}}}\n"
        "        }}}}\n"
        "    }}}})\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_doctor_command(
        ("doctor",),
        project=discover_project(root),
        stdout=stdout,
    )

    checks = _checks(json.loads(stdout.getvalue()))
    assert exit_code == 2
    assert checks["source.catalog"]["status"] == "passed"
    assert checks["source.response_schema"]["status"] == "failed"
    assert (
        "1 exposed endpoint has no response fields"
        in checks["source.response_schema"]["message"]
    )


def _flask_project(tmp_path: Path) -> Path:
    root = tmp_path / "flask_api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'flask-api'\ndependencies = ['flask>=3.0']\n",
        encoding="utf-8",
    )
    return root


def _write_config(root: Path, *, app_target: str = "app:app") -> None:
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "fervis.json").write_text(
        json.dumps(
            {
                "schema_version": "v0.1",
                "framework": "flask",
                "default_environment": "local",
                "host": {
                    "organization_name": "Acme",
                    "about_api": "Acme operations API.",
                    "timezone": "UTC",
                },
                "routes": {"prefix": "/fervis/"},
                "models": {
                    "providers": [
                        {
                            "name": "openai",
                            "allowed_model_keys": ["gpt-5.4-mini"],
                        }
                    ]
                },
                "sources": [
                    {
                        "kind": "flask_app",
                        "name": "commerce",
                        "app": app_target,
                        "app_args": [],
                        "app_kwargs": {},
                        "path_prefixes": ["/api/"],
                        "blueprints": [],
                    }
                ],
                "environments": {
                    "local": {
                        "models": {
                            "default": {
                                "provider": "openai",
                                "model_key": "gpt-5.4-mini",
                            }
                        },
                        "persistence": {
                            "kind": "sqlite",
                            "path": ".fervis/fervis.sqlite3",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _checks(envelope: dict[str, object]) -> dict[str, dict[str, object]]:
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    checks = payload["checks"]
    assert isinstance(checks, list)
    return {str(check["id"]): check for check in checks if isinstance(check, dict)}
