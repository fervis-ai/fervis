from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from fervis.interfaces.cli.dispatch import run_fervis
from fervis.project import discover_project

from ._support import _ports


def test_fervis_catalog_returns_configured_fastapi_endpoint_contracts(
    tmp_path: Path,
) -> None:
    root = _fastapi_project(tmp_path)
    _write_config(
        root,
        {
            "schema_version": "v0.1",
            "framework": "fastapi",
            "default_environment": "local",
            "host": {"organization_name": "Acme", "about_api": "Acme operations API.", "timezone": "UTC"},
            "routes": {"prefix": "/fervis/"},
            "models": {
                "providers": [
                    {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
                ],
            },
            "sources": [
                {
                    "kind": "fastapi_app",
                    "name": "commerce",
                    "import_paths": ["app.main:create_app"],
                    "path_prefixes": ["/api/"],
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
        },
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("catalog",),
        ports=_ports(project=discover_project(root)),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert envelope["schema"] == "fervis-command-result.v0.1"
    assert envelope["command"] == "catalog"
    assert envelope["status"] == "succeeded"
    assert envelope["payload_schema"] == "fervis-catalog-result.v0.1"
    assert envelope["next_actions"] == []
    assert envelope["payload"] == {
        "schema_version": "v0.1",
        "status": "passed",
        "source_count": 1,
        "endpoint_count": 1,
        "sources": [
            {
                "name": "commerce",
                "kind": "fastapi_app",
                "configured": {
                    "import_paths": ["app.main:create_app"],
                    "path_prefixes": ["/api/"],
                },
                "endpoint_count": 1,
                "endpoints": [
                    {
                        "name": "list_orders",
                        "method": "GET",
                        "path": "/api/orders/",
                        "query_params": [
                            {
                                "name": "status",
                                "required": False,
                                "source": "query",
                                "type": "string",
                            }
                        ],
                        "path_params": [],
                        "response_fields": [
                            {"path": "id", "type": "string"},
                            {"path": "status", "type": "string"},
                            {"path": "total", "type": "decimal"},
                        ],
                        "quality": "schema_backed",
                        "eligible": True,
                        "blocked_reason": None,
                        "next_actions": [],
                        "capabilities": {
                            "read": True,
                            "filter": True,
                            "aggregate_candidate": True,
                        },
                    }
                ],
            }
        ],
        "blocked_sources": [],
    }


def test_fervis_catalog_returns_configured_flask_route_only_endpoints(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(
        root,
        {
            "schema_version": "v0.1",
            "framework": "flask",
            "default_environment": "local",
            "host": {"organization_name": "Acme", "about_api": "Acme operations API.", "timezone": "UTC"},
            "routes": {"prefix": "/fervis/"},
            "models": {
                "providers": [
                    {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
                ],
            },
            "sources": [
                {
                    "kind": "flask_app",
                    "name": "commerce",
                    "app": "app:app",
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
        },
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("catalog",),
        ports=_ports(project=discover_project(root)),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["status"] == "blocked"
    assert envelope["payload"]["status"] == "blocked"
    assert envelope["next_actions"] == [
        {
            "kind": "add_schema_metadata",
            "endpoint": "list_orders",
            "description": (
                "Expose this endpoint's response/query contract through a "
                "supported Flask surface: OpenAPI/Swagger, Marshmallow metadata, "
                "JSON:API resource/schema metadata, or Flask-AppBuilder metadata. "
                "For plain Flask routes, follow "
                "github.com/fervis-ai/fervis/python/flask/AGENTS.md."
            ),
        }
    ]
    assert envelope["payload"]["sources"] == [
        {
            "name": "commerce",
            "kind": "flask_app",
            "configured": {
                "app": "app:app",
                "app_args": [],
                "app_kwargs": {},
                "path_prefixes": ["/api/"],
                "blueprints": [],
            },
            "endpoint_count": 1,
            "endpoints": [
                {
                    "name": "list_orders",
                    "method": "GET",
                    "path": "/api/orders/",
                    "query_params": [],
                    "path_params": [],
                    "response_fields": [],
                    "quality": "route_only",
                    "eligible": False,
                    "blocked_reason": "response_schema_missing",
                    "next_actions": [
                        {
                            "kind": "add_schema_metadata",
                            "endpoint": "list_orders",
                            "description": (
                                "Expose this endpoint's response/query contract "
                                "through a supported Flask surface: "
                                "OpenAPI/Swagger, Marshmallow metadata, "
                                "JSON:API resource/schema metadata, or "
                                "Flask-AppBuilder metadata. For plain "
                                "Flask routes, follow "
                                "github.com/fervis-ai/fervis/python/flask/AGENTS.md."
                            ),
                        }
                    ],
                    "capabilities": {
                        "read": False,
                        "filter": False,
                        "aggregate_candidate": False,
                    },
                }
            ],
        }
    ]


def test_fervis_catalog_blocks_without_config(tmp_path: Path) -> None:
    root = _fastapi_project(tmp_path)
    stdout = StringIO()

    exit_code = run_fervis(
        ("catalog",),
        ports=_ports(project=discover_project(root)),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["command"] == "catalog"
    assert envelope["status"] == "blocked"
    assert envelope["payload_schema"] == "fervis-command-error.v0.1"
    assert envelope["payload"]["error"]["code"] == "config_missing"


def test_fervis_catalog_reports_missing_flask_dependency_action(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(
        root,
        {
            "schema_version": "v0.1",
            "framework": "flask",
            "default_environment": "local",
            "host": {"organization_name": "Acme", "about_api": "Acme operations API.", "timezone": "UTC"},
            "routes": {"prefix": "/fervis/"},
            "models": {
                "providers": [
                    {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
                ],
            },
            "sources": [
                {
                    "kind": "flask_app",
                    "name": "commerce",
                    "app": "app:app",
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
        },
    )
    (root / "app.py").write_text(
        "import definitely_missing_flask_dep\n\n"
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("catalog",),
        ports=_ports(project=discover_project(root)),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["next_actions"] == [
        {
            "command": "uv sync",
            "description": (
                "Install the host project dependencies so Python can import "
                "'definitely_missing_flask_dep' while building the Fervis source "
                "catalog."
            ),
            "kind": "install_dependencies",
            "module": "definitely_missing_flask_dep",
        }
    ]


def test_fervis_catalog_reports_missing_flask_submodule_dependency_action(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(
        root,
        {
            "schema_version": "v0.1",
            "framework": "flask",
            "default_environment": "local",
            "host": {"organization_name": "Acme", "about_api": "Acme operations API.", "timezone": "UTC"},
            "routes": {"prefix": "/fervis/"},
            "models": {
                "providers": [
                    {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
                ],
            },
            "sources": [
                {
                    "kind": "flask_app",
                    "name": "commerce",
                    "app": "app:app",
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
        },
    )
    (root / "app.py").write_text(
        "import werkzeug.contrib\n\n"
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("catalog",),
        ports=_ports(project=discover_project(root)),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["next_actions"] == [
        {
            "command": "uv sync",
            "description": (
                "Install the host project dependencies so Python can import "
                "'werkzeug.contrib' while building the Fervis source catalog."
            ),
            "kind": "install_dependencies",
            "module": "werkzeug.contrib",
        }
    ]


def test_fervis_catalog_reports_incompatible_flask_dependency_action(
    tmp_path: Path,
) -> None:
    root = _flask_project(tmp_path)
    _write_config(
        root,
        {
            "schema_version": "v0.1",
            "framework": "flask",
            "default_environment": "local",
            "host": {"organization_name": "Acme", "about_api": "Acme operations API.", "timezone": "UTC"},
            "routes": {"prefix": "/fervis/"},
            "models": {
                "providers": [
                    {"name": "openai", "allowed_model_keys": ["gpt-5.4-mini"]}
                ],
            },
            "sources": [
                {
                    "kind": "flask_app",
                    "name": "commerce",
                    "app": "app:app",
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
        },
    )
    (root / "app.py").write_text(
        "from flask_sqlalchemy import Model\n\n"
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n",
        encoding="utf-8",
    )
    stdout = StringIO()

    exit_code = run_fervis(
        ("catalog",),
        ports=_ports(project=discover_project(root)),
        stdout=stdout,
        stderr=StringIO(),
    )

    envelope = json.loads(stdout.getvalue())
    assert exit_code == 2
    assert envelope["next_actions"] == [
        {
            "command": "uv sync",
            "description": (
                "Install the host project dependencies so Python can import "
                "'flask_sqlalchemy' while building the Fervis source catalog."
            ),
            "kind": "install_dependencies",
            "module": "flask_sqlalchemy",
        }
    ]


def _fastapi_project(tmp_path: Path) -> Path:
    root = tmp_path / "api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    app_dir = root / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text(
        """
class App:
    def openapi(self):
        return {
            "paths": {
                "/api/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "parameters": [
                            {
                                "name": "status",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "string"},
                                                    "status": {"type": "string"},
                                                    "total": {"type": "number"},
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
                "/internal/orders/": {
                    "get": {
                        "operationId": "internal_orders",
                        "responses": {"200": {"description": "ok"}},
                    }
                },
            }
        }

def create_app():
    return App()
""",
        encoding="utf-8",
    )
    return root


def _flask_project(tmp_path: Path) -> Path:
    root = tmp_path / "flask_api"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'flask-api'\ndependencies = ['flask']\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return []\n",
        encoding="utf-8",
    )
    return root


def _write_config(root: Path, schema: dict[str, object]) -> None:
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "fervis.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
