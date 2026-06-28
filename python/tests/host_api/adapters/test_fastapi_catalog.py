from __future__ import annotations

import sys
import types
from pathlib import Path

from fervis.host_api.adapters.fastapi.catalog import (
    get_fastapi_endpoint_contracts,
)
from fervis.project.integration import FastAPIAppSource


def test_fastapi_catalog_uses_explicit_source_imports_and_path_prefixes(
    tmp_path: Path,
) -> None:
    module = types.ModuleType("commerce_api")
    module.app = _FakeFastAPIApp()
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="commerce",
                    import_paths=["commerce_api:app"],
                    path_prefixes=["/api/v1/"],
                ),
            ),
            project_root=tmp_path,
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert [contract.path_template for contract in contracts] == ["/api/v1/orders/"]
    contract = contracts[0]
    assert contract.endpoint_name == "list_orders"
    assert [param.name for param in contract.query_params] == ["status"]
    assert {field.path for field in contract.response_fields} == {
        "id",
        "status",
        "amount",
    }
    assert contract.catalog_endpoint is not None
    assert contract.catalog_endpoint.framework_kind == "fastapi"
    assert contract.catalog_endpoint.source_namespace_kind == "fastapi_app"
    assert contract.catalog_endpoint.source_namespace_path == ("commerce",)


def test_fastapi_catalog_imports_from_declared_project_source_roots(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "src" / "service"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        """
class App:
    def openapi(self):
        return {
            "paths": {
                "/api/v1/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"id": {"type": "string"}},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
        }

app = App()
""",
        encoding="utf-8",
    )

    contracts = get_fastapi_endpoint_contracts(
        sources=(
            FastAPIAppSource(
                name="commerce",
                import_paths=["service.main:app"],
                path_prefixes=["/api/v1/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_fastapi_catalog_loads_configured_factory_app(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "app"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        """
class App:
    def openapi(self):
        return {
            "paths": {
                "/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"id": {"type": "string"}},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
        }

def create_app():
    return App()
""",
        encoding="utf-8",
    )

    contracts = get_fastapi_endpoint_contracts(
        sources=(
            FastAPIAppSource(
                name="commerce",
                import_paths=["app.main:create_app"],
                path_prefixes=["/orders/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_fastapi_catalog_suppresses_host_stdout(tmp_path: Path, capsys) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'api'\ndependencies = ['fastapi']\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "app"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "main.py").write_text(
        """
print("host import noise")

class App:
    def openapi(self):
        print("host schema noise")
        return {
            "paths": {
                "/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"id": {"type": "string"}},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
        }

app = App()
""",
        encoding="utf-8",
    )

    contracts = get_fastapi_endpoint_contracts(
        sources=(
            FastAPIAppSource(
                name="commerce",
                import_paths=["app.main:app"],
                path_prefixes=["/orders/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]
    assert capsys.readouterr().out == ""


def test_fastapi_catalog_path_prefixes_match_path_segments() -> None:
    module = types.ModuleType("segment_api")
    module.app = _SegmentPrefixApp()
    sys.modules[module.__name__] = module
    try:
        contracts = get_fastapi_endpoint_contracts(
            sources=(
                FastAPIAppSource(
                    name="commerce",
                    import_paths=["segment_api:app"],
                    path_prefixes=["/api"],
                ),
            ),
            project_root=Path.cwd(),
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert [contract.path_template for contract in contracts] == ["/api/orders/"]


class _FakeFastAPIApp:
    def openapi(self) -> dict[str, object]:
        return {
            "paths": {
                "/api/v1/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "tags": ["orders"],
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
                                                    "amount": {"type": "number"},
                                                },
                                                "required": ["id"],
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


class _SegmentPrefixApp:
    def openapi(self) -> dict[str, object]:
        operation = {
            "operationId": "list_orders",
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"id": {"type": "string"}},
                            }
                        }
                    }
                }
            },
        }
        return {
            "paths": {
                "/api/orders/": {"get": operation},
                "/apiary/orders/": {"get": {**operation, "operationId": "apiary"}},
            }
        }
