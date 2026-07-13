from __future__ import annotations

from pathlib import Path

from fervis.host_api.adapters.flask.catalog import (
    get_flask_endpoint_contracts,
)
from fervis.project.integration import FlaskAppSource


def test_flask_catalog_uses_runtime_url_map_and_configured_prefixes(
    tmp_path: Path,
) -> None:
    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return []\n\n"
        "@app.get('/internal/orders/')\n"
        "def internal_orders():\n"
        "    return []\n\n"
        "@app.post('/api/orders/')\n"
        "def create_order():\n"
        "    return {}\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app.main:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.path_template for contract in contracts] == ["/api/orders/"]
    contract = contracts[0]
    assert contract.endpoint_name == "list_orders"
    assert contract.method == "GET"
    assert contract.response_fields == ()
    assert contract.query_schema_source == "route"
    assert contract.response_schema_source == "missing"
    assert contract.catalog_endpoint is not None
    assert contract.catalog_endpoint.framework_kind == "flask"
    assert contract.catalog_endpoint.source_namespace_kind == "flask_blueprint"
    assert contract.catalog_endpoint.source_namespace_path == ("commerce",)


def test_flask_catalog_ignores_framework_documentation_routes_at_source_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "def docs():\n"
        "    return {}\n\n"
        "def api_root():\n"
        "    return {}\n\n"
        "@app.get('/api/v1/orders/')\n"
        "def list_orders():\n"
        "    return []\n\n"
        "app.add_url_rule('/api/v1/', endpoint='api.doc', view_func=docs)\n"
        "app.add_url_rule('/api/v1/', endpoint='api.root', view_func=api_root)\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/v1/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_flask_catalog_applies_blueprint_allowlist(tmp_path: Path) -> None:
    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from flask import Blueprint, Flask\n\n"
        "orders = Blueprint('orders', __name__)\n"
        "inventory = Blueprint('inventory', __name__)\n\n"
        "@orders.get('/orders/')\n"
        "def list_orders():\n"
        "    return []\n\n"
        "@inventory.get('/inventory/')\n"
        "def list_inventory():\n"
        "    return []\n\n"
        "app = Flask(__name__)\n"
        "app.register_blueprint(orders, url_prefix='/api')\n"
        "app.register_blueprint(inventory, url_prefix='/api')\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app.main:app",
                path_prefixes=["/api/"],
                blueprints=["orders"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["orders.list_orders"]
    assert contracts[0].catalog_endpoint is not None
    assert contracts[0].catalog_endpoint.source_namespace_path == (
        "commerce",
        "orders",
    )


def test_flask_catalog_extracts_converter_backed_path_param_types(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/people/<int:person_id>')\n"
        "def read_person(person_id):\n"
        "    return {}\n\n"
        "@app.get('/api/prices/<float:amount>')\n"
        "def read_price(amount):\n"
        "    return {}\n\n"
        "@app.get('/api/files/<path:file_path>')\n"
        "def read_file(file_path):\n"
        "    return {}\n\n"
        "@app.get('/api/tokens/<uuid:token_id>')\n"
        "def read_token(token_id):\n"
        "    return {}\n\n"
        "@app.get('/api/slugs/<slug>')\n"
        "def read_slug(slug):\n"
        "    return {}\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    params_by_path = {
        contract.path_template: {
            param.name: param.type for param in contract.path_params
        }
        for contract in contracts
    }
    assert params_by_path == {
        "/api/files/{file_path}": {"file_path": "path"},
        "/api/people/{person_id}": {"person_id": "integer"},
        "/api/prices/{amount}": {"amount": "decimal"},
        "/api/slugs/{slug}": {"slug": "string"},
        "/api/tokens/{token_id}": {"token_id": "uuid"},
    }


def test_plain_flask_json_route_without_contract_does_not_become_lookup_read(
    tmp_path: Path,
) -> None:
    from fervis.lookup.relation_catalog.from_host_api import (
        relation_catalog_from_endpoint_contracts,
    )

    (tmp_path / "app.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return [{'id': 'ord_1'}]\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    catalog = relation_catalog_from_endpoint_contracts(contracts)

    assert contracts[0].response_schema_source == "missing"
    assert contracts[0].response_fields == ()
    assert catalog.reads == ()


def test_empty_plain_flask_json_route_remains_route_only(
    tmp_path: Path,
) -> None:
    from fervis.lookup.relation_catalog.from_host_api import (
        relation_catalog_from_endpoint_contracts,
    )

    (tmp_path / "app.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return []\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )
    catalog = relation_catalog_from_endpoint_contracts(contracts)

    assert contracts[0].response_schema_source == "missing"
    assert catalog.reads == ()


def test_flask_catalog_enriches_routes_from_openapi_document(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify({'orders': [{'id': 'ord_1', 'total': '10.00'}], 'total': 1})\n\n"
        "@app.get('/api/swagger.json')\n"
        "def openapi():\n"
        "    return jsonify({\n"
        "        'openapi': '3.1.0',\n"
        "        'paths': {\n"
        "            '/api/orders/': {\n"
        "                'get': {\n"
        "                    'operationId': 'list_orders',\n"
        "                    'x-fervis': {\n"
        "                        'pagination': {\n"
        "                            'kind': 'page_number',\n"
        "                            'positionQueryParam': 'page',\n"
        "                            'pageSizeQueryParam': 'per_page',\n"
        "                            'resultsPath': 'orders',\n"
        "                            'pageSize': 50,\n"
        "                            'maxPageSize': 100,\n"
        "                            'totalPath': 'total'\n"
        "                        },\n"
        "                        'candidateKeys': [{\n"
        "                            'keyId': 'primary_key',\n"
        "                            'entityKind': 'order',\n"
        "                            'components': [{\n"
        "                                'componentId': 'order_id',\n"
        "                                'fieldPath': 'orders.id'\n"
        "                            }],\n"
        "                            'primary': True\n"
        "                        }],\n"
        "                        'entityReferences': [{\n"
        "                            'referenceId': 'store_reference',\n"
        "                            'targetEntityKind': 'store',\n"
        "                            'targetKeyId': 'primary_key',\n"
        "                            'components': [{\n"
        "                                'targetComponentId': 'store_id',\n"
        "                                'localFieldPath': 'orders.store_id'\n"
        "                            }]\n"
        "                        }]\n"
        "                    },\n"
        "                    'parameters': [\n"
        "                        {'name': 'status', 'in': 'query', "
        "'schema': {'type': 'string'}},\n"
        "                        {'name': 'page', 'in': 'query', "
        "'schema': {'type': 'integer'}},\n"
        "                        {'name': 'per_page', 'in': 'query', "
        "'schema': {'type': 'integer'}}\n"
        "                    ],\n"
        "                    'responses': {'200': {'content': {\n"
        "                        'application/json': {'schema': {\n"
        "                            'type': 'object',\n"
        "                            'properties': {\n"
        "                                'orders': {'type': 'array', 'items': {\n"
        "                                    'type': 'object', 'properties': {\n"
        "                                        'id': {'type': 'string'},\n"
        "                                        'store_id': {'type': 'string'},\n"
        "                                        'total': {'type': 'number'}\n"
        "                                    }\n"
        "                                }},\n"
        "                                'total': {'type': 'integer'}\n"
        "                            }\n"
        "                        }}\n"
        "                    }}}\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    })\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.endpoint_name == "list_orders"
    assert contract.query_schema_source == "openapi"
    assert contract.response_schema_source == "openapi"
    assert [param.name for param in contract.query_params] == [
        "status",
        "page",
        "per_page",
    ]
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("orders", "array"),
        ("orders.id", "string"),
        ("orders.store_id", "string"),
        ("orders.total", "decimal"),
        ("total", "integer"),
    ]
    assert contract.pagination is not None
    assert contract.pagination.results_path == "orders"
    assert contract.candidate_keys[0].entity_kind == "order"
    assert contract.entity_references[0].target_entity_kind == "store"


def test_flask_catalog_ignores_documentation_routes_under_source_prefix(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/ui/')\n"
        "def swagger_ui_index():\n"
        "    return '<html>docs</html>'\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([])\n\n"
        "@app.get('/api/swagger.json')\n"
        "def swagger_json():\n"
        "    return jsonify({'paths': {'/api/orders/': {'get': {\n"
        "        'operationId': 'list_orders',\n"
        "        'responses': {'200': {'content': {'application/json': {\n"
        "            'schema': {'type': 'array', 'items': {'type': 'object', "
        "'properties': {'id': {'type': 'string'}}}}\n"
        "        }}}}\n"
        "    }}}})\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_flask_catalog_accepts_wrappers_that_expose_flask_app(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "class ConnexionLikeApp:\n"
        "    def __init__(self):\n"
        "        self.app = Flask(__name__)\n\n"
        "        @self.app.get('/api/orders/')\n"
        "        def list_orders():\n"
        "            return jsonify([{'id': 'ord_1'}])\n\n"
        "        @self.app.get('/api/swagger.json')\n"
        "        def openapi():\n"
        "            return jsonify({\n"
        "                'openapi': '3.1.0',\n"
        "                'paths': {'/api/orders/': {'get': {\n"
        "                    'operationId': 'list_orders',\n"
        "                    'responses': {'200': {'content': {\n"
        "                        'application/json': {'schema': {\n"
        "                            'type': 'array',\n"
        "                            'items': {'type': 'object', 'properties': {\n"
        "                                'id': {'type': 'string'}\n"
        "                            }}\n"
        "                        }}\n"
        "                    }}}\n"
        "                }}}\n"
        "            })\n\n"
        "    def __call__(self, environ, start_response):\n"
        "        return self.app(environ, start_response)\n\n"
        "connex_app = ConnexionLikeApp()\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:connex_app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_flask_catalog_enriches_routes_from_swagger_2_document(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/people')\n"
        "def list_people():\n"
        "    return jsonify([{'person_id': 1, 'fname': 'Ada'}])\n\n"
        "@app.get('/api/swagger.json')\n"
        "def swagger():\n"
        "    return jsonify({\n"
        "        'swagger': '2.0',\n"
        "        'basePath': '/api',\n"
        "        'paths': {'/people': {'get': {\n"
        "            'operationId': 'people.read_all',\n"
        "            'parameters': [\n"
        "                {'name': 'limit', 'in': 'query', 'type': 'integer'}\n"
        "            ],\n"
        "            'responses': {'200': {'schema': {\n"
        "                'type': 'array',\n"
        "                'items': {'type': 'object', 'properties': {\n"
        "                    'person_id': {'type': 'integer'},\n"
        "                    'fname': {'type': 'string'}\n"
        "                }}\n"
        "            }}}\n"
        "        }}}\n"
        "    })\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="people",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.endpoint_name == "people.read_all"
    assert contract.path_template == "/api/people"
    assert [(param.name, param.type) for param in contract.query_params] == [
        ("limit", "integer")
    ]
    assert {field.path: field.type for field in contract.response_fields} == {
        "person_id": "integer",
        "fname": "string",
    }
    assert contract.response_cardinality == "many"


def test_openapi_operation_id_wins_over_flask_internal_endpoint_name(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def generated_internal_endpoint_name():\n"
        "    return jsonify([{'id': 'ord_1'}])\n\n"
        "@app.get('/api/swagger.json')\n"
        "def swagger():\n"
        "    return jsonify({\n"
        "        'swagger': '2.0',\n"
        "        'paths': {'/api/orders/': {'get': {\n"
        "            'operationId': 'orders.read_all',\n"
        "            'responses': {'200': {'schema': {\n"
        "                'type': 'array',\n"
        "                'items': {'type': 'object', 'properties': {\n"
        "                    'id': {'type': 'string'}\n"
        "                }}\n"
        "            }}}\n"
        "        }}}\n"
        "    })\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="orders",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    assert contracts[0].endpoint_name == "orders.read_all"
    assert contracts[0].url_name == "orders.read_all"
    assert contracts[0].catalog_endpoint is not None
    assert (
        contracts[0].catalog_endpoint.route_name == "generated_internal_endpoint_name"
    )
    assert contracts[0].catalog_endpoint.api_schema_operation_id == "orders.read_all"


def test_flask_catalog_enriches_routes_from_flask_apispec_marshmallow_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n"
        "\n"
        "class String:\n"
        "    required = False\n"
        "    metadata = {}\n"
        "    validators = ()\n\n"
        "class Integer(String):\n"
        "    pass\n\n"
        "class Decimal(String):\n"
        "    pass\n\n"
        "class Schema:\n"
        "    def __init__(self, many=False):\n"
        "        self.many = many\n"
        "        self.fields = {\n"
        "            name: value for name, value in self.__class__.__dict__.items()\n"
        "            if isinstance(value, String)\n"
        "        }\n\n"
        "class Annotation:\n"
        "    def __init__(self, *options):\n"
        "        self.options = list(options)\n\n"
        "def use_kwargs(fields):\n"
        "    def decorate(fn):\n"
        "        metadata = getattr(fn, '__apispec__', {})\n"
        "        metadata.setdefault('args', []).append(\n"
        "            Annotation({'args': fields, 'kwargs': {'location': 'query'}})\n"
        "        )\n"
        "        fn.__apispec__ = metadata\n"
        "        return fn\n"
        "    return decorate\n\n"
        "def marshal_with(schema, code=200):\n"
        "    def decorate(fn):\n"
        "        metadata = getattr(fn, '__apispec__', {})\n"
        "        metadata.setdefault('schemas', []).append(\n"
        "            Annotation({code: {'schema': schema}})\n"
        "        )\n"
        "        fn.__apispec__ = metadata\n"
        "        return fn\n"
        "    return decorate\n\n"
        "class OrderSchema(Schema):\n"
        "    id = String()\n"
        "    id.required = True\n"
        "    total = Decimal()\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "@use_kwargs({'status': String(), 'limit': Integer()})\n"
        "@marshal_with(OrderSchema(many=True), code=200)\n"
        "def list_orders(**kwargs):\n"
        "    return jsonify([{'id': 'ord_1', 'total': '10.00'}])\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.endpoint_name == "list_orders"
    assert contract.query_schema_source == "flask_apispec"
    assert contract.response_schema_source == "flask_apispec"
    assert [(param.name, param.type) for param in contract.query_params] == [
        ("status", "string"),
        ("limit", "integer"),
    ]
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("id", "string"),
        ("total", "decimal"),
    ]
    assert contract.response_cardinality == "many"


def test_flask_catalog_reads_flask_apispec_default_response_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n"
        "\n"
        "class String:\n"
        "    required = False\n"
        "    metadata = {}\n"
        "    validators = ()\n\n"
        "class Schema:\n"
        "    def __init__(self, many=False):\n"
        "        self.many = many\n"
        "        self.fields = {\n"
        "            name: value for name, value in self.__class__.__dict__.items()\n"
        "            if isinstance(value, String)\n"
        "        }\n\n"
        "class Annotation:\n"
        "    def __init__(self, *options):\n"
        "        self.options = list(options)\n\n"
        "def marshal_with(schema):\n"
        "    def decorate(fn):\n"
        "        metadata = getattr(fn, '__apispec__', {})\n"
        "        metadata.setdefault('schemas', []).append(\n"
        "            Annotation({'default': {'schema': schema}})\n"
        "        )\n"
        "        fn.__apispec__ = metadata\n"
        "        return fn\n"
        "    return decorate\n\n"
        "class TagSchema(Schema):\n"
        "    name = String()\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/tags/')\n"
        "@marshal_with(TagSchema(many=True))\n"
        "def list_tags():\n"
        "    return jsonify([{'name': 'priority'}])\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.response_schema_source == "flask_apispec"
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("name", "string"),
    ]
    assert contract.response_cardinality == "many"


def test_flask_catalog_enriches_jsonapi_resource_schema_metadata(
    tmp_path: Path,
) -> None:
    from fervis.lookup.relation_catalog.from_host_api import (
        relation_catalog_from_endpoint_contracts,
    )

    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n"
        "from flask.views import MethodView\n"
        "\n"
        "class String:\n"
        "    required = False\n"
        "    metadata = {}\n"
        "    validators = ()\n\n"
        "class Integer(String):\n"
        "    pass\n\n"
        "class Schema:\n"
        "    def __init__(self, many=False):\n"
        "        self.many = many\n"
        "        self.fields = {\n"
        "            name: value for name, value in self.__class__.__dict__.items()\n"
        "            if isinstance(value, String)\n"
        "        }\n\n"
        "class PersonSchema(Schema):\n"
        "    id = Integer()\n"
        "    id.required = True\n"
        "    name = String()\n\n"
        "class ResourceList:\n"
        "    pass\n\n"
        "class PersonList(ResourceList, MethodView):\n"
        "    schema = PersonSchema\n\n"
        "    def get(self):\n"
        "        return jsonify({'data': []})\n\n"
        "app = Flask(__name__)\n"
        "app.add_url_rule('/api/people/', view_func=PersonList.as_view('people'))\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="people",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.response_schema_source == "jsonapi"
    assert contract.response_cardinality == "many"
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("data", "array"),
        ("data.attributes.id", "integer"),
        ("data.attributes.name", "string"),
    ]
    read = relation_catalog_from_endpoint_contracts(contracts).read("people")
    assert read.response_envelope.results_path == "data"
    assert read.fields_by_path["data.attributes.id"].row_path_id == "data"


def test_flask_catalog_enriches_flask_appbuilder_openapi_method_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n"
        "from flask.views import MethodView\n\n"
        "class ChartApi(MethodView):\n"
        "    openapi_spec_methods = {\n"
        "        'get': {\n"
        "            'parameters': [\n"
        "                {'name': 'q', 'in': 'query', 'schema': {'type': 'string'}}\n"
        "            ],\n"
        "            'responses': {'200': {'content': {'application/json': {\n"
        "                'schema': {'type': 'object', 'properties': {\n"
        "                    'id': {'type': 'integer'},\n"
        "                    'slice_name': {'type': 'string'}\n"
        "                }}\n"
        "            }}}}\n"
        "        }\n"
        "    }\n\n"
        "    def get(self):\n"
        "        return jsonify({'id': 1, 'slice_name': 'Sales'})\n\n"
        "app = Flask(__name__)\n"
        "app.add_url_rule('/api/v1/chart/', view_func=ChartApi.as_view('chart_api'))\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="superset",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.query_schema_source == "flask_appbuilder"
    assert contract.response_schema_source == "flask_appbuilder"
    assert [(param.name, param.type) for param in contract.query_params] == [
        ("q", "string")
    ]
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("id", "integer"),
        ("slice_name", "string"),
    ]


def test_flask_catalog_enriches_flask_appbuilder_bound_method_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "class ChartApi:\n"
        "    openapi_spec_methods = {\n"
        "        'get_list': {\n"
        "            'responses': {'200': {'content': {'application/json': {\n"
        "                'schema': {'type': 'object', 'properties': {\n"
        "                    'result': {'type': 'array', 'items': {\n"
        "                        'type': 'object', 'properties': {\n"
        "                            'id': {'type': 'integer'},\n"
        "                            'slice_name': {'type': 'string'}\n"
        "                        }\n"
        "                    }}\n"
        "                }}\n"
        "            }}}}\n"
        "        }\n"
        "    }\n\n"
        "    def get_list(self):\n"
        "        return jsonify({'result': [{'id': 1, 'slice_name': 'Sales'}]})\n\n"
        "app = Flask(__name__)\n"
        "resource = ChartApi()\n"
        "app.add_url_rule(\n"
        "    '/api/v1/chart/',\n"
        "    endpoint='ChartRestApi.get_list',\n"
        "    view_func=resource.get_list,\n"
        ")\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="superset",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.response_schema_source == "flask_appbuilder"
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("result", "array"),
        ("result.id", "integer"),
        ("result.slice_name", "string"),
    ]


def test_openapi_array_response_becomes_many_root_row_population(
    tmp_path: Path,
) -> None:
    from fervis.lookup.relation_catalog import RowCardinality
    from fervis.lookup.relation_catalog.from_host_api import (
        relation_catalog_from_endpoint_contracts,
    )

    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([{'id': 'ord_1', 'total': '10.00'}])\n\n"
        "@app.get('/api/swagger.json')\n"
        "def openapi():\n"
        "    return jsonify({\n"
        "        'openapi': '3.1.0',\n"
        "        'paths': {'/api/orders/': {'get': {\n"
        "            'operationId': 'list_orders',\n"
        "            'responses': {'200': {'content': {\n"
        "                'application/json': {'schema': {\n"
        "                    'type': 'array',\n"
        "                    'items': {'type': 'object', 'properties': {\n"
        "                        'id': {'type': 'string'},\n"
        "                        'total': {'type': 'number'}\n"
        "                    }}\n"
        "                }}\n"
        "            }}}\n"
        "        }}}\n"
        "    })\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )
    catalog = relation_catalog_from_endpoint_contracts(contracts)

    read = catalog.read("list_orders")
    root = next(item for item in read.row_paths if item.id == "root")
    assert root.path == ""
    assert root.cardinality is RowCardinality.MANY
    assert {field.row_path_id for field in read.fields} == {"root"}


def test_flask_catalog_suppresses_host_stdout(tmp_path: Path, capsys) -> None:
    (tmp_path / "app.py").write_text(
        "from flask import Flask, jsonify\n\n"
        "print('host import noise')\n"
        "app = Flask(__name__)\n\n"
        "@app.get('/api/orders/')\n"
        "def list_orders():\n"
        "    return jsonify([{'id': 'ord_1'}])\n\n"
        "@app.get('/api/swagger.json')\n"
        "def openapi():\n"
        "    print('host schema noise')\n"
        "    return jsonify({\n"
        "        'openapi': '3.1.0',\n"
        "        'paths': {'/api/orders/': {'get': {\n"
        "            'operationId': 'list_orders',\n"
        "            'responses': {'200': {'content': {\n"
        "                'application/json': {'schema': {\n"
        "                    'type': 'array',\n"
        "                    'items': {'type': 'object', 'properties': {\n"
        "                        'id': {'type': 'string'}\n"
        "                    }}\n"
        "                }}\n"
        "            }}}\n"
        "        }}}\n"
        "    })\n",
        encoding="utf-8",
    )

    contracts = get_flask_endpoint_contracts(
        sources=(
            FlaskAppSource(
                name="commerce",
                app="app:app",
                path_prefixes=["/api/"],
            ),
        ),
        project_root=tmp_path,
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]
    assert capsys.readouterr().out == ""
