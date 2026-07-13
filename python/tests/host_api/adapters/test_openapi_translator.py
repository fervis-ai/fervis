from __future__ import annotations

from fervis.host_api.adapters.openapi import endpoint_contracts_from_openapi


def test_openapi_contract_translation_preserves_declared_nested_relation_identity() -> (
    None
):
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/items/": {
                    "get": {
                        "operationId": "list_items",
                        "x-fervis": {
                            "pagination": {
                                "kind": "page_number",
                                "positionQueryParam": "page",
                                "pageSizeQueryParam": "per_page",
                                "resultsPath": "data",
                                "pageSize": 50,
                                "maxPageSize": 1000,
                                "totalPath": "count",
                            },
                            "candidateKeys": [
                                {
                                    "keyId": "primary_key",
                                    "entityKind": "item",
                                    "components": [
                                        {
                                            "componentId": "item_id",
                                            "fieldPath": "data.id",
                                        }
                                    ],
                                    "primary": True,
                                }
                            ],
                            "entityReferences": [
                                {
                                    "referenceId": "owner_reference",
                                    "targetEntityKind": "user",
                                    "targetKeyId": "primary_key",
                                    "components": [
                                        {
                                            "targetComponentId": "user_id",
                                            "localFieldPath": "data.owner_id",
                                        }
                                    ],
                                }
                            ],
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "data": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {"type": "string"},
                                                            "owner_id": {
                                                                "type": "string"
                                                            },
                                                        },
                                                    },
                                                },
                                                "count": {"type": "integer"},
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        },
        source_name="inventory",
        import_path="app.main:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="fastapi",
        source_namespace_kind="fastapi_app",
        source_namespace_path=("inventory",),
    )

    contract = contracts[0]
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("data", "array"),
        ("data.id", "string"),
        ("data.owner_id", "string"),
        ("count", "integer"),
    ]
    assert contract.candidate_keys[0].components[0].field_path == "data.id"
    assert contract.pagination is not None
    assert contract.pagination.position_query_param == "page"
    assert contract.pagination.results_path == "data"
    assert (
        contract.entity_references[0].components[0].local_field_path == "data.owner_id"
    )


def test_openapi3_contract_translation_extracts_read_shape() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "tags": ["orders"],
                        "parameters": [
                            {
                                "name": "status",
                                "in": "query",
                                "required": False,
                                "schema": {
                                    "type": "string",
                                    "enum": ["open", "closed"],
                                },
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
                                                    "amount": {"type": "number"},
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        },
        source_name="commerce",
        import_path="app.main:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="fastapi",
        source_namespace_kind="fastapi_app",
        source_namespace_path=("commerce",),
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.endpoint_name == "list_orders"
    assert contract.path_template == "/api/orders/"
    assert contract.response_cardinality == "many"
    assert [(field.path, field.type) for field in contract.response_fields] == [
        ("id", "string"),
        ("amount", "decimal"),
    ]
    assert [
        (param.name, param.type, param.choices) for param in contract.query_params
    ] == [("status", "string", ("open", "closed"))]
    assert contract.catalog_endpoint is not None
    assert contract.catalog_endpoint.api_schema_operation_id == "list_orders"


def test_swagger2_contract_translation_applies_base_path_and_direct_schemas() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "swagger": "2.0",
            "basePath": "/api",
            "paths": {
                "/people/{person_id}": {
                    "parameters": [
                        {
                            "name": "person_id",
                            "in": "path",
                            "required": True,
                            "type": "integer",
                        }
                    ],
                    "get": {
                        "operationId": "people.read_one",
                        "parameters": [
                            {
                                "name": "include_notes",
                                "in": "query",
                                "type": "boolean",
                            }
                        ],
                        "responses": {
                            "200": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "person_id": {"type": "integer"},
                                        "name": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                }
            },
        },
        source_name="people",
        import_path="server:connex_app#swagger",
        path_prefixes=("/api/",),
        framework_kind="flask",
        source_namespace_kind="flask_blueprint",
        source_namespace_path=("people",),
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract.path_template == "/api/people/{person_id}"
    assert [(param.name, param.type) for param in contract.path_params] == [
        ("person_id", "integer")
    ]
    assert [(param.name, param.type) for param in contract.query_params] == [
        ("include_notes", "boolean")
    ]
    assert {field.path: field.type for field in contract.response_fields} == {
        "person_id": "integer",
        "name": "string",
    }


def test_openapi_contract_translation_resolves_local_refs() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/OrderList"
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "OrderList": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Order"},
                    },
                    "Order": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                    },
                }
            },
        },
        source_name="orders",
        import_path="app:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="fastapi",
        source_namespace_kind="fastapi_app",
        source_namespace_path=("orders",),
    )

    assert len(contracts) == 1
    assert contracts[0].response_cardinality == "many"
    assert {field.path: field.type for field in contracts[0].response_fields} == {
        "id": "string",
        "quantity": "integer",
    }


def test_openapi_contract_translation_resolves_parameter_refs() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "parameters": [
                            {"$ref": "#/components/parameters/StoreId"},
                        ],
                    }
                }
            },
            "components": {
                "parameters": {
                    "StoreId": {
                        "name": "store_id",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                }
            },
        },
        source_name="orders",
        import_path="app:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="fastapi",
        source_namespace_kind="fastapi_app",
        source_namespace_path=("orders",),
    )

    assert [(param.name, param.required) for param in contracts[0].query_params] == [
        ("store_id", True)
    ]


def test_openapi_contract_translation_treats_path_params_as_required() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/orders/{order_id}/": {
                    "get": {
                        "operationId": "get_order",
                        "parameters": [
                            {
                                "name": "order_id",
                                "in": "path",
                                "schema": {"type": "integer"},
                            },
                        ],
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"id": {"type": "integer"}},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        },
        source_name="orders",
        import_path="app:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="flask",
        source_namespace_kind="flask_blueprint",
        source_namespace_path=("orders",),
    )

    assert [(param.name, param.required) for param in contracts[0].path_params] == [
        ("order_id", True)
    ]


def test_openapi_detail_path_parameter_targets_matching_declared_key_component() -> (
    None
):
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/warehouses/{warehouse_id}": {
                    "get": {
                        "operationId": "get_warehouse",
                        "parameters": [
                            {
                                "name": "warehouse_id",
                                "in": "path",
                                "schema": {"type": "integer"},
                            }
                        ],
                        "x-fervis": {
                            "candidateKeys": [
                                {
                                    "keyId": "primary_key",
                                    "entityKind": "warehouse",
                                    "components": [
                                        {
                                            "componentId": "warehouse_id",
                                            "fieldPath": "warehouse.warehouse_id",
                                        }
                                    ],
                                    "primary": True,
                                }
                            ]
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "warehouse": {
                                                    "type": "object",
                                                    "properties": {
                                                        "warehouse_id": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        },
        source_name="warehouses",
        import_path="app:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="flask",
        source_namespace_kind="flask_blueprint",
        source_namespace_path=("warehouses",),
    )

    target = contracts[0].path_params[0].entity_target

    assert target is not None
    assert target.entity_kind == "warehouse"
    assert target.key_id == "primary_key"
    assert target.component_id == "warehouse_id"


def test_openapi_contract_translation_merges_simple_all_of_object_schema() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/orders/": {
                    "get": {
                        "operationId": "list_orders",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "allOf": [
                                                    {
                                                        "$ref": "#/components/schemas/BaseOrder"
                                                    },
                                                    {
                                                        "type": "object",
                                                        "properties": {
                                                            "amount": {"type": "number"}
                                                        },
                                                    },
                                                ]
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "BaseOrder": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    }
                }
            },
        },
        source_name="orders",
        import_path="app:app#openapi",
        path_prefixes=("/api/",),
        framework_kind="fastapi",
        source_namespace_kind="fastapi_app",
        source_namespace_path=("orders",),
    )

    assert {field.path: field.type for field in contracts[0].response_fields} == {
        "id": "string",
        "amount": "decimal",
    }


def test_openapi_contract_translation_filters_by_segment_prefix() -> None:
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/api/orders/": {"get": {"operationId": "list_orders"}},
                "/apiary/orders/": {"get": {"operationId": "apiary_orders"}},
            },
        },
        source_name="orders",
        import_path="app:app#openapi",
        path_prefixes=("/api",),
        framework_kind="fastapi",
        source_namespace_kind="fastapi_app",
        source_namespace_path=("orders",),
    )

    assert [contract.endpoint_name for contract in contracts] == ["list_orders"]


def test_openapi_contract_translation_generates_distinct_detail_endpoint_names() -> (
    None
):
    contracts = endpoint_contracts_from_openapi(
        {
            "openapi": "3.1.0",
            "paths": {
                "/books/": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"}
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
                "/books/{book_id}": {
                    "parameters": [
                        {
                            "name": "book_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "integer"},
                        }
                    ],
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"id": {"type": "integer"}},
                                        }
                                    }
                                }
                            }
                        },
                    },
                },
            },
        },
        source_name="books",
        import_path="app:app#openapi",
        path_prefixes=("/",),
        framework_kind="flask",
        source_namespace_kind="flask_blueprint",
        source_namespace_path=("books",),
    )

    assert [contract.endpoint_name for contract in contracts] == [
        "get_books",
        "get_books_by_book_id",
    ]
