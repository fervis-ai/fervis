from __future__ import annotations

from flask import Flask, jsonify, request


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/retail/catalog/products/")
    def list_products():
        return jsonify(
            [
                {
                    "id": 1,
                    "sku": "TSHIRT-001",
                    "name": "Classic T-Shirt",
                    "category_id": 1,
                    "category_name": "Apparel",
                    "unit_price": "24.00",
                    "active": True,
                }
            ]
        )

    @app.get("/retail/catalog/products/<int:product_id>/")
    def get_product(product_id: int):
        return jsonify(
            {
                "id": product_id,
                "sku": "TSHIRT-001",
                "name": "Classic T-Shirt",
                "category_id": 1,
                "category_name": "Apparel",
                "unit_price": "24.00",
                "active": True,
            }
        )

    @app.get("/retail/inventory/stock-records/")
    def list_stock_records():
        location_id = int(request.args.get("location_id") or 1)
        return jsonify(
            [
                {
                    "id": 1,
                    "product_id": 1,
                    "product_sku": "TSHIRT-001",
                    "location_id": location_id,
                    "location_name": "ABC Mall Floor",
                    "quantity_on_hand": 7,
                    "reorder_point": 10,
                }
            ]
        )

    @app.get("/retail/sales/orders/")
    def list_orders():
        return jsonify(
            [
                {
                    "id": 1,
                    "order_number": "SO-1001",
                    "store_id": 1,
                    "store_name": "ABC Mall",
                    "status": "paid",
                    "total_amount": "48.00",
                }
            ]
        )

    @app.post("/retail/sales/orders/<int:order_id>/cancel/")
    def cancel_order(order_id: int):
        return jsonify({"id": order_id, "status": "cancelled"})

    @app.get("/retail/reports/sales-summary/")
    def list_sales_summary():
        return jsonify(
            [{"label": "ABC Mall", "total_orders": 1, "total_amount": "48.00"}]
        )

    @app.get("/retail/openapi.json")
    def openapi():
        return jsonify(_openapi_schema())

    return app


def _openapi_schema() -> dict[str, object]:
    return {
        "openapi": "3.1.0",
        "paths": {
            "/retail/catalog/products/": {
                "get": _operation(
                    "list_products",
                    ["catalog"],
                    _array_schema(_product_schema()),
                    query_params={
                        "category_id": {"type": "integer"},
                        "active": {"type": "boolean"},
                        "ordering": {
                            "type": "string",
                            "enum": [
                                "sku",
                                "-sku",
                                "name",
                                "-name",
                                "unit_price",
                                "-unit_price",
                            ],
                        },
                    },
                )
            },
            "/retail/catalog/products/{product_id}/": {
                "get": _operation(
                    "get_product",
                    ["catalog"],
                    _product_schema(),
                    path_params={"product_id": {"type": "integer"}},
                )
            },
            "/retail/inventory/stock-records/": {
                "get": _operation(
                    "list_stock_records",
                    ["inventory"],
                    _array_schema(_stock_record_schema()),
                    query_params={
                        "product_id": {"type": "integer"},
                        "location_id": {"type": "integer"},
                        "below_reorder_point": {"type": "boolean"},
                    },
                )
            },
            "/retail/sales/orders/": {
                "get": _operation(
                    "list_orders",
                    ["sales"],
                    _array_schema(_order_schema()),
                    query_params={
                        "store_id": {"type": "integer"},
                        "status": {
                            "type": "string",
                            "enum": ["open", "paid", "cancelled"],
                        },
                    },
                )
            },
            "/retail/reports/sales-summary/": {
                "get": _operation(
                    "list_sales_summary",
                    ["reports"],
                    _array_schema(_sales_summary_schema()),
                    query_params={
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                        "store_id": {"type": "integer"},
                        "group_by": {
                            "type": "string",
                            "enum": ["store", "day", "status"],
                        },
                    },
                )
            },
        },
    }


def _operation(
    operation_id: str,
    tags: list[str],
    response_schema: dict[str, object],
    *,
    path_params: dict[str, dict[str, object]] | None = None,
    query_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    params = [
        {"name": name, "in": "path", "required": True, "schema": schema}
        for name, schema in (path_params or {}).items()
    ]
    params.extend(
        {"name": name, "in": "query", "required": False, "schema": schema}
        for name, schema in (query_params or {}).items()
    )
    return {
        "operationId": operation_id,
        "tags": tags,
        "parameters": params,
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": response_schema,
                    }
                }
            }
        },
    }


def _array_schema(item_schema: dict[str, object]) -> dict[str, object]:
    return {"type": "array", "items": item_schema}


def _product_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "sku": {"type": "string"},
            "name": {"type": "string"},
            "category_id": {"type": "integer"},
            "category_name": {"type": "string"},
            "unit_price": {"type": "number"},
            "active": {"type": "boolean"},
        },
    }


def _stock_record_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "product_id": {"type": "integer"},
            "product_sku": {"type": "string"},
            "location_id": {"type": "integer"},
            "location_name": {"type": "string"},
            "quantity_on_hand": {"type": "integer"},
            "reorder_point": {"type": "integer"},
        },
    }


def _order_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "order_number": {"type": "string"},
            "store_id": {"type": "integer"},
            "store_name": {"type": "string"},
            "status": {"type": "string"},
            "total_amount": {"type": "number"},
        },
    }


def _sales_summary_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "total_orders": {"type": "integer"},
            "total_amount": {"type": "number"},
        },
    }
