from __future__ import annotations

from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel


class ProductRow(BaseModel):
    id: int
    sku: str
    name: str
    category_id: int
    category_name: str
    unit_price: Decimal
    active: bool


class StockRecordRow(BaseModel):
    id: int
    product_id: int
    product_sku: str
    location_id: int
    location_name: str
    quantity_on_hand: int
    reorder_point: int


class OrderRow(BaseModel):
    id: int
    order_number: str
    store_id: int
    store_name: str
    status: str
    total_amount: Decimal


class SalesSummaryRow(BaseModel):
    label: str
    total_orders: int
    total_amount: Decimal


app = FastAPI(title="Retail Ops Fixture")
catalog = APIRouter(prefix="/retail/catalog", tags=["catalog"])
inventory = APIRouter(prefix="/retail/inventory", tags=["inventory"])
sales = APIRouter(prefix="/retail/sales", tags=["sales"])
reports = APIRouter(prefix="/retail/reports", tags=["reports"])


@catalog.get("/products/", operation_id="list_products", response_model=list[ProductRow])
def list_products(
    category_id: int | None = None,
    active: bool | None = None,
    ordering: Literal["sku", "-sku", "name", "-name", "unit_price", "-unit_price"]
    | None = None,
) -> list[ProductRow]:
    del category_id, active, ordering
    return [
        ProductRow(
            id=1,
            sku="TSHIRT-001",
            name="Classic T-Shirt",
            category_id=1,
            category_name="Apparel",
            unit_price=Decimal("24.00"),
            active=True,
        )
    ]


@catalog.get("/products/{product_id}/", operation_id="get_product", response_model=ProductRow)
def get_product(product_id: int) -> ProductRow:
    return ProductRow(
        id=product_id,
        sku="TSHIRT-001",
        name="Classic T-Shirt",
        category_id=1,
        category_name="Apparel",
        unit_price=Decimal("24.00"),
        active=True,
    )


@inventory.get(
    "/stock-records/",
    operation_id="list_stock_records",
    response_model=list[StockRecordRow],
)
def list_stock_records(
    product_id: int | None = None,
    location_id: int | None = None,
    below_reorder_point: bool | None = None,
) -> list[StockRecordRow]:
    del product_id, below_reorder_point
    location = location_id or 1
    return [
        StockRecordRow(
            id=1,
            product_id=1,
            product_sku="TSHIRT-001",
            location_id=location,
            location_name="ABC Mall Floor",
            quantity_on_hand=7,
            reorder_point=10,
        )
    ]


@sales.get("/orders/", operation_id="list_orders", response_model=list[OrderRow])
def list_orders(
    store_id: int | None = None,
    status: Literal["open", "paid", "cancelled"] | None = None,
) -> list[OrderRow]:
    del store_id, status
    return [
        OrderRow(
            id=1,
            order_number="SO-1001",
            store_id=1,
            store_name="ABC Mall",
            status="paid",
            total_amount=Decimal("48.00"),
        )
    ]


@sales.post("/orders/{order_id}/cancel/", operation_id="cancel_order")
def cancel_order(order_id: int) -> dict[str, int | str]:
    return {"id": order_id, "status": "cancelled"}


@reports.get(
    "/sales-summary/",
    operation_id="list_sales_summary",
    response_model=list[SalesSummaryRow],
)
def list_sales_summary(
    start_date: str | None = None,
    end_date: str | None = None,
    store_id: int | None = None,
    group_by: Literal["store", "day", "status"] | None = None,
) -> list[SalesSummaryRow]:
    del start_date, end_date, store_id, group_by
    return [SalesSummaryRow(label="ABC Mall", total_orders=1, total_amount=Decimal("48.00"))]


app.include_router(catalog)
app.include_router(inventory)
app.include_router(sales)
app.include_router(reports)

