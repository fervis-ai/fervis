from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model

from fervis.host_api.adapters.django.adapter import DjangoHostApiAdapter
from fervis.host_api.adapters.django.catalog import clear_endpoint_contract_cache
from fervis.host_api.adapters.fastapi.adapter import FastAPIHostApiAdapter
from fervis.host_api.adapters.flask.adapter import FlaskHostApiAdapter
from fervis.host_api.contracts import (
    EndpointContract,
    ReadAuthority,
    ReadContextRef,
    ReadInvocation,
)
from fervis.host_api.contracts.ports import EndpointExecutionError, HostApiAdapter
from fervis.project.integration import FastAPIAppSource, FlaskAppSource
from fervis.project.source_scope import DjangoSourceScope
from tests.fixtures.django_drf_retail_ops.catalog.models import Category, Product
from tests.fixtures.django_drf_retail_ops.fulfillment.models import StockTransfer
from tests.fixtures.django_drf_retail_ops.inventory.models import (
    InventoryLocation,
    StockRecord,
    Store,
)
from tests.fixtures.django_drf_retail_ops.sales.models import Order, OrderItem
from tests.testkit.django import SEEDED_USER_PK

pytestmark = pytest.mark.django_db


@dataclass(frozen=True)
class RetailCatalogCase:
    adapter: HostApiAdapter
    expected_route_names: frozenset[str]
    product_list_endpoint: str
    product_detail_endpoint: str
    product_detail_path_template: str
    product_detail_path_params: dict[str, str]
    order_list_endpoint: str
    stock_records_endpoint: str
    sales_summary_endpoint: str
    expected_resource_names: dict[str, frozenset[str]]
    read_authority: ReadAuthority
    seed: Callable[[], int]
    nested_response_paths: frozenset[str] = frozenset()


@pytest.fixture(
    params=("django", "fastapi", "flask"),
)
def retail_catalog_case(request) -> RetailCatalogCase:
    if request.param == "django":
        case = _django_retail_case()
    elif request.param == "fastapi":
        case = _fastapi_retail_case()
    else:
        case = _flask_retail_case()
    request.addfinalizer(case.adapter.close)
    return case


def test_retail_catalog_discovers_generic_read_routes(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    contracts = _contracts_by_name(retail_catalog_case.adapter)

    assert retail_catalog_case.expected_route_names.issubset(contracts)


def test_retail_catalog_preserves_path_params_and_declared_query_params(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    contracts = _contracts_by_name(retail_catalog_case.adapter)
    product_detail = contracts[retail_catalog_case.product_detail_endpoint]
    sales_summary = contracts[retail_catalog_case.sales_summary_endpoint]

    assert product_detail.path_template == (
        retail_catalog_case.product_detail_path_template
    )
    assert {item.name: item.type for item in product_detail.path_params} == (
        retail_catalog_case.product_detail_path_params
    )
    assert {item.name for item in sales_summary.query_params} == {
        "start_date",
        "end_date",
        "store_id",
        "group_by",
    }


def test_django_catalog_does_not_infer_identity_from_imperative_queryset() -> None:
    contracts = _contracts_by_name(_django_retail_case().adapter)
    order_list = contracts["list_orders_list"]
    params = {param.name: param for param in order_list.query_params}

    assert params["store_id"].entity_target is None


def test_retail_catalog_preserves_response_fields(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    contracts = _contracts_by_name(retail_catalog_case.adapter)
    order_contract = contracts[retail_catalog_case.order_list_endpoint]

    response_paths = {field.path for field in order_contract.response_fields}
    assert {"store_id", "store_name"}.issubset(response_paths)
    assert retail_catalog_case.nested_response_paths.issubset(response_paths)


def test_retail_catalog_exposes_semantic_resource_names_for_grounding(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    contracts = _contracts_by_name(retail_catalog_case.adapter)

    for (
        endpoint_name,
        expected_names,
    ) in retail_catalog_case.expected_resource_names.items():
        assert expected_names.issubset(set(contracts[endpoint_name].resource_names))
        catalog_endpoint = contracts[endpoint_name].catalog_endpoint
        if catalog_endpoint is None:
            raise AssertionError(
                f"{endpoint_name} is missing catalog endpoint metadata"
            )
        assert expected_names.issubset(set(catalog_endpoint.domain_resource_names))


def test_retail_catalog_does_not_invent_undeclared_filters(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    product_contract = _contracts_by_name(retail_catalog_case.adapter)[
        retail_catalog_case.product_list_endpoint
    ]

    assert {item.name for item in product_contract.query_params} == {
        "active",
        "category_id",
        "ordering",
    }


def test_django_catalog_excludes_optional_full_response_projection_control() -> None:
    product_contract = _contracts_by_name(_django_retail_case().adapter)[
        "list_products_list"
    ]

    assert "fields" not in {param.name for param in product_contract.query_params}


def test_retail_catalog_excludes_custom_post_actions(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    endpoint_names = set(_contracts_by_name(retail_catalog_case.adapter))

    assert all("cancel" not in endpoint_name for endpoint_name in endpoint_names)
    assert all("receive" not in endpoint_name for endpoint_name in endpoint_names)


def test_retail_read_execution_executes_get_with_declared_query_params(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    location_id = retail_catalog_case.seed()

    result = retail_catalog_case.adapter.execute_read(
        authority=retail_catalog_case.read_authority,
        invocation=ReadInvocation(
            endpoint_name=retail_catalog_case.stock_records_endpoint,
            query_params={"location_id": location_id},
        ),
    )

    assert result.response_status == 200
    assert [row["location_id"] for row in result.response_body] == [location_id]


def test_retail_read_execution_blocks_undeclared_query_params(
    retail_catalog_case: RetailCatalogCase,
) -> None:
    retail_catalog_case.seed()

    with pytest.raises(EndpointExecutionError, match="Unknown query params"):
        retail_catalog_case.adapter.execute_read(
            authority=retail_catalog_case.read_authority,
            invocation=ReadInvocation(
                endpoint_name=retail_catalog_case.product_list_endpoint,
                query_params={"store_id": "not-declared"},
            ),
        )


def _django_retail_case() -> RetailCatalogCase:
    clear_endpoint_contract_cache()
    return RetailCatalogCase(
        adapter=DjangoHostApiAdapter(
            sources=(
                DjangoSourceScope(
                    name="retail",
                    app_modules=("tests.fixtures.django_drf_retail_ops",),
                    path_prefixes=("/retail/",),
                ),
            )
        ),
        expected_route_names=frozenset(
            {
                "list_products_list",
                "get_products_detail",
                "list_orders_list",
                "get_orders_detail",
                "list_sales_summary",
                "list_low_stock",
            }
        ),
        product_list_endpoint="list_products_list",
        product_detail_endpoint="get_products_detail",
        product_detail_path_template="/retail/catalog/products/{pk}/",
        product_detail_path_params={"pk": "string"},
        order_list_endpoint="list_orders_list",
        stock_records_endpoint="list_stock_records_list",
        sales_summary_endpoint="list_sales_summary",
        expected_resource_names={
            "list_products_list": frozenset({"product"}),
            "list_orders_list": frozenset({"order"}),
            "list_stock_records_list": frozenset({"stock record"}),
            "list_sales_summary": frozenset({"sales summary"}),
        },
        read_authority=_django_read_authority(),
        seed=_seed_django_retail_ops_data,
        nested_response_paths=frozenset({"items", "items.product_sku"}),
    )


def _fastapi_retail_case() -> RetailCatalogCase:
    return RetailCatalogCase(
        adapter=FastAPIHostApiAdapter(
            sources=(
                FastAPIAppSource(
                    name="retail",
                    import_paths=["tests.fixtures.fastapi_retail_ops.main:app"],
                    path_prefixes=["/retail/"],
                ),
            ),
            project_root=Path.cwd(),
        ),
        expected_route_names=_openapi_route_names(),
        product_list_endpoint="list_products",
        product_detail_endpoint="get_product",
        product_detail_path_template="/retail/catalog/products/{product_id}/",
        product_detail_path_params={"product_id": "integer"},
        order_list_endpoint="list_orders",
        stock_records_endpoint="list_stock_records",
        sales_summary_endpoint="list_sales_summary",
        expected_resource_names=_openapi_resource_names(),
        read_authority=_anonymous_read_authority(),
        seed=lambda: 1,
    )


def _flask_retail_case() -> RetailCatalogCase:
    return RetailCatalogCase(
        adapter=FlaskHostApiAdapter(
            sources=(
                FlaskAppSource(
                    name="retail",
                    app="tests.fixtures.flask_retail_ops.main:create_app",
                    path_prefixes=["/retail/"],
                ),
            ),
            project_root=Path.cwd(),
        ),
        expected_route_names=_openapi_route_names(),
        product_list_endpoint="list_products",
        product_detail_endpoint="get_product",
        product_detail_path_template="/retail/catalog/products/{product_id}/",
        product_detail_path_params={"product_id": "integer"},
        order_list_endpoint="list_orders",
        stock_records_endpoint="list_stock_records",
        sales_summary_endpoint="list_sales_summary",
        expected_resource_names=_openapi_resource_names(),
        read_authority=_anonymous_read_authority(),
        seed=lambda: 1,
    )


def _openapi_route_names() -> frozenset[str]:
    return frozenset(
        {
            "list_products",
            "get_product",
            "list_orders",
            "list_stock_records",
            "list_sales_summary",
        }
    )


def _openapi_resource_names() -> dict[str, frozenset[str]]:
    return {
        "list_products": frozenset({"products"}),
        "list_orders": frozenset({"orders"}),
        "list_stock_records": frozenset({"stock records"}),
        "list_sales_summary": frozenset({"sales summary"}),
    }


def _seed_django_retail_ops_data() -> int:
    user_model = get_user_model()
    user_model._default_manager.update_or_create(
        pk=SEEDED_USER_PK,
        defaults={"username": "retail-contract-user", "is_active": True},
    )
    apparel = Category.objects.create(name="Apparel")
    product = Product.objects.create(
        sku="TSHIRT-001",
        name="Classic T-Shirt",
        category=apparel,
        unit_price=Decimal("24.00"),
    )
    mall = Store.objects.create(name="ABC Mall", region="Kampala")
    store_location = InventoryLocation.objects.create(
        name="ABC Mall Floor",
        kind="store",
        store=mall,
    )
    warehouse = InventoryLocation.objects.create(
        name="Central Warehouse",
        kind="warehouse",
    )
    StockRecord.objects.create(
        product=product,
        location=store_location,
        quantity_on_hand=7,
        reorder_point=10,
    )
    StockRecord.objects.create(
        product=product,
        location=warehouse,
        quantity_on_hand=50,
        reorder_point=20,
    )
    order = Order.objects.create(
        order_number="SO-1001",
        store=mall,
        status="paid",
        ordered_at=date(2026, 6, 1),
        total_amount=Decimal("48.00"),
    )
    OrderItem.objects.create(
        order=order,
        product=product,
        quantity=2,
        unit_price=Decimal("24.00"),
    )
    StockTransfer.objects.create(
        reference="TR-1001",
        product=product,
        from_location=warehouse,
        to_location=store_location,
        quantity=12,
        status="in_transit",
        requested_at=date(2026, 6, 2),
    )
    return store_location.id


def _contracts_by_name(adapter: HostApiAdapter) -> dict[str, EndpointContract]:
    return {contract.endpoint_name: contract for contract in adapter.describe_sources()}


def _django_read_authority() -> ReadAuthority:
    return ReadAuthority(
        tenant_id="tenant_1",
        read_context_ref=ReadContextRef(
            scheme="django_principal",
            key=str(SEEDED_USER_PK),
        ),
    )


def _anonymous_read_authority() -> ReadAuthority:
    return ReadAuthority(
        tenant_id="tenant_1",
        read_context_ref=ReadContextRef(scheme="anonymous"),
    )
