from django.urls import include, path

app_name = "retail_ops"

urlpatterns = [
    path("catalog/", include("tests.fixtures.django_drf_retail_ops.catalog.urls")),
    path("inventory/", include("tests.fixtures.django_drf_retail_ops.inventory.urls")),
    path("sales/", include("tests.fixtures.django_drf_retail_ops.sales.urls")),
    path(
        "fulfillment/",
        include("tests.fixtures.django_drf_retail_ops.fulfillment.urls"),
    ),
    path("reports/", include("tests.fixtures.django_drf_retail_ops.reports.urls")),
]
