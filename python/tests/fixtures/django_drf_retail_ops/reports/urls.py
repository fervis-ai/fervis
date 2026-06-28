from django.urls import path

from .views import LowStockAPIView, SalesSummaryAPIView

app_name = "reports"

urlpatterns = [
    path("sales-summary/", SalesSummaryAPIView.as_view(), name="sales-summary"),
    path("low-stock/", LowStockAPIView.as_view(), name="low-stock"),
]

