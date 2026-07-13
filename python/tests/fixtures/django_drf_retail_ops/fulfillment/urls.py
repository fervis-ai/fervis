from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import ReceiveTransferAPIView, StockTransferViewSet

app_name = "fulfillment"

router = DefaultRouter()
router.register("transfers", StockTransferViewSet, basename="transfers")

urlpatterns = [
    path("", include(router.urls)),
    path(
        "transfers/<int:transfer_id>/receive/",
        ReceiveTransferAPIView.as_view(),
        name="receive-transfer",
    ),
]
